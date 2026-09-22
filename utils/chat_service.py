# utils/chat_service.py
"""
Service métier partagé pour le chatbot Puls-Events : classification de la requête,
recherche dans le vector store, construction du prompt, appel au LLM, et logging.
Ce module est indépendant de l'interface (Streamlit, API...) pour être réutilisable partout.
"""

import datetime
import logging
import time
from typing import Optional, Dict, Any, List

from mistralai.client import Mistral

from utils.config import MISTRAL_API_KEY, CHAT_MODEL, APP_NAME, SEARCH_K, MAX_CONTEXT_CHARS
from utils.chat_message import ChatMessage
from utils.vector_store import VectorStoreManager
from utils.query_classifier import QueryClassifier
from utils.database import log_interaction
from utils.gemini_client import gemini_complete  # branché temporairement à la place de Mistral pour la génération finale
from utils.formatting import format_date_fr

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Instances partagées, créées une seule fois au chargement du module ---
# (le chargement de l'index Faiss depuis le disque est l'opération coûteuse à éviter de répéter)
_vector_store = VectorStoreManager()
_query_classifier = QueryClassifier()
_mistral_client = Mistral(api_key=MISTRAL_API_KEY)


def get_vector_store() -> VectorStoreManager:
    """Expose l'instance partagée du vector store (utilisée par /rebuild pour la mettre à jour en place)."""
    return _vector_store


# Mots-clés signalant que l'utilisateur veut explicitement des événements PASSÉS.
# Détection déterministe (pas de LLM) : le tri passé/à venir par le LLM lui-même s'est montré
# peu fiable en pratique (ex: confond février et septembre) - le filtre réel est fait dans
# vector_store.search() via `include_past`, ceci ne fait que décider s'il faut l'autoriser.
_PAST_EVENT_KEYWORDS = [
    "passé", "passés", "passée", "passées",
    # "eu lieu"/"déroulé" plutôt que "a eu lieu"/"s'est déroulé" : ces formes exactes ne
    # matchaient pas l'inversion interrogative française ("a-t-il eu lieu", "s'est-il déroulé"),
    # bug réel trouvé sur "Quand le Festival des Solidarités a-t-il eu lieu ?" - non détecté comme
    # question sur le passé, alors que "a-t-il eu lieu" l'est sans ambiguïté.
    "eu lieu", "déroulé",
    "s'est passé", "s'est-il passé", "s'est-elle passée",
    "s'est tenu", "s'est-il tenu", "s'est-elle tenue",
    "l'an dernier", "l'année dernière", "le mois dernier", "la semaine dernière",
    "déjà eu", "historique",
]


def _wants_past_events(query: str) -> bool:
    """Détecte si la question porte explicitement sur des événements passés."""
    query_lower = query.lower()
    return any(kw in query_lower for kw in _PAST_EVENT_KEYWORDS)


def _build_system_prompt(mode: str, context_str: str = "") -> str:
    """
    Construit le prompt système selon le mode déterminé par le classificateur.

    Args:
        mode: "RAG_WITH_RESULTS", "RAG_NO_RESULTS" ou "DIRECT"
        context_str: contexte documentaire à injecter (uniquement pour RAG_WITH_RESULTS)
    """
    # Calculée en Python (jamais laissée à la "connaissance" du LLM, peu fiable sur la date courante)
    today_str = datetime.date.today().strftime("%A %d %B %Y")

    if mode == "RAG_WITH_RESULTS":
        return f"""Vous êtes {APP_NAME}, un assistant virtuel de recommandations d'événements culturels.
Nous sommes le {today_str}.
Répondez à la question de l'utilisateur en vous basant UNIQUEMENT sur le contexte fourni ci-dessous.
Le contexte a DÉJÀ été filtré par date en amont (par le code, pas par vous) : vous n'avez PAS besoin de
vérifier vous-même si un événement est passé ou à venir, ni d'exclure quoi que ce soit sur ce critère -
présentez simplement les événements du contexte tels quels.
Si l'information n'est pas dans le contexte, dites que vous ne savez pas ou que l'information n'est pas disponible.
Soyez concis et précis. Citez vos sources (nom de l'événement, ville, date, et le lien si pertinent).

Contexte fourni:
---
{context_str}
---
"""
    elif mode == "RAG_NO_RESULTS":
        return f"""Vous êtes {APP_NAME}, un assistant virtuel de recommandations d'événements culturels.
Nous sommes le {today_str}.
L'utilisateur a posé une question qui semble porter sur des événements culturels précis, mais aucun
événement pertinent n'a été trouvé dans la base de données.
Indiquez poliment que vous n'avez pas trouvé d'événement correspondant et suggérez de reformuler la question
(ex: préciser une ville, une date, un type d'événement).
N'inventez aucun événement.
"""
    else:  # DIRECT
        return f"""Vous êtes {APP_NAME}, un assistant virtuel de recommandations d'événements culturels.
Nous sommes le {today_str}.
Répondez à la question de l'utilisateur en utilisant vos connaissances générales.
Soyez concis, précis et utile.
Si la question concerne des événements culturels précis que vous ne connaissez pas, indiquez-le clairement.
N'inventez pas d'événements.
"""


def _truncate_by_context_budget(
    docs: List[Dict[str, Any]], max_chars: int = MAX_CONTEXT_CHARS
) -> List[Dict[str, Any]]:
    """
    Garde les documents (déjà triés du plus pertinent au moins pertinent) jusqu'à atteindre
    max_chars de texte cumulé, plutôt qu'un nombre de documents fixe.

    Pourquoi : un plafond en nombre de documents est arbitraire (voir SEARCH_K dans config.py) -
    il exclut des résultats pertinents sur une recherche filtrée (ville+mois réduit déjà le pool
    à une poignée d'événements, tous devraient passer) et n'empêche pas l'explosion sur une
    question large (des milliers de résultats franchissent le seuil de qualité minimal). Un
    budget de caractères s'adapte aux deux cas sans nombre choisi au hasard.

    Toujours au moins 1 document si `docs` n'est pas vide, même s'il dépasse le budget à lui seul.
    """
    if not docs:
        return []
    kept = [docs[0]]
    total_chars = len(docs[0]["text"])
    for doc in docs[1:]:
        total_chars += len(doc["text"])
        if total_chars > max_chars:
            break
        kept.append(doc)
    return kept


def _search_with_past_fallback(
    query: str,
    num_docs: int,
    min_score: Optional[float],
    city: Optional[str],
    region: Optional[str],
    month: Optional[int],
    year: Optional[int],
    include_past: bool,
) -> tuple:
    """
    Cherche `query`. Si include_past=False et qu'AUCUN résultat futur ne passe le seuil de
    qualité, on retente en incluant le passé plutôt que de répondre "rien trouvé" - un événement
    passé pertinent (clairement signalé comme tel au LLM) vaut mieux qu'une absence de réponse.

    Règle volontairement simple (décision explicite) : le repli ne se déclenche QUE si la
    recherche future est vide, jamais sur une simple comparaison de score. Une version précédente
    comparait les scores (avec une marge) et retombait parfois sur des événements passés alors que
    des résultats futurs pertinents existaient déjà (ex: 45 résultats futurs valides écartés au
    profit de 2 concerts déjà terminés) - une règle basée sur le score reste fragile face à ce
    genre de cas, "aucun résultat futur" est un signal sans ambiguïté.

    Returns:
        (résultats, repli_utilisé) - repli_utilisé indique si le LLM doit être prévenu que le
        contexte peut contenir des événements déjà passés.
    """
    docs = _vector_store.search(
        query, k=num_docs, min_score=min_score, city=city, region=region,
        month=month, year=year, include_past=include_past,
    )
    if include_past or docs:
        return docs, False

    candidates_with_past = _vector_store.search(
        query, k=num_docs, min_score=min_score, city=city, region=region,
        month=month, year=year, include_past=True,
    )
    if candidates_with_past:
        logging.info(f"Aucun événement futur trouvé pour '{query}', repli sur le passé.")
        return candidates_with_past, True
    return docs, False


def answer_query(
    query: str,
    conversation_history: Optional[List[Dict[str, str]]] = None,
    num_docs: int = SEARCH_K,
    min_score: Optional[float] = 0.75,
    model: str = CHAT_MODEL,
) -> Dict[str, Any]:
    """
    Traite une question utilisateur de bout en bout : reformulation avec l'historique,
    classification, recherche documentaire éventuelle, génération de la réponse, et
    enregistrement en base.

    Args:
        query: question posée par l'utilisateur
        conversation_history: tours précédents de la conversation, format [{"role": "user"/"assistant", "content": "..."}],
                               du plus ancien au plus récent (sans inclure `query`). Optionnel.
        num_docs: nombre max de documents à récupérer si RAG est nécessaire
        min_score: score de similarité minimum (0-1) pour garder un résultat de recherche
        model: modèle Mistral à utiliser pour la génération

    Returns:
        dict avec les clés: response, sources, mode, confidence, reason, interaction_id
    """
    conversation_history = conversation_history or []

    # 0. Reformuler la question en question autonome à partir de l'historique
    # (ex: "et pas trop loin ?" -> "événements de volley pas trop loin" si le sujet était le volley)
    search_query = _query_classifier.reformulate_query(query, conversation_history)
    if search_query != query:
        logging.info(f"Question reformulée avec l'historique: '{query}' -> '{search_query}'")

    # 1. Classifier la requête (reformulée) : a-t-on besoin d'aller chercher des événements ?
    needs_rag, confidence, reason = _query_classifier.needs_rag(search_query)
    logging.info(f"Classification: {'RAG' if needs_rag else 'DIRECT'} (confiance={confidence:.2f}) - {reason}")

    retrieved_docs = []
    detected_city = None
    detected_region = None
    used_past_fallback = False
    if needs_rag:
        detected_city = _query_classifier.extract_city(search_query)
        # La région n'est cherchée que si aucune ville n'a été détectée : une ville est plus
        # précise qu'une région, pas besoin des deux (et ça évite un appel LLM inutile).
        if not detected_city:
            detected_region = _query_classifier.extract_region(search_query)
        detected_month, detected_year = _query_classifier.extract_month(search_query)
        include_past = _wants_past_events(search_query)
        logging.info(
            f"Ville: {detected_city} | Région: {detected_region} | Mois: {detected_month}/{detected_year} | "
            f"Événements passés autorisés: {include_past}"
        )

        # Décompose en plusieurs thèmes si la question en combine plusieurs distincts (ex: "sport
        # et musique") - un seul embedding de toute la phrase dilue chaque thème (vérifié en
        # pratique : une recherche unique sur une phrase à plusieurs critères ne retrouve ni l'un
        # ni l'autre correctement, juste un mélange flou). Retourne [search_query] si un seul thème.
        facets = _query_classifier.extract_search_facets(search_query)
        if len(facets) > 1:
            logging.info(f"Question décomposée en {len(facets)} facettes: {facets}")

        # Une recherche par facette (chacune avec son propre filet de sécurité passé/à venir,
        # voir _search_with_past_fallback), puis fusion : dédupliquées par URL, triées par score.
        seen_urls = set()
        merged_docs = []
        for facet in facets:
            docs, fallback_used = _search_with_past_fallback(
                facet, num_docs, min_score, detected_city, detected_region, detected_month,
                detected_year, include_past
            )
            used_past_fallback = used_past_fallback or fallback_used
            for doc in docs:
                url = doc["metadata"].get("url")
                if url not in seen_urls:
                    seen_urls.add(url)
                    merged_docs.append(doc)

        merged_docs.sort(key=lambda d: d["score"], reverse=True)
        # Deux garde-fous combinés : num_docs reste un plafond de sécurité en nombre de documents,
        # mais la limite qui joue réellement en pratique est le budget de contexte (voir
        # _truncate_by_context_budget) - un chiffre de documents fixe est arbitraire, un budget de
        # texte s'adapte à la taille réelle du pool filtré.
        retrieved_docs = _truncate_by_context_budget(merged_docs)[:num_docs]

    # 2. Construire le prompt système et le contexte selon ce qui a été trouvé
    if needs_rag and retrieved_docs:
        # On inclut la ville/date/lien de chaque événement dans le contexte pour que le LLM puisse les citer.
        # La date est reformatée en français AVANT d'être donnée au LLM (déterministe, comme le filtre
        # passé/à venir) plutôt que de lui demander de la reformater lui-même, ce qui s'est montré peu fiable.
        context_str = "\n\n---\n\n".join([
            f"Événement: {doc['metadata'].get('filename', 'Inconnu')} "
            f"(Ville: {doc['metadata'].get('ville', 'N/A')}, Date: {format_date_fr(doc['metadata'].get('date'))}, "
            f"Lien: {doc['metadata'].get('url', 'N/A')}, Score: {doc['score']:.1f}%)\n"
            f"Contenu: {doc['text']}"
            for doc in retrieved_docs
        ])
        if used_past_fallback:
            # Ici le contexte N'A PAS été filtré par date (contrairement à ce que dit le prompt
            # normal) - ces événements peuvent être passés, on le signale explicitement au LLM.
            context_str = (
                "ATTENTION : aucun événement à venir trouvé, les événements ci-dessous incluent "
                "peut-être des événements déjà passés (comparez vous-même chaque date à aujourd'hui, "
                "et précisez clairement dans votre réponse si l'événement demandé est déjà terminé).\n\n"
                + context_str
            )
        system_prompt = _build_system_prompt("RAG_WITH_RESULTS", context_str)
        sources_for_log = [
            {"text": doc["text"], "metadata": doc["metadata"], "score": doc["score"]}
            for doc in retrieved_docs
        ]
    elif needs_rag:
        # RAG jugé nécessaire mais aucun résultat au-dessus du score minimum
        system_prompt = _build_system_prompt("RAG_NO_RESULTS")
        sources_for_log = []
    else:
        system_prompt = _build_system_prompt("DIRECT")
        sources_for_log = []

    # 3. Appeler le LLM, en incluant l'historique récent pour une réponse naturellement continue
    # (les messages sont convertis en dict via .format(), voir bug corrigé dans query_classifier.py)
    messages = [
        ChatMessage(role="system", content=system_prompt).format(),
        *[ChatMessage(role=m["role"], content=m["content"]).format() for m in conversation_history[-10:]],
        ChatMessage(role="user", content=query).format(),
    ]
    # --- MISTRAL (désactivé temporairement, compte bloqué sur chat/completions) ---
    # max_retries = 5
    # chat_response = None
    # for attempt in range(max_retries):
    #     try:
    #         chat_response = _mistral_client.chat.complete(model=model, messages=messages)
    #         break
    #     except Exception as e:
    #         status_code = VectorStoreManager._get_status_code(e)  # réutilise la détection déjà écrite pour l'indexation
    #         if status_code == 429 and attempt < max_retries - 1:
    #             wait_time = 2 ** (attempt + 1)  # 2s, 4s, 8s, 16s, 32s
    #             logging.warning(
    #                 f"Rate limit (429) sur la génération finale, nouvelle tentative dans "
    #                 f"{wait_time}s (essai {attempt + 1}/{max_retries})..."
    #             )
    #             time.sleep(wait_time)
    #             continue
    #         raise  # pas un 429, ou plus de tentatives disponibles : on relance l'exception
    # response_text = chat_response.choices[0].message.content

    # --- GEMINI (actif) : même format de messages, gemini_complete() gère déjà son propre
    # retry sur les erreurs serveur transitoires (503), pas besoin de le dupliquer ici ---
    response_text = gemini_complete(messages)

    # 4. Logger l'interaction (question, réponse, sources, métadonnées de classification)
    interaction_id = log_interaction(
        query=query,
        response=response_text,
        sources=sources_for_log,
        metadata={"mode": "RAG" if needs_rag else "DIRECT", "confidence": confidence, "reason": reason},
    )

    return {
        "response": response_text,
        "sources": sources_for_log,
        "mode": "RAG" if needs_rag else "DIRECT",
        "confidence": confidence,
        "reason": reason,
        "interaction_id": interaction_id,
    }
