# utils/query_classifier.py
"""
Module de classification des requêtes pour déterminer si une question nécessite RAG
"""

import re
import datetime
import logging
from typing import Dict, List, Tuple, Optional
from mistralai.client import Mistral
from utils.chat_message import ChatMessage
from utils.gemini_client import gemini_complete  # branché temporairement à la place de Mistral (voir chaque appel ci-dessous)

from utils.config import MISTRAL_API_KEY, CHAT_MODEL, APP_NAME

class QueryClassifier:
    """
    Classe pour classifier les requêtes et déterminer si elles nécessitent RAG
    """

    def __init__(self):
        """
        Initialise le classificateur de requêtes
        """
        self.mistral_client = Mistral(api_key=MISTRAL_API_KEY) if MISTRAL_API_KEY else None

        # Mots-clés liés aux événements culturels qui suggèrent un besoin de RAG
        self.event_keywords = [
            "événement", "événements", "evenement", "evenements",
            "concert", "concerts", "festival", "festivals",
            "spectacle", "spectacles", "exposition", "expositions",
            "sortie", "sorties", "agenda", "programme", "programmation",
            "culturel", "culturelle", "culturels", "culturelles",
            "musée", "musee", "théâtre", "theatre", "cinéma", "cinema",
            "danse", "atelier", "ateliers", "conférence", "conference",
            "animation", "animations", "fête", "fete", "salon",
            "billetterie", "réserver", "reserver", "réservation", "reservation",
            "ce soir", "ce week-end", "cette semaine", "ce weekend",
            "où sortir", "quoi faire", "quand a lieu", "à quelle heure",
        ]

        # Questions générales qui ne nécessitent pas de RAG
        self.general_patterns = [
            r"^(bonjour|salut|hello|coucou|hey|bonsoir)[\s\.,!]*$",
            r"^(merci|thanks|thank you|je te remercie)[\s\.,!]*$",
            r"^(comment ça va|ça va|comment vas-tu|comment allez-vous)[\s\.,!?]*$",
            r"^(au revoir|bye|à bientôt|à plus tard|à la prochaine)[\s\.,!]*$",
            r"^(qui es[- ]tu|qu'es[- ]tu|que fais[- ]tu|comment fonctionnes[- ]tu|tu es quoi)[\s\?]*$",
            r"^(aide|help|sos|besoin d'aide)[\s\.,!?]*$"
        ]

    def needs_rag(self, query: str) -> Tuple[bool, float, str]:
        """
        Détermine si une requête nécessite RAG

        Args:
            query: Requête de l'utilisateur

        Returns:
            Tuple (besoin_rag, confiance, raison)
        """
        # Convertir la requête en minuscules pour la comparaison
        query_lower = query.lower()

        # 1. Vérifier les patterns de questions générales (salutations, remerciements, etc.)
        for pattern in self.general_patterns:
            if re.match(pattern, query_lower):
                return False, 0.95, "Question générale ou salutation"

        # 2. Vérifier la présence de mots-clés liés aux événements culturels
        event_keywords_found = [kw for kw in self.event_keywords if kw in query_lower]
        if event_keywords_found:
            keywords_str = ", ".join(event_keywords_found)
            return True, 0.9, f"Contient des mots-clés liés aux événements: {keywords_str}"

        # 3. Utiliser le LLM pour les cas ambigus
        if self.mistral_client:
            return self._classify_with_llm(query)

        # Par défaut, utiliser RAG pour les questions longues (plus de 5 mots)
        words = query.split()
        if len(words) > 5:
            return True, 0.6, "Question complexe (plus de 5 mots)"

        # Par défaut, ne pas utiliser RAG
        return False, 0.5, "Aucun critère spécifique détecté"

    def _classify_with_llm(self, query: str) -> Tuple[bool, float, str]:
        """
        Utilise le LLM pour classifier la requête

        Args:
            query: Requête de l'utilisateur

        Returns:
            Tuple (besoin_rag, confiance, raison)
        """
        try:
            system_prompt = f"""Vous êtes un classificateur de requêtes pour {APP_NAME}, un chatbot de recommandations d'événements culturels.
Votre tâche est de déterminer si une question nécessite une recherche dans une base de données d'événements culturels à venir.

Répondez UNIQUEMENT par "RAG" ou "DIRECT" suivi d'une brève explication:
- "RAG" si la question porte sur des événements culturels concrets (concerts, expositions, festivals, spectacles, sorties, dates, lieux, horaires, etc.)
- "DIRECT" si c'est une question générale, une salutation, ou une question qui ne nécessite pas d'informations sur des événements précis.

Exemples:
Question: "Bonjour, comment ça va?"
Réponse: DIRECT - Simple salutation

Question: "Quels concerts ont lieu ce week-end à Bordeaux ?"
Réponse: RAG - Demande d'informations sur des événements concrets

Question: "Y a-t-il une expo photo en ce moment ?"
Réponse: RAG - Demande d'informations sur des événements concrets

Question: "Qu'est-ce que l'intelligence artificielle?"
Réponse: DIRECT - Question générale de connaissance
"""

            messages = [
                ChatMessage(role="system", content=system_prompt),
                ChatMessage(role="user", content=query)
            ]

            # --- MISTRAL (désactivé temporairement, compte bloqué sur chat/completions) ---
            # self.mistral_client.chat est un sous-objet (namespace), pas une fonction :
            # la méthode réelle d'appel est .chat.complete(). Elle attend des dicts,
            # d'où le .format() sur chaque ChatMessage (vérifié par test direct de l'API).
            # response = self.mistral_client.chat.complete(
            #     model=CHAT_MODEL,
            #     messages=[m.format() for m in messages],
            #     temperature=0.1,  # Température basse pour des réponses cohérentes
            #     max_tokens=50  # Réponse courte suffisante
            # )
            # result = response.choices[0].message.content.strip()

            # --- GEMINI (actif) : même format de messages, gemini_complete() fait la traduction ---
            result = gemini_complete(
                [m.format() for m in messages],
                temperature=0.1,  # Température basse pour des réponses cohérentes
                max_tokens=50,  # Réponse courte suffisante
            ).strip()

            logging.info(f"Classification LLM pour '{query}': {result}")

            # Analyser la réponse
            if result.startswith("RAG"):
                confidence = 0.85  # Confiance élevée dans la décision du LLM
                reason = result.replace("RAG - ", "").replace("RAG-", "").replace("RAG:", "").strip()
                return True, confidence, reason
            elif result.startswith("DIRECT"):
                confidence = 0.85
                reason = result.replace("DIRECT - ", "").replace("DIRECT-", "").replace("DIRECT:", "").strip()
                return False, confidence, reason
            else:
                # Réponse ambiguë, utiliser RAG par défaut
                return True, 0.6, "Classification ambiguë, utilisation de RAG par précaution"

        except Exception as e:
            logging.error(f"Erreur lors de la classification avec LLM: {e}")
            # En cas d'erreur, utiliser RAG par défaut
            return True, 0.5, f"Erreur de classification: {str(e)}"

    def extract_city(self, query: str) -> Optional[str]:
        """
        Extrait le nom de la ville mentionnée dans la question, via un appel léger au LLM.
        Retourne None si aucune ville n'est mentionnée, ou si le client Mistral n'est pas configuré.
        """
        if not self.mistral_client:
            return None
        try:
            system_prompt = """Extrayez le nom de la ville française mentionnée dans la question suivante, s'il y en a une.
Répondez UNIQUEMENT par le nom de la ville (ex: "Bordeaux"), sans aucune explication.
S'il n'y a aucune ville mentionnée, répondez UNIQUEMENT par: AUCUNE
"""
            messages = [
                ChatMessage(role="system", content=system_prompt),
                ChatMessage(role="user", content=query),
            ]
            # --- MISTRAL (désactivé temporairement) ---
            # response = self.mistral_client.chat.complete(
            #     model=CHAT_MODEL,
            #     messages=[m.format() for m in messages],
            #     temperature=0.0,
            #     max_tokens=20,
            # )
            # result = response.choices[0].message.content.strip()

            # --- GEMINI (actif) ---
            result = gemini_complete(
                [m.format() for m in messages],
                temperature=0.0,
                max_tokens=20,
            ).strip()

            if not result or result.upper() == "AUCUNE":
                return None
            return result
        except Exception as e:
            logging.error(f"Erreur lors de l'extraction de la ville: {e}")
            return None

    def extract_month(self, query: str) -> Tuple[Optional[int], Optional[int]]:
        """
        Extrait le mois ET l'année mentionnés dans la question, via un appel léger au LLM.

        Pourquoi l'année : "en octobre" dit en septembre 2026 signifie normalement "octobre 2026"
        (le prochain), pas "n'importe quel octobre" - sans l'année, le filtre attrapait aussi de
        vieux événements d'octobre 2025 (constaté en pratique). On donne la date du jour au LLM
        pour qu'il infère l'année la plus probable quand elle n'est pas explicite, plutôt que de
        coder nous-mêmes une règle "mois avant aujourd'hui -> année prochaine" fragile.

        Returns:
            (mois 1-12 ou None, année ou None) - année seule sans mois n'a pas de sens ici, les
            deux sont toujours renvoyés ensemble ou (None, None).
        """
        if not self.mistral_client:
            return None, None
        try:
            today_str = datetime.date.today().isoformat()
            system_prompt = f"""Nous sommes le {today_str}. Extrayez le mois ET l'année mentionnés ou
implicites dans la question suivante, UNIQUEMENT si elle cible un mois précis :
- Nom de mois explicite (ex: "en octobre") -> le mois à venir le plus proche portant ce nom,
  pas un mois déjà passé.
- Période clairement contenue dans un seul mois (ex: "ce week-end", "cette semaine",
  "aujourd'hui", "demain") -> le mois en cours.

Ne renvoyez AUCUN mois pour une période floue ou étalée sur plusieurs semaines/mois qui ne
correspond à AUCUN mois précis (ex: "dans les prochaines semaines", "bientôt", "prochainement",
"dans les mois à venir", "cette année") - un filtre sur un seul mois exclurait à tort des
événements pertinents d'un mois voisin.

Répondez UNIQUEMENT sous la forme "MOIS/ANNEE" (ex: "10/2026"), sans aucune explication.
Si aucun mois précis ne peut être déduit selon ces règles, répondez UNIQUEMENT par: AUCUN
"""
            messages = [
                ChatMessage(role="system", content=system_prompt),
                ChatMessage(role="user", content=query),
            ]
            result = gemini_complete([m.format() for m in messages], temperature=0.0, max_tokens=15).strip()
            if not result or result.upper() == "AUCUN" or "/" not in result:
                return None, None
            month_str, year_str = result.split("/", 1)
            month, year = int(month_str), int(year_str)
            if not (1 <= month <= 12):
                return None, None
            return month, year
        except Exception as e:
            logging.error(f"Erreur lors de l'extraction du mois/année: {e}")
            return None, None

    def extract_region(self, query: str) -> Optional[str]:
        """
        Extrait le nom de la RÉGION française mentionnée dans la question (distincte de la ville,
        voir extract_city), via un appel léger au LLM.
        Retourne None si aucune région n'est mentionnée.
        """
        if not self.mistral_client:
            return None
        try:
            system_prompt = """Extrayez le nom de la région française mentionnée dans la question suivante,
s'il y en a une (ex: "Auvergne-Rhône-Alpes", "Bretagne", "Nouvelle-Aquitaine"...). Ne confondez pas
avec une ville : "Lyon" n'est pas une région, mais "Rhône-Alpes" ou "Auvergne-Rhône-Alpes" en est une.
Répondez UNIQUEMENT par le nom OFFICIEL de la région, sans aucune explication.
S'il n'y a aucune région mentionnée, répondez UNIQUEMENT par: AUCUNE
"""
            messages = [
                ChatMessage(role="system", content=system_prompt),
                ChatMessage(role="user", content=query),
            ]
            result = gemini_complete([m.format() for m in messages], temperature=0.0, max_tokens=20).strip()
            if not result or result.upper() == "AUCUNE":
                return None
            return result
        except Exception as e:
            logging.error(f"Erreur lors de l'extraction de la région: {e}")
            return None

    def extract_search_facets(self, query: str) -> List[str]:
        """
        Décompose une question combinant plusieurs thèmes distincts (ex: "sport et musique") en
        plusieurs requêtes de recherche autonomes, une par thème.

        Pourquoi : un seul embedding de toute la phrase dilue chaque thème - vérifié en pratique,
        une recherche unique sur "sport et musique à Lyon en octobre" ne retrouve ni du sport ni
        de la musique de façon satisfaisante, juste un mélange flou lié à "Lyon".

        Deux points calibrés empiriquement (pas des suppositions) :
        - Si un seul thème est détecté, on renvoie TOUJOURS `query` inchangée (jamais la
          reformulation du LLM, même s'il devait la produire à l'identique) : on a mesuré que le
          LLM ne respecte pas toujours "renvoie telle quelle" et raccourcit quand même la
          question, ce qui fait BAISSER le score (ex: 82% avec la question complète contre 75%
          avec sa version raccourcie, sur les mêmes événements).
        - Quand plusieurs thèmes sont détectés, chaque facette doit être une VRAIE QUESTION
          naturelle, pas une expression compressée façon mots-clés : mesuré aussi, une phrase
          naturelle score ~5 points de plus qu'une expression du type "thème à ville en période".
        """
        if not self.mistral_client:
            return [query]
        try:
            system_prompt = """Analysez la question et identifiez si elle combine plusieurs THÈMES ou
CENTRES D'INTÉRÊT distincts (ex: "sport et musique", "expositions et concerts").

Si un seul thème (question simple), répondez avec UNE SEULE ligne : le mot UNIQUE.
Si plusieurs thèmes distincts, générez UNE ligne de recherche PAR THÈME, en gardant le contexte
commun (ville, période). Chaque ligne doit être une VRAIE QUESTION en français naturel, pas une
expression compressée façon mots-clés (ex: "Quels événements sportifs se déroulent à Lyon en
octobre ?", pas "événements sportifs Lyon octobre").

Répondez UNIQUEMENT avec les lignes de recherche (ou le mot UNIQUE), une par ligne, sans
numérotation ni explication.

Exemple:
Question: "Si j'aime le sport et la musique et que je suis à Lyon en octobre, qu'est-ce que tu me proposes ?"
Réponse:
Quels événements sportifs se déroulent à Lyon en octobre ?
Quels événements musicaux se déroulent à Lyon en octobre ?

Exemple:
Question: "Quels concerts de jazz à Paris ce week-end ?"
Réponse:
UNIQUE
"""
            messages = [
                ChatMessage(role="system", content=system_prompt),
                ChatMessage(role="user", content=query),
            ]
            result = gemini_complete([m.format() for m in messages], temperature=0.0, max_tokens=150)
            lines = [line.strip() for line in result.splitlines() if line.strip()]

            # Thème unique détecté (ou réponse vide/mal formée) : toujours la question ORIGINALE,
            # jamais une reformulation du LLM - voir la note de perf ci-dessus.
            if len(lines) <= 1:
                return [query]
            return lines
        except Exception as e:
            logging.error(f"Erreur lors de la décomposition en facettes: {e}")
            return [query]

    def reformulate_query(self, query: str, history: List[Dict[str, str]]) -> str:
        """
        Reformule la question en une question autonome intégrant le contexte de l'historique
        (ex: "et pas trop loin ?" + historique sur le volley -> "événements de volley pas trop loin").
        Retourne la question originale si pas d'historique, pas de client, ou en cas d'erreur.
        """
        if not history or not self.mistral_client:
            return query
        try:
            recent_history = history[-10:]  # limite pour ne pas gonfler le prompt indéfiniment
            history_str = "\n".join(f"{m['role']}: {m['content']}" for m in recent_history)
            system_prompt = f"""Voici l'historique récent d'une conversation avec un chatbot d'événements culturels.
Reformulez la DERNIÈRE question de l'utilisateur en une question autonome et complète, intégrant le contexte
nécessaire de l'historique (sujet, ville, type d'événement...) pour qu'elle ait du sens seule.
Si elle est déjà autonome, renvoyez-la telle quelle. Répondez UNIQUEMENT par la question reformulée.

Historique:
{history_str}

Dernière question: {query}
"""
            messages = [
                ChatMessage(role="system", content=system_prompt),
                ChatMessage(role="user", content=query),
            ]
            # --- MISTRAL (désactivé temporairement) ---
            # response = self.mistral_client.chat.complete(
            #     model=CHAT_MODEL,
            #     messages=[m.format() for m in messages],
            #     temperature=0.0,
            #     max_tokens=100,
            # )
            # reformulated = response.choices[0].message.content.strip()

            # --- GEMINI (actif) ---
            reformulated = gemini_complete(
                [m.format() for m in messages],
                temperature=0.0,
                max_tokens=100,
            ).strip()

            return reformulated or query
        except Exception as e:
            logging.error(f"Erreur lors de la reformulation de la requête: {e}")
            return query
