# evaluate_rag.py
"""
Évaluation automatique de la qualité des réponses du système RAG — alternative "maison" à Ragas.

Pourquoi pas Ragas : sa chaîne de dépendances (via langchain_community puis instructor) est
cassée avec les versions actuellement publiées sur PyPI, indépendamment de nos propres
versions (vérifié en environnement totalement isolé). Le brief autorise explicitement cette
alternative : "vous pouvez utiliser des métriques telles que le score de similarité... ou une
classification manuelle en correcte/partiellement correcte/incorrecte".

Deux signaux complémentaires par question, pas un seul :
1. Score de similarité cosinus (embeddings mistral-embed) entre réponse générée et référence.
   Rapide, gratuit, mais mesure une distance vectorielle globale : pénalise une réponse
   correcte mais verbeuse, ou un refus correct sur un sujet différent (vérifié en pratique).
2. Verdict d'un juge LLM (Gemini) qui lit vraiment les deux réponses et juge le sens.
   Plus lent (un appel API en plus), mais ne se laisse pas tromper par la formulation -
   c'est ce que Ragas aurait fait avec ses métriques "faithfulness"/"answer_correctness".

Seuils de similarité calibrés empiriquement (pas choisis au hasard) en comparant une phrase
de référence à :
  - une reformulation quasi-identique          -> 0.995
  - une paraphrase correcte                    -> 0.935
  - une réponse partielle (info manquante)     -> 0.909
  - une réponse factuellement fausse           -> 0.814
  - une réponse hors-sujet                     -> 0.772
D'où : >= 0.92 "correcte", >= 0.85 "partiellement correcte", en dessous "incorrecte".
"""

import sys
import logging
import numpy as np
from typing import List, Dict, Any

from mistralai.client import Mistral

from utils.config import MISTRAL_API_KEY, EMBEDDING_MODEL
from utils.chat_service import answer_query
from utils.gemini_client import gemini_complete

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

SEUIL_CORRECT = 0.92
SEUIL_PARTIEL = 0.85

# Seuil de réussite pour la CI (voir check_quality_gate) : proportion minimale de cas jugés
# "correcte" ou "partiellement correcte" par le juge LLM (signal privilégié sur la similarité
# brute, voir le docstring de ce module - moins trompé par une réponse correcte mais reformulée).
# 70% choisi comme point de départ raisonnable, pas une vérité absolue : sur un jeu de 10 cas,
# ça tolère jusqu'à 3 échecs avant de faire échouer la CI. À ajuster avec le recul de l'usage réel.
QUALITY_GATE_THRESHOLD = 0.70

# Jeu d'évaluation : questions + réponses de référence VÉRIFIÉES manuellement dans l'index
# (contenu réel trouvé par recherche directe avant d'écrire ce fichier, pas inventé).
EVAL_DATASET = [
    {
        "question": "Quel concert de Noël a eu lieu à Bordeaux ?",
        "reference": "Le Concert de Noël Radio Classique, avec l'Orchestre National Bordeaux Aquitaine "
                     "et le Chœur de l'Opéra National de Bordeaux, a eu lieu le 18 décembre 2025 à Bordeaux.",
    },
    {
        "question": "Quand a lieu l'exposition du prix de Paris à Lyon ?",
        "reference": "L'exposition du prix de Paris et l'ouverture de l'École des beaux-arts de Lyon "
                     "ont lieu le 19 septembre 2026 à Lyon.",
    },
    {
        # Formulée au passé avec inversion interrogative ("a-t-il eu lieu") - ce cas précis a
        # servi de non-régression à DEUX bugs distincts trouvés en pratique : (1) un filtrage par
        # date trop strict qui excluait le bon résultat (corrigé par le repli passé/futur dans
        # chat_service.py), et (2) la détection de mots-clés "événement passé" qui cherchait la
        # sous-chaîne exacte "a eu lieu" et ne matchait donc pas sa forme interrogative "a-t-il eu
        # lieu" (corrigé en généralisant les mots-clés à "eu lieu"/"déroulé").
        "question": "Quand le Festival des Solidarités a-t-il eu lieu à Lyon ?",
        "reference": "Le Festival des Solidarités 2025 s'est déroulé du 15 au 30 novembre 2025 à Lyon, "
                     "avec un rendez-vous le samedi 15 novembre de 10h à 18h à l'Hôtel de Ville.",
    },
    {
        # Cas piège : rien de tel n'existe dans l'index. Sert à vérifier que le système
        # n'invente pas de réponse plutôt qu'à tester un vrai événement.
        "question": "Quels événements de plongée sous-marine polaire y a-t-il en Antarctique ?",
        "reference": "Aucune information disponible : ce type d'événement n'existe pas dans la base de données.",
    },
    {
        # Question simple ville+thème, sans filtre temporel - cas de base pour un thème différent
        # (spectacle vivant) de celui des deux premiers cas (musique, exposition).
        "question": "Y a-t-il un événement autour du théâtre à Nantes ?",
        "reference": "La Nuit du Théâtre 2026 (6e édition) a lieu le samedi 26 septembre 2026 à Nantes : "
                     "un festival réunissant humour, comédie, impro, théâtre contemporain, comédie "
                     "musicale, cabaret et stand-up.",
    },
    {
        # Mois ET année explicites dans la question - vérifie que le filtre n'attrape pas une
        # exposition similaire d'une autre année (bug historique : le filtre mois ignorait
        # l'année avant sa correction).
        "question": "Quelle exposition de photographie a lieu à Paris en novembre 2026 ?",
        "reference": "L'exposition de Sam Samore à la Galerie Anne de Villepoix, dans le cadre du "
                     "parcours Photographie dans les galeries d'art, a lieu le 4 novembre 2026 à Paris.",
    },
    {
        # Filtre région SEULE, sans ville précisée - vérifie extract_region() et le filtre région
        # ajouté par migration sur les chunks déjà indexés (scripts/migrate_add_region.py).
        "question": "Quel concert de musique classique a lieu en Bretagne ?",
        "reference": "Un concert de musique classique (Ravel, Debussy, Mozart, Bach, Vivaldi...) a lieu "
                     "le 12 octobre 2026 à la Basilique Saint-Aubin de Rennes, en Bretagne, avec "
                     "l'Ensemble Musicâme.",
    },
    {
        # Multi-critère (région + mois) ET multi-thème (sport ET musique) dans la même question -
        # teste la décomposition en facettes (extract_search_facets) en conditions réelles, sur la
        # question qui a motivé son développement dans cette session.
        "question": "Si j'aime le sport et la musique et que je suis dans la région Rhône-Alpes "
                    "pour le mois d'octobre, qu'est-ce que tu me proposes ?",
        "reference": "Côté sport : un stage de basket a lieu le 19 octobre 2026 à Bessenay. Côté "
                     "musique : un concert (Ravel, Debussy, Mozart, Vivaldi, Bach...) a lieu le "
                     "25 octobre 2026 à Évian-les-Bains, tous deux en Auvergne-Rhône-Alpes.",
    },
    {
        # Période floue ("prochaines semaines", sans mois nommé) - non-régression sur le bug où
        # cette formulation était interprétée à tort comme un mois précis (souvent le mois en
        # cours), excluant des événements futurs pertinents d'un autre mois.
        "question": "J'aime la musique, tu me conseilles quoi sur Lyon dans les prochaines semaines ?",
        "reference": "SuperMegaSuperCool Révolution, un concert de jazz psychédélique dans le cadre du "
                     "Festival Amply, a lieu le vendredi 2 octobre 2026 à Lyon.",
    },
    {
        # Cas non-RAG (salutation générale) - vérifie que needs_rag() route correctement vers le
        # mode DIRECT plutôt que de lancer une recherche documentaire inutile.
        "question": "Bonjour, qui es-tu et que peux-tu faire ?",
        "reference": "Je suis l'assistant virtuel Puls-Events, spécialisé dans les recommandations "
                     "d'événements culturels (concerts, expositions, spectacles, festivals...). "
                     "Je peux vous aider à trouver des sorties selon vos envies et votre ville.",
    },
]


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def _classify_by_similarity(score: float) -> str:
    if score >= SEUIL_CORRECT:
        return "correcte"
    if score >= SEUIL_PARTIEL:
        return "partiellement correcte"
    return "incorrecte"


def _llm_judge(question: str, reference: str, response: str) -> Dict[str, str]:
    """
    Demande à Gemini de juger si la réponse générée porte le même sens/information que la
    référence, même formulée différemment - contourne les faux négatifs du score de similarité
    (réponse correcte mais verbeuse, refus correct sur un sujet différent).
    """
    prompt = f"""Question posée : {question}

Réponse de référence (vérifiée manuellement) : {reference}

Réponse générée par le système à évaluer : {response}

La réponse générée contient-elle la même information que la référence, même si la formulation
est différente (plus longue, un ordre différent, du texte en plus) ? Un refus correct compte
comme "correcte" si la référence est elle-même un refus/absence d'information.

Répondez STRICTEMENT sous la forme :
VERDICT: correcte|partiellement correcte|incorrecte
JUSTIFICATION: <une phrase>"""

    result = gemini_complete([{"role": "user", "content": prompt}], temperature=0.0)

    verdict, justification = "incorrecte", result.strip()  # repli si le format n'est pas respecté
    for line in result.splitlines():
        if line.upper().startswith("VERDICT:"):
            verdict = line.split(":", 1)[1].strip().lower()
        elif line.upper().startswith("JUSTIFICATION:"):
            justification = line.split(":", 1)[1].strip()

    return {"verdict": verdict, "justification": justification}


def evaluate() -> List[Dict[str, Any]]:
    """Exécute le RAG sur chaque question du jeu d'évaluation et note la réponse obtenue."""
    client = Mistral(api_key=MISTRAL_API_KEY)
    results = []

    for case in EVAL_DATASET:
        question, reference = case["question"], case["reference"]
        logging.info(f"Évaluation: {question}")

        rag_result = answer_query(question)
        response = rag_result["response"]

        # Signal 1 : similarité d'embeddings (un seul appel pour les deux textes)
        embeddings = client.embeddings.create(model=EMBEDDING_MODEL, inputs=[reference, response])
        ref_emb, resp_emb = embeddings.data[0].embedding, embeddings.data[1].embedding
        score = _cosine_similarity(ref_emb, resp_emb)
        verdict_similarity = _classify_by_similarity(score)

        # Signal 2 : jugement LLM
        judge = _llm_judge(question, reference, response)

        results.append({
            "question": question,
            "reference": reference,
            "response": response,
            "mode": rag_result["mode"],
            "similarity_score": score,
            "verdict_similarity": verdict_similarity,
            "verdict_llm": judge["verdict"],
            "justification_llm": judge["justification"],
        })
        logging.info(f"  Similarité: {score:.3f} ({verdict_similarity}) | Juge LLM: {judge['verdict']}")

    return results


def print_report(results: List[Dict[str, Any]]) -> None:
    print("\n=== Rapport d'évaluation RAG ===\n")
    for r in results:
        print(f"Question : {r['question']}")
        print(f"  Référence : {r['reference'][:120]}")
        print(f"  Réponse   : {r['response'][:120]}")
        print(f"  Similarité : {r['similarity_score']:.3f} -> {r['verdict_similarity']}")
        print(f"  Juge LLM   : {r['verdict_llm']} ({r['justification_llm']})")
        print()

    scores = [r["similarity_score"] for r in results]
    print(f"Score de similarité moyen : {sum(scores) / len(scores):.3f}")
    print()
    print("Répartition par méthode :")
    for label, key in [("Similarité", "verdict_similarity"), ("Juge LLM", "verdict_llm")]:
        print(f"  {label} :")
        for verdict in ("correcte", "partiellement correcte", "incorrecte"):
            count = sum(1 for r in results if r[key] == verdict)
            print(f"    {verdict} : {count}/{len(results)}")


def check_quality_gate(results: List[Dict[str, Any]]) -> bool:
    """
    Décide si l'évaluation "passe" pour la CI, sur la base du juge LLM (voir
    QUALITY_GATE_THRESHOLD). Affiche le verdict et retourne True/False plutôt que de renvoyer un
    simple booléen silencieux : la CI doit pouvoir expliquer pourquoi elle a échoué sans avoir à
    ré-exécuter le script en local.
    """
    not_incorrect = sum(1 for r in results if r["verdict_llm"] != "incorrecte")
    ratio = not_incorrect / len(results)
    passed = ratio >= QUALITY_GATE_THRESHOLD
    status = "OK" if passed else "ÉCHEC"
    print(f"\nSeuil de qualité CI : {not_incorrect}/{len(results)} cas non-incorrects "
          f"({ratio:.0%}) — seuil requis {QUALITY_GATE_THRESHOLD:.0%} — {status}")
    return passed


if __name__ == "__main__":
    results = evaluate()
    print_report(results)
    sys.exit(0 if check_quality_gate(results) else 1)
