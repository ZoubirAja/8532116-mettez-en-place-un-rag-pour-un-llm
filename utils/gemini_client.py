# utils/gemini_client.py
"""
Client Gemini (Google), utilisé temporairement à la place de Mistral pour la génération
de texte (chat/completions), le temps que l'accès Mistral soit débloqué. Les embeddings
restent sur Mistral (mistral-embed n'est pas concerné par le blocage, aucune raison d'y toucher).
"""

import os
import time
import logging
from typing import List, Dict, Optional

import httpx
from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.genai.errors import ServerError, ClientError

load_dotenv()  # ne dépend pas de l'ordre d'import d'utils.config ailleurs dans le projet

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
# gemini-3.6-flash a un quota gratuit de seulement 20 requêtes/JOUR (vérifié: erreur 429
# RESOURCE_EXHAUSTED, quotaId "GenerateRequestsPerDayPerProjectPerModel-FreeTier", valeur 20).
# Avec jusqu'à 4 appels Gemini par tour de conversation, ça s'épuise en 5 messages.
# gemini-flash-lite-latest n'a pas ce problème et ne supporte de toute façon pas
# thinking_config (testé : 400 INVALID_ARGUMENT si on le passe) - donc pas de raisonnement
# interne qui mange le budget de tokens, contrairement à gemini-3.6-flash.
GEMINI_MODEL = "gemini-flash-lite-latest"

_client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None


def _extract_retry_delay_seconds(error: ClientError) -> Optional[float]:
    """
    Extrait le délai d'attente recommandé par Gemini lui-même sur un 429 (champ RetryInfo.retryDelay
    de la réponse, ex: "10s") plutôt que de deviner un backoff arbitraire - trouvé en pratique
    largement plus fiable : le quota gratuit (15 req/min) se réinitialise à un instant précis que
    seul le serveur connaît, un backoff exponentiel classique (2s, 4s, 8s...) est soit trop court
    (nouvel échec immédiat) soit trop long (attente inutile) par rapport à ce délai réel.
    """
    try:
        details = error.details.get("error", {}).get("details", [])
        for item in details:
            if item.get("@type", "").endswith("RetryInfo"):
                delay_str = item.get("retryDelay", "")  # ex: "10s", "1.012171887s"
                return float(delay_str.rstrip("s"))
    except (AttributeError, ValueError, TypeError):
        pass
    return None


def gemini_complete(messages: List[Dict[str, str]], temperature: float = 0.7, max_tokens: Optional[int] = None) -> str:
    """
    Génère une réponse avec Gemini à partir d'une liste de messages au format Mistral
    ([{"role": "system"|"user"|"assistant", "content": "..."}]) — même format que .format()
    produit déjà partout ailleurs dans le projet, pour ne pas avoir à réécrire les appelants.

    Différence structurelle avec Mistral que cette fonction gère pour vous :
    - Gemini sépare le prompt système du reste (paramètre `system_instruction`), il ne fait
      pas partie de la liste de messages comme chez Mistral.
    - Gemini nomme le rôle de l'IA "model", pas "assistant".
    """
    if not _client:
        raise RuntimeError("GEMINI_API_KEY manquante dans .env")

    system_instruction = None
    contents = []
    for m in messages:
        if m["role"] == "system":
            system_instruction = m["content"]  # extrait à part, pas mis dans `contents`
        else:
            gemini_role = "model" if m["role"] == "assistant" else "user"
            contents.append({"role": gemini_role, "parts": [{"text": m["content"]}]})

    config = types.GenerateContentConfig(
        system_instruction=system_instruction,
        temperature=temperature,
        max_output_tokens=max_tokens,
        # Pas de thinking_config ici : gemini-flash-lite-latest ne le supporte pas (testé :
        # 400 INVALID_ARGUMENT si on le passe), et n'a de toute façon pas le problème de
        # "raisonnement interne qui mange le budget de tokens" rencontré avec gemini-3.6-flash.
    )

    # Retry avec backoff sur les erreurs 503 (surcharge temporaire des serveurs Gemini) ET sur
    # les erreurs réseau bas niveau type "Server disconnected without sending a response"
    # (httpx.RequestError - rencontré en pratique sur une requête avec un contexte volumineux,
    # pas seulement une panne Gemini classique), ET sur le 429 du quota gratuit (15 req/min,
    # rencontré très régulièrement en pratique dès qu'on enchaîne plusieurs questions - chaque
    # question du pipeline déclenche ~5-7 appels Gemini légers en plus de la génération finale).
    max_retries = 5  # plus élevé que pour ServerError/réseau : le 429 est fréquent et transitoire,
                      # pas une vraie panne, ça vaut le coup d'insister davantage.
    for attempt in range(max_retries):
        try:
            response = _client.models.generate_content(model=GEMINI_MODEL, contents=contents, config=config)
            # Filet de sécurité : si Gemini ne produit toujours aucun texte (contenu bloqué par
            # un filtre de sécurité, budget de tokens encore trop juste...), on renvoie une chaîne
            # vide plutôt que None, pour que les appelants (qui font tous .strip() dessus) ne cassent pas.
            return response.text or ""
        except ClientError as e:
            if e.code == 429 and attempt < max_retries - 1:
                # Le serveur Gemini indique lui-même le délai exact avant réinitialisation du
                # quota (voir _extract_retry_delay_seconds) - plus fiable qu'un backoff deviné.
                wait_time = _extract_retry_delay_seconds(e) or (2 ** (attempt + 1))
                wait_time += 0.5  # marge de sécurité, évite de retomber pile sur la limite
                logging.warning(f"Quota Gemini atteint (429), nouvelle tentative dans {wait_time:.1f}s (essai {attempt + 1}/{max_retries})...")
                time.sleep(wait_time)
                continue
            raise
        except (ServerError, httpx.RequestError) as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** (attempt + 1)  # 2s, 4s, 8s, 16s
                logging.warning(f"Erreur Gemini ({type(e).__name__}: {e}), nouvelle tentative dans {wait_time}s (essai {attempt + 1}/{max_retries})...")
                time.sleep(wait_time)
                continue
            raise
