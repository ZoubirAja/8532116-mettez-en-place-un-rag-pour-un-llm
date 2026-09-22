# puls_events_api.py
"""API REST du chatbot Puls-Events, exploitable par les équipes produit et marketing."""

import os
import logging
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks, Header, Depends
from pydantic import BaseModel, Field

from utils.config import APP_NAME, SEARCH_K
from utils.chat_service import answer_query, get_vector_store
from utils.indexing import rebuild_index

ADMIN_TOKEN = os.environ.get("ADMIN_TOKEN")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = FastAPI(
    title=APP_NAME,
    description="API de recommandations d'événements culturels basée sur un système RAG.",
    version="1.0.0",
)


class ChatRequest(BaseModel):
    """Corps de la requête POST /chat."""
    query: str = Field(..., min_length=1, description="Question de l'utilisateur")
    conversation_history: List[Dict[str, str]] = Field(
        default_factory=list,
        description="Tours précédents [{'role': 'user'|'assistant', 'content': '...'}], du plus ancien au plus récent. "
                    "L'API étant stateless, c'est à l'appelant de renvoyer l'historique à chaque appel.",
    )
    num_docs: int = Field(SEARCH_K, ge=1, le=20, description="Nombre max d'événements à récupérer")
    min_score: float = Field(0.75, ge=0.0, le=1.0, description="Score de similarité minimum (0-1)")
    model: str = Field("mistral-medium-latest", description="Modèle Mistral à utiliser")


class Source(BaseModel):
    """Un événement source cité dans la réponse."""
    text: str
    metadata: Dict[str, Any]
    score: float


class ChatResponse(BaseModel):
    """Corps de la réponse de POST /chat."""
    response: str
    sources: List[Source]
    mode: str
    confidence: float
    reason: str
    interaction_id: Optional[int]


@app.get("/health")
def health() -> Dict[str, str]:
    """Vérification simple que l'API est en ligne (utile pour monitoring/intégration)."""
    return {"status": "ok", "app": APP_NAME}


def _handle_chat(request: ChatRequest) -> ChatResponse:
    """Logique partagée entre /ask (nom demandé par le brief) et /chat (alias conservé)."""
    try:
        result = answer_query(
            request.query,
            conversation_history=request.conversation_history,
            num_docs=request.num_docs,
            min_score=request.min_score,
            model=request.model,
        )
        return ChatResponse(**result)
    except Exception as e:
        logging.error(f"Erreur lors du traitement de la requête: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ask", response_model=ChatResponse)
def ask(request: ChatRequest) -> ChatResponse:
    """
    Point d'entrée principal : pose une question au chatbot et reçoit une réponse
    éventuellement enrichie d'événements culturels sources.
    """
    return _handle_chat(request)


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    """Alias de /ask, conservé pour compatibilité (nom utilisé pendant le développement)."""
    return _handle_chat(request)


def _verify_admin_token(x_admin_token: str = Header(None)):
    """Bloque par défaut si ADMIN_TOKEN n'est pas configuré côté serveur (sécurisé par défaut)."""
    if not ADMIN_TOKEN:
        raise HTTPException(status_code=403, detail="Endpoint protégé: ADMIN_TOKEN non configuré côté serveur.")
    if x_admin_token != ADMIN_TOKEN:
        raise HTTPException(status_code=403, detail="Token invalide.")


# Suivi en mémoire du dernier rebuild (simple, pas de file d'attente : un seul rebuild à la fois)
rebuild_status: Dict[str, Any] = {"state": "idle"}


def _run_rebuild():
    rebuild_status["state"] = "running"
    try:
        result = rebuild_index(get_vector_store())
        rebuild_status["state"] = result["status"]
        rebuild_status["last_result"] = result
    except Exception as e:
        rebuild_status["state"] = "error"
        rebuild_status["last_result"] = {"error": str(e)}


@app.post("/rebuild", dependencies=[Depends(_verify_admin_token)])
def rebuild(background_tasks: BackgroundTasks) -> Dict[str, str]:
    """
    Relance la reconstruction complète de l'index vectoriel en arrière-plan (~40 min).
    Protégé par le header X-Admin-Token. Répond immédiatement sans attendre la fin.
    """
    if rebuild_status["state"] == "running":
        raise HTTPException(status_code=409, detail="Un rebuild est déjà en cours.")
    background_tasks.add_task(_run_rebuild)
    rebuild_status["state"] = "starting"
    return {"status": "rebuild_started"}


@app.get("/rebuild/status")
def get_rebuild_status() -> Dict[str, Any]:
    """Consulte l'état du dernier rebuild déclenché (utile puisque /rebuild répond avant la fin réelle)."""
    return rebuild_status
