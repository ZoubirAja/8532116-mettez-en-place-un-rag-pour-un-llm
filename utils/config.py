# utils/config.py
import os
from dotenv import load_dotenv

# Charger les variables d'environnement du fichier .env
load_dotenv()

# --- Clé API ---
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
if not MISTRAL_API_KEY:
    print("⚠️ Attention: La clé API Mistral (MISTRAL_API_KEY) n'est pas définie dans le fichier .env")
    # Vous pouvez choisir de lever une exception ici ou de continuer avec des fonctionnalités limitées
    # raise ValueError("Clé API Mistral manquante. Veuillez la définir dans le fichier .env")

# --- Modèles Mistral ---
EMBEDDING_MODEL = "mistral-embed"
CHAT_MODEL = "mistral-small-latest" # Ou un autre modèle comme mistral-large-latest

# --- Configuration de l'Indexation ---
VECTOR_DB_DIR = "vector_db"         # Dossier pour stocker l'index Faiss et les chunks
FAISS_INDEX_FILE = os.path.join(VECTOR_DB_DIR, "faiss_index.idx")
DOCUMENT_CHUNKS_FILE = os.path.join(VECTOR_DB_DIR, "document_chunks.pkl")

CHUNK_SIZE = 1500                   # Taille des chunks en *caractères* (vise ~512 tokens)
CHUNK_OVERLAP = 150                 # Chevauchement en *caractères*
EMBEDDING_BATCH_SIZE = 32           # Taille des lots pour l'API d'embedding

# --- Configuration de la Recherche ---
# Plafond de SÉCURITÉ sur le nombre de documents (pas la vraie limite pratique, voir
# MAX_CONTEXT_CHARS ci-dessous). Un plafond fixe en nombre de documents s'est révélé arbitraire :
# k=5 excluait des événements pertinents dès qu'un filtre ville/mois réduisait le pool de
# candidats (vérifié : un vrai concert classé 14e sur 52 candidats Lyon+octobre restait hors du
# top 5), mais un k illimité explose sur une question large (vérifié : 13 242 résultats
# franchissent le seuil de qualité pour "concerts de musique classique en France" sans filtre
# ville/date - enverrait des millions de tokens de contexte). SEARCH_K reste un garde-fou haut,
# rarement la contrainte qui joue réellement.
SEARCH_K = 50

# Budget de contexte texte (en caractères, pas en tokens : on ne dispose pas d'un tokenizer
# fiable identique à celui de Gemini côté client, donc on approxime avec les caractères plutôt
# que de dépendre d'une bibliothèque de tokenisation tierce). C'est la vraie limite pratique :
# les résultats sont ajoutés au contexte par ordre de pertinence jusqu'à atteindre ce budget, ce
# qui laisse passer TOUS les résultats pertinents d'une recherche filtrée (ville+mois réduit déjà
# le pool à quelques dizaines) sans jamais dépendre d'un nombre de documents choisi arbitrairement,
# tout en bornant le coût/temps sur une question large. ~24000 caractères ≈ 6000 tokens (ratio
# approximatif ~4 caractères/token en français) ≈ une trentaine d'événements de taille moyenne.
MAX_CONTEXT_CHARS = 24000

# --- Configuration de la Base de Données ---
DATABASE_DIR = "database"
DATABASE_FILE = os.path.join(DATABASE_DIR, "interactions.db")
DATABASE_URL = f"sqlite:///{DATABASE_FILE}" # URL pour SQLAlchemy

# --- Configuration de l'Application ---
APP_TITLE = "Puls-Events — Assistant Événements Culturels"
APP_NAME = "Puls-Events" # Nom utilisé dans les prompts système et l'interface