# index_by_region.py
"""Script CLI : construit ou complète l'index vectoriel pondéré par région (logique dans utils/indexing.py)."""

import argparse
import logging
from utils.indexing import rebuild_index, extend_index
from utils.vector_store import VectorStoreManager

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Indexation OpenAgenda pondérée par région")
    parser.add_argument(
        "--mode",
        choices=["rebuild", "extend"],
        default="rebuild",
        help="'rebuild' reconstruit tout depuis zéro (défaut). "
             "'extend' n'ajoute que les événements pas encore indexés (embeddings uniquement sur les nouveaux).",
    )
    args = parser.parse_args()

    vector_store = VectorStoreManager()
    if args.mode == "extend":
        result = extend_index(vector_store)
    else:
        result = rebuild_index(vector_store)
    logging.info(f"Résultat: {result}")
