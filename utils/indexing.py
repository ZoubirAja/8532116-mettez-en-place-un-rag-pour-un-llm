# utils/indexing.py
"""
Logique de (re)construction de l'index vectoriel, pondérée par région.
Réutilisée à la fois par le script CLI (index_by_region.py) et par l'API (/rebuild).
"""

import shutil
import logging
from typing import List, Tuple, Dict, Any, Optional

from utils.openagenda_loader import fetch_events
from utils.vector_store import VectorStoreManager

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# (région, max_records) — budget calibré sur la taille de la ville principale de la région
DEFAULT_REGION_BUDGETS: List[Tuple[str, int]] = [
    ("Île-de-France", 8000),
    ("Provence-Alpes-Côte d'Azur", 8000),
    ("Auvergne-Rhône-Alpes", 8000),
    ("Nouvelle-Aquitaine", 4000),
    ("Occitanie", 4000),
    ("Hauts-de-France", 4000),
    ("Pays de la Loire", 4000),
    ("Grand Est", 4000),
    ("Bretagne", 4000),
    ("Normandie", 1500),
    ("Bourgogne-Franche-Comté", 1500),
    ("Centre-Val de Loire", 1500),
    ("Corse", 1500),
]

MONTHS_BACK = 12
MAX_INDEX_SIZE_GB = 1.0
# Estimation mesurée sur un index réel (12.4 Mo / 2498 chunks) : ~5074 octets/chunk.
ESTIMATED_BYTES_PER_CHUNK = 5074

# Budgets "sans limite réelle" : le total_count de l'API borne naturellement la récupération,
# 999_999 ne sert qu'à ne pas boucler indéfiniment si jamais un total_count était énorme.
MAX_REGION_BUDGETS: List[Tuple[str, int]] = [
    (region, 999_999) for region, _ in DEFAULT_REGION_BUDGETS
]

# Espace disque minimum devant rester libre sur la machine APRÈS l'opération (contrainte personnelle,
# pas liée à la taille de l'index en soi).
MIN_FREE_DISK_GB = 3.0


def _check_disk_safety(estimated_new_bytes: float, path: str = ".") -> Tuple[bool, str]:
    """Vérifie l'espace disque RÉEL restant (pas une estimation abstraite), avant de dépenser des appels API."""
    _, _, free = shutil.disk_usage(path)
    free_gb = free / (1024 ** 3)
    projected_free_gb = (free - estimated_new_bytes) / (1024 ** 3)
    if projected_free_gb < MIN_FREE_DISK_GB:
        return False, (
            f"Espace libre actuel: {free_gb:.2f} Go. Après ajout estimé "
            f"(~{estimated_new_bytes / 1024**3:.2f} Go), il resterait {projected_free_gb:.2f} Go — "
            f"sous le minimum de {MIN_FREE_DISK_GB} Go requis."
        )
    return True, f"Espace libre actuel: {free_gb:.2f} Go, ~{projected_free_gb:.2f} Go resteraient après ajout — OK."


def rebuild_index(
    vector_store: VectorStoreManager,
    region_budgets: Optional[List[Tuple[str, int]]] = None,
    months_back: int = MONTHS_BACK,
) -> Dict[str, Any]:
    """
    Reconstruit l'index vectoriel à partir d'événements OpenAgenda, région par région,
    avec un budget d'événements différent par région (grandes métropoles = budget plus élevé).

    Args:
        vector_store: instance à mettre à jour (modifiée EN PLACE : index, chunks, bm25)
        region_budgets: liste de (région, max_records). Par défaut: DEFAULT_REGION_BUDGETS.
        months_back: ancienneté maximale des événements en mois.

    Returns:
        dict résumant le résultat: status ("ok" ou "aborted"), n_events, n_chunks, size_gb
    """
    region_budgets = region_budgets or DEFAULT_REGION_BUDGETS

    all_documents = []
    for region, max_records in region_budgets:
        logging.info(f"--- Récupération: {region} (max {max_records}) ---")
        docs = fetch_events(region=region, months_back=months_back, max_records=max_records)
        logging.info(f"{len(docs)} événements récupérés pour {region}")
        all_documents.extend(docs)

    logging.info(f"Total: {len(all_documents)} événements sur {len(region_budgets)} régions")

    # --- Vérification de sécurité AVANT les embeddings (étape payante) ---
    chunks = vector_store._split_documents_to_chunks(all_documents)
    estimated_size_gb = (len(chunks) * ESTIMATED_BYTES_PER_CHUNK) / (1024 ** 3)
    logging.info(f"{len(chunks)} chunks au total, taille estimée: {estimated_size_gb:.2f} Go")

    if estimated_size_gb > MAX_INDEX_SIZE_GB:
        message = (
            f"taille estimée ({estimated_size_gb:.2f} Go) dépasse la limite de sécurité "
            f"({MAX_INDEX_SIZE_GB} Go). Aucun appel d'embedding n'a été fait."
        )
        logging.error(f"ARRÊT: {message}")
        return {"status": "aborted", "reason": message, "n_events": len(all_documents), "n_chunks": len(chunks)}

    # --- Construction de l'index (embeddings + Faiss + BM25 + sauvegarde) ---
    # Modifie `vector_store` EN PLACE : les futurs appels à search() sur cette même
    # instance utilisent immédiatement les nouvelles données, sans redémarrage du serveur.
    vector_store.build_index(all_documents)

    n_chunks = vector_store.index.ntotal if vector_store.index else 0
    logging.info(f"Indexation terminée: {n_chunks} chunks indexés")

    return {"status": "ok", "n_events": len(all_documents), "n_chunks": n_chunks, "size_gb": estimated_size_gb}


def extend_index(
    vector_store: VectorStoreManager,
    region_budgets: Optional[List[Tuple[str, int]]] = None,
    months_back: int = MONTHS_BACK,
) -> Dict[str, Any]:
    """
    Complète l'index EXISTANT avec les événements pas encore indexés (déduplication par URL),
    sans réembeder ce qui l'est déjà. Contrairement à rebuild_index(), n'écrase rien.

    Args:
        vector_store: instance à étendre (déjà chargée avec son index existant)
        region_budgets: liste de (région, max_records). Par défaut: MAX_REGION_BUDGETS (quasi sans limite).
        months_back: ancienneté maximale des événements en mois.

    Returns:
        dict résumant le résultat: status ("ok" ou "aborted"), n_new_events, n_new_chunks
    """
    region_budgets = region_budgets or MAX_REGION_BUDGETS

    existing_urls = {c["metadata"].get("url") for c in vector_store.document_chunks}
    logging.info(f"{len(existing_urls)} événements déjà indexés (par URL unique)")

    new_documents = []
    for region, max_records in region_budgets:
        logging.info(f"--- Récupération: {region} (max {max_records}) ---")
        docs = fetch_events(region=region, months_back=months_back, max_records=max_records)
        new_docs = [d for d in docs if d["metadata"].get("url") not in existing_urls]
        logging.info(f"{region}: {len(docs)} récupérés, {len(new_docs)} réellement nouveaux")
        new_documents.extend(new_docs)
        # Évite d'ajouter deux fois le même événement si deux régions le retournent toutes les deux
        existing_urls.update(d["metadata"].get("url") for d in new_docs)

    logging.info(f"Total: {len(new_documents)} nouveaux événements sur {len(region_budgets)} régions")

    if not new_documents:
        logging.info("Aucun nouvel événement à ajouter.")
        return {"status": "ok", "n_new_events": 0, "n_new_chunks": 0}

    # --- Vérification de sécurité AVANT les embeddings (étape payante) : espace disque RÉEL ---
    new_chunks = vector_store._split_documents_to_chunks(new_documents)
    estimated_new_bytes = len(new_chunks) * ESTIMATED_BYTES_PER_CHUNK
    ok, message = _check_disk_safety(estimated_new_bytes)
    logging.info(message)

    if not ok:
        logging.error(f"ARRÊT: {message} Aucun appel d'embedding n'a été fait.")
        return {"status": "aborted", "reason": message, "n_new_events": len(new_documents), "n_new_chunks": len(new_chunks)}

    # --- Ajout incrémental (embeddings uniquement sur les nouveaux chunks) ---
    vector_store.add_chunks(new_chunks)

    return {"status": "ok", "n_new_events": len(new_documents), "n_new_chunks": len(new_chunks)}
