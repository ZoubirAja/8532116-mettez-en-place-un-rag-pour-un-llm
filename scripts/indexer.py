#!/bin/env python

import argparse
import logging
from typing import Optional

from utils.openagenda_loader import fetch_events
from utils.vector_store import VectorStoreManager

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_indexing(
    region: Optional[str] = None,
    city: Optional[str] = None,
    months_back: int = 12,
    max_records: int = 500,
):
    """
    Exécute le processus complet d'indexation depuis l'API OpenAgenda : récupération des
    événements, découpage en chunks, embeddings, construction de l'index Faiss.

    Args:
        region, city, months_back, max_records: paramètres transmis à fetch_events().
    """
    logging.info(f"--- Démarrage du processus d'indexation ---")
    logging.info(f"Récupération des événements OpenAgenda (région={region}, ville={city}, "
                 f"depuis {months_back} mois, max {max_records})...")
    documents = fetch_events(region=region, city=city, months_back=months_back, max_records=max_records)

    if not documents:
        logging.warning("Aucun document n'a été chargé. Vérifiez les filtres région/ville.")
        logging.info("--- Processus d'indexation terminé (aucun document traité) ---")
        return

    logging.info("Initialisation du gestionnaire de Vector Store...")
    vector_store = VectorStoreManager() # Le constructeur ne fait que charger s'il existe

    logging.info("Construction de l'index Faiss (cela peut prendre du temps)...")
    # Cette méthode va splitter, générer les embeddings, créer l'index et sauvegarder
    vector_store.build_index(documents)

    logging.info("--- Processus d'indexation terminé avec succès ---")
    logging.info(f"Nombre de documents traités: {len(documents)}")
    if vector_store.index:
        logging.info(f"Nombre de chunks indexés: {vector_store.index.ntotal}")
    else:
        logging.warning("L'index final n'a pas pu être créé ou est vide.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Indexation d'événements OpenAgenda (région/ville ciblée)")
    parser.add_argument("--region", type=str, default=None, help="Filtre par région (ex: 'Nouvelle-Aquitaine')")
    parser.add_argument("--city", type=str, default=None, help="Filtre par ville (ex: 'Bordeaux')")
    parser.add_argument("--months-back", type=int, default=12, help="Ancienneté maximale des événements en mois (défaut: 12)")
    parser.add_argument("--max-records", type=int, default=500, help="Nombre maximum d'événements à récupérer (défaut: 500)")
    args = parser.parse_args()

    run_indexing(
        region=args.region,
        city=args.city,
        months_back=args.months_back,
        max_records=args.max_records,
    )
