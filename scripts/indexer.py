#!/bin/env python

import argparse
import logging
from typing import Optional

from utils.config import INPUT_DIR # INPUT_DATA_URL (décommentez si besoin)
from utils.data_loader import download_and_extract_zip, load_and_parse_files
from utils.openagenda_loader import fetch_events
from utils.vector_store import VectorStoreManager

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_indexing(
    input_directory: str,
    data_url: Optional[str] = None,
    source: str = "files",
    region: Optional[str] = None,
    city: Optional[str] = None,
    months_back: int = 12,
    max_records: int = 500,
):
    """
    Exécute le processus complet d'indexation.

    Args:
        input_directory: Dossier des fichiers sources (utilisé seulement si source="files").
        data_url: URL optionnelle d'un zip à télécharger avant indexation (source="files" uniquement).
        source: "files" pour indexer les documents locaux (comportement historique, mairie),
                "openagenda" pour indexer des événements culturels via l'API OpenAgenda (Puls-Events).
        region, city, months_back, max_records: paramètres transmis à fetch_events()
                quand source="openagenda" (ignorés sinon).
    """
    logging.info(f"--- Démarrage du processus d'indexation (source={source}) ---")

    # --- Chargement des documents : deux chemins possibles selon `source` ---
    # Les deux fonctions retournent le même format {"page_content": str, "metadata": dict},
    # donc tout ce qui suit (chunking, embeddings, Faiss, BM25) est identique dans les deux cas.
    if source == "openagenda":
        logging.info(f"Récupération des événements OpenAgenda (région={region}, ville={city}, "
                     f"depuis {months_back} mois, max {max_records})...")
        documents = fetch_events(region=region, city=city, months_back=months_back, max_records=max_records)
    else:
        # --- Téléchargement et Extraction (Optionnel) ---
        if data_url:
            logging.info(f"Tentative de téléchargement depuis l'URL: {data_url}")
            success = download_and_extract_zip(data_url, input_directory)
            if not success:
                logging.error("Échec du téléchargement ou de l'extraction. Arrêt.")
                # Décider si on continue avec le contenu local existant ou si on arrête.
                # Ici, on arrête pour éviter d'indexer des données potentiellement incomplètes/anciennes.
                return
        else:
            logging.info(f"Aucune URL fournie. Utilisation des fichiers locaux dans: {input_directory}")

        # --- Chargement et Parsing des Fichiers ---
        logging.info(f"Chargement et parsing des fichiers depuis: {input_directory}")
        documents = load_and_parse_files(input_directory)

    if not documents:
        logging.warning("Aucun document n'a été chargé ou parsé. Vérifiez le contenu du dossier d'entrée.")
        logging.info("--- Processus d'indexation terminé (aucun document traité) ---")
        return

    # --- Étape 3: Création/Mise à jour de l'index Vectoriel ---
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
    parser = argparse.ArgumentParser(description="Script d'indexation pour l'application RAG")
    parser.add_argument(
        "--input-dir",
        type=str,
        default=INPUT_DIR,
        help=f"Répertoire contenant les fichiers sources (par défaut: {INPUT_DIR})"
    )
    parser.add_argument(
        "--data-url",
        type=str,
        # default=INPUT_DATA_URL, # Décommentez pour utiliser la valeur du .env par défaut
        default=None,
        help="URL optionnelle pour télécharger et extraire un fichier inputs.zip"
    )
    # --- Nouveaux arguments pour la source "openagenda" (projet Puls-Events) ---
    parser.add_argument(
        "--source",
        type=str,
        choices=["files", "openagenda"],
        default="files",
        help="Source des documents: 'files' (défaut, fichiers locaux) ou 'openagenda' (API événements culturels)"
    )
    parser.add_argument(
        "--region",
        type=str,
        default=None,
        help="[source=openagenda] Filtre par région (ex: 'Nouvelle-Aquitaine')"
    )
    parser.add_argument(
        "--city",
        type=str,
        default=None,
        help="[source=openagenda] Filtre par ville (ex: 'Bordeaux')"
    )
    parser.add_argument(
        "--months-back",
        type=int,
        default=12,
        help="[source=openagenda] Ancienneté maximale des événements en mois (défaut: 12)"
    )
    parser.add_argument(
        "--max-records",
        type=int,
        default=500,
        help="[source=openagenda] Nombre maximum d'événements à récupérer (défaut: 500)"
    )
    args = parser.parse_args()

    # Vérifier si l'URL est passée en argument, sinon prendre celle du .env (si définie)
    # final_data_url = args.data_url if args.data_url is not None else INPUT_DATA_URL
    # Simplification: on utilise seulement l'argument --data-url pour l'instant
    final_data_url = args.data_url

    run_indexing(
        input_directory=args.input_dir,
        data_url=final_data_url,
        source=args.source,
        region=args.region,
        city=args.city,
        months_back=args.months_back,
        max_records=args.max_records,
    )