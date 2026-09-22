# migrate_add_region.py
"""
Migration légère : remplit le champ metadata["region"] des chunks déjà indexés qui ne l'ont pas
(indexés avant l'ajout de ce champ dans openagenda_loader.py).

Approche : agrégation ville -> région via l'API OpenAgenda (group_by), PAS un fetch par événement
(131 210 événements uniques dans l'index -> beaucoup trop lent/coûteux en requêtes). Une ville
française appartient presque toujours à une seule région ; en cas de doublons/bruit dans les
données OpenAgenda elles-mêmes (constaté : "Auvergne Rhône-Alpes" sans tiret à côté de la forme
correcte "Auvergne-Rhône-Alpes" pour Lyon, 1 cas contre 6825), on prend la région majoritaire par
ville (vote sur le nombre d'événements OpenAgenda associés à chaque couple ville/région).

Aucun ré-embedding : le texte des chunks ne change pas, donc les vecteurs Faiss restent valides.
On patche uniquement le fichier de métadonnées document_chunks.pkl, en place.
"""

import pickle
import logging
from collections import defaultdict
from typing import Dict, List

import requests

from utils.config import DOCUMENT_CHUNKS_FILE

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

OPENAGENDA_API_URL = "https://public.opendatasoft.com/api/explore/v2.1/catalog/datasets/evenements-publics-openagenda/records"
PAGE_SIZE = 100
CITY_BATCH_SIZE = 50  # une agrégation globale (villes du monde entier) dépasse la limite de
                       # pagination de l'API (offset + limit <= 20000, testé en pratique) - on ne
                       # demande donc que les villes présentes dans NOTRE index, par lots.


def fetch_city_region_mapping(target_cities: List[str]) -> Dict[str, str]:
    """
    Récupère, pour chaque ville de `target_cities`, la région la plus fréquemment associée dans
    OpenAgenda (vote majoritaire sur le nombre d'événements, gère le bruit constaté dans leurs
    données : ex. "Auvergne Rhône-Alpes" sans tiret à côté de la forme correcte pour Lyon).
    Aucun appel Mistral/Gemini, uniquement l'API publique OpenAgenda.
    """
    counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    # Les guillemets doubles délimitent les chaînes en ODSQL - on écarte les rares noms de ville
    # qui en contiendraient plutôt que de gérer un échappement fragile pour un cas quasi inexistant.
    cities = [c for c in target_cities if '"' not in c]
    skipped = set(target_cities) - set(cities)
    if skipped:
        logging.warning(f"{len(skipped)} villes ignorées (guillemet dans le nom): {sorted(skipped)}")

    n_batches = (len(cities) + CITY_BATCH_SIZE - 1) // CITY_BATCH_SIZE
    for i in range(0, len(cities), CITY_BATCH_SIZE):
        batch = cities[i:i + CITY_BATCH_SIZE]
        clause = " or ".join(f'location_city="{c}"' for c in batch)
        offset = 0
        while True:
            response = requests.get(
                OPENAGENDA_API_URL,
                params={
                    "select": "location_city, location_region, count(*) as n",
                    "group_by": "location_city, location_region",
                    "where": f"({clause}) and location_region is not null",
                    "limit": PAGE_SIZE,
                    "offset": offset,
                },
                timeout=30,
            )
            response.raise_for_status()
            rows = response.json().get("results", [])
            if not rows:
                break
            for row in rows:
                city, region, n = row.get("location_city"), row.get("location_region"), row.get("n", 0)
                if city and region:
                    counts[city][region] += n
            offset += len(rows)
            if len(rows) < PAGE_SIZE:
                break

        logging.info(f"Lot {i // CITY_BATCH_SIZE + 1}/{n_batches} traité ({len(batch)} villes).")

    mapping = {city: max(regions.items(), key=lambda kv: kv[1])[0] for city, regions in counts.items()}
    logging.info(f"Mapping construit pour {len(mapping)}/{len(cities)} villes demandées.")
    return mapping


def migrate() -> None:
    logging.info(f"Chargement de {DOCUMENT_CHUNKS_FILE}...")
    with open(DOCUMENT_CHUNKS_FILE, "rb") as f:
        chunks = pickle.load(f)
    logging.info(f"{len(chunks)} chunks chargés.")

    missing_before = sum(
        1 for c in chunks if not c["metadata"].get("region") or c["metadata"]["region"] == "N/A"
    )
    logging.info(f"{missing_before} chunks sans région avant migration.")

    target_cities = sorted({
        c["metadata"].get("ville")
        for c in chunks
        if (not c["metadata"].get("region") or c["metadata"]["region"] == "N/A")
        and c["metadata"].get("ville") and c["metadata"]["ville"] != "N/A"
    })
    logging.info(f"{len(target_cities)} villes distinctes à résoudre.")

    mapping = fetch_city_region_mapping(target_cities)

    updated = 0
    not_found_cities = set()
    for chunk in chunks:
        md = chunk["metadata"]
        if md.get("region") and md["region"] != "N/A":
            continue  # déjà renseigné (chunks indexés après le fix), on ne touche pas
        ville = md.get("ville")
        if not ville or ville == "N/A":
            continue  # pas de ville connue, impossible de déduire la région
        region = mapping.get(ville)
        if region:
            md["region"] = region
            updated += 1
        else:
            not_found_cities.add(ville)

    logging.info(f"{updated} chunks mis à jour avec une région.")
    logging.info(f"{len(not_found_cities)} villes locales sans correspondance dans le mapping OpenAgenda.")
    if not_found_cities:
        sample = sorted(not_found_cities)[:20]
        logging.info(f"Exemples de villes non trouvées: {sample}")

    with open(DOCUMENT_CHUNKS_FILE, "wb") as f:
        pickle.dump(chunks, f)
    logging.info(f"{DOCUMENT_CHUNKS_FILE} sauvegardé (métadonnées uniquement, index Faiss inchangé).")


if __name__ == "__main__":
    migrate()
