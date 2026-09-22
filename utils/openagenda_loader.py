# utils/openagenda_loader.py
"""Charge des événements culturels depuis l'API publique OpenAgenda (portail Opendatasoft)."""

import re
import html
import logging
import datetime
from typing import Dict, List, Optional

import requests

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

OPENAGENDA_API_URL = "https://public.opendatasoft.com/api/explore/v2.1/catalog/datasets/evenements-publics-openagenda/records"
PAGE_SIZE = 100  # taille de page max supportée par l'API Opendatasoft

# Motifs de originagenda_title à exclure car non-culturels (emploi, hébergement touristique...).
# Liste "best effort" basée sur des échantillons observés, PAS exhaustive : le dataset mélange
# des milliers d'agendas et beaucoup d'autres sources bruyantes ne sont probablement pas couvertes.
DEFAULT_EXCLUDED_AGENDAS = [
    "France Travail",
    "structures d'accueil et d'hébergement",
    "Semaine des métiers",
    "Semaine de l'industrie",
]


def _strip_html(raw_html: Optional[str]) -> str:
    """Retire les balises HTML (ex: longdescription_fr) et décode les entités (&eacute; etc.)."""
    if not raw_html:
        return ""
    text = re.sub(r"<[^>]+>", " ", raw_html)
    text = html.unescape(text)
    return re.sub(r"\s+", " ", text).strip()


def _months_ago(months: int) -> datetime.date:
    """Calcule la date d'il y a `months` mois (arithmétique de calendrier, pas une approximation en jours)."""
    today = datetime.date.today()
    total_months = today.year * 12 + (today.month - 1) - months
    year, month = divmod(total_months, 12)
    month += 1
    day = min(today.day, 28)  # évite les erreurs de fin de mois (ex: 31 février)
    return datetime.date(year, month, day)


def _build_where_clause(
    region: Optional[str],
    city: Optional[str],
    months_back: int,
    excluded_agendas: Optional[List[str]] = None,
) -> str:
    """
    Construit la clause ODSQL 'where' de l'API Opendatasoft.
    Filtre les événements dont la date de début n'est pas plus ancienne que `months_back` mois,
    sans aucune limite dans le futur (comme demandé), et exclut les agendas connus comme bruyants
    (non-culturels).
    """
    cutoff_date = _months_ago(months_back).isoformat()
    clauses = [f"firstdate_begin >= date'{cutoff_date}'"]
    if region:
        clauses.append(f'location_region="{region}"')
    if city:
        clauses.append(f'location_city="{city}"')
    for pattern in (excluded_agendas or []):
        clauses.append(f'not(originagenda_title like "{pattern}")')
    return " AND ".join(clauses)


def fetch_events(
    region: Optional[str] = None,
    city: Optional[str] = None,
    months_back: int = 12,
    max_records: int = 500,
    excluded_agendas: Optional[List[str]] = None,
) -> List[Dict[str, any]]:
    """
    Récupère des événements culturels depuis l'API OpenAgenda et les transforme au format
    attendu par VectorStoreManager.build_index() : {"page_content": str, "metadata": dict}.

    Args:
        region: Filtre sur location_region (ex: "Nouvelle-Aquitaine"). Optionnel.
        city: Filtre sur location_city (ex: "Bordeaux"). Optionnel.
        months_back: Ancienneté maximale acceptée en mois (défaut 12 = 1 an dans le passé).
                     Aucune limite n'est appliquée dans le futur.
        max_records: Nombre maximum d'événements à récupérer. Protection nécessaire car le
                     dataset complet contient plus d'1,2 million d'événements.
        excluded_agendas: Motifs de originagenda_title à exclure (filtre "best effort" non
                           exhaustif). Par défaut, utilise DEFAULT_EXCLUDED_AGENDAS.

    Returns:
        Liste de documents au format {"page_content": ..., "metadata": ...}.
    """
    if excluded_agendas is None:
        excluded_agendas = DEFAULT_EXCLUDED_AGENDAS
    where_clause = _build_where_clause(region, city, months_back, excluded_agendas)
    logging.info(f"Requête OpenAgenda avec where=\"{where_clause}\"")

    documents = []
    offset = 0

    while len(documents) < max_records:
        page_limit = min(PAGE_SIZE, max_records - len(documents))
        params = {"where": where_clause, "limit": page_limit, "offset": offset}

        try:
            response = requests.get(OPENAGENDA_API_URL, params=params, timeout=30)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            logging.error(f"Erreur lors de l'appel à l'API OpenAgenda (offset={offset}): {e}")
            break

        data = response.json()
        results = data.get("results", [])
        if not results:
            break

        for event in results:
            title = event.get("title_fr") or ""
            description = event.get("description_fr") or ""
            long_description = _strip_html(event.get("longdescription_fr"))

            page_content = "\n\n".join(part for part in [title, description, long_description] if part)
            if not page_content:
                logging.debug(f"Événement {event.get('uid')} ignoré: aucun texte exploitable.")
                continue

            documents.append({
                "page_content": page_content,
                "metadata": {
                    "source": event.get("canonicalurl", "N/A"),
                    "filename": title or event.get("uid", "N/A"),
                    "category": ", ".join(event.get("keywords_fr") or []) or "non catégorisé",
                    "date": event.get("firstdate_begin", "N/A"),
                    "ville": event.get("location_city", "N/A"),
                    "region": event.get("location_region", "N/A"),
                    "lieu": event.get("location_name", "N/A"),
                    "url": event.get("canonicalurl", "N/A"),
                }
            })

        offset += len(results)
        logging.info(f"{len(documents)} événements chargés (total disponible côté API: {data.get('total_count', '?')})")

        if len(results) < page_limit:
            break  # Plus de résultats disponibles

    logging.info(f"Chargement terminé: {len(documents)} événements transformés en documents.")
    return documents
