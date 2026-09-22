# utils/formatting.py
"""Formatage de dates lisibles en français, à partir du format ISO renvoyé par OpenAgenda."""

import datetime
from typing import Optional

_JOURS_FR = ["Lundi", "Mardi", "Mercredi", "Jeudi", "Vendredi", "Samedi", "Dimanche"]
_MOIS_FR = ["Janvier", "Février", "Mars", "Avril", "Mai", "Juin",
            "Juillet", "Août", "Septembre", "Octobre", "Novembre", "Décembre"]


def format_date_fr(date_str: Optional[str]) -> str:
    """
    Convertit une date ISO ("2026-09-20T14:00:00+00:00") en format lisible
    ("Le Mardi 20 Septembre 2026"). Écrit à la main (pas de locale système,
    pour rester portable entre machines/Docker sans dépendre d'un paquet de langue installé).
    """
    if not date_str or date_str == "N/A":
        return "Date inconnue"
    try:
        dt = datetime.datetime.fromisoformat(date_str)
    except (ValueError, TypeError):
        return date_str  # non parsable : on renvoie tel quel plutôt que planter
    return f"Le {_JOURS_FR[dt.weekday()]} {dt.day} {_MOIS_FR[dt.month - 1]} {dt.year}"
