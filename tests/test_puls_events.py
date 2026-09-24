# tests/test_puls_events.py
"""Tests unitaires pour les modules purs de Puls-Events (pas d'appel réseau/API réel)."""

import datetime
from unittest.mock import MagicMock

import pytest

from utils.chat_message import ChatMessage
from utils.openagenda_loader import _strip_html, _months_ago, _build_where_clause
from utils.query_classifier import QueryClassifier


# --- ChatMessage ---

def test_chat_message_format():
    msg = ChatMessage(role="user", content="Bonjour")
    assert msg.format() == {"role": "user", "content": "Bonjour"}


# --- openagenda_loader._strip_html ---

def test_strip_html_removes_tags():
    assert _strip_html("<p>Bonjour <b>le monde</b></p>") == "Bonjour le monde"

def test_strip_html_decodes_entities():
    assert _strip_html("Caf&eacute; &amp; Th&eacute;") == "Café & Thé"

def test_strip_html_collapses_repeated_spaces_and_tabs():
    # Espaces/tabulations multiples -> un seul, mais les sauts de ligne réels ne sont PLUS
    # aplatis (voir test_strip_html_preserves_paragraph_breaks) - changement volontaire.
    assert _strip_html("Un   texte avec\tdes espaces") == "Un texte avec des espaces"

def test_strip_html_none_returns_empty_string():
    assert _strip_html(None) == ""

def test_strip_html_empty_string_returns_empty_string():
    assert _strip_html("") == ""

def test_strip_html_preserves_paragraph_breaks():
    # OpenAgenda structure ses descriptions avec <p> (ex: intro puis programme minuté) - vérifié
    # sur des exemples réels de l'API. Perdre ces coupures nuit à la lisibilité pour le LLM et
    # aux points de coupure disponibles pour le chunking (bug réel corrigé, voir docstring).
    html_input = "<p>Intro du texte.</p><p>Deuxième paragraphe.</p>"
    assert _strip_html(html_input) == "Intro du texte.\n\nDeuxième paragraphe."

def test_strip_html_converts_br_to_newline():
    html_input = "<p>Ligne un.<br>Ligne deux.<br/>Ligne trois.</p>"
    assert _strip_html(html_input) == "Ligne un.\nLigne deux.\nLigne trois."

def test_strip_html_converts_li_to_bullet():
    html_input = "<ul><li>Premier point</li><li>Deuxième point</li></ul>"
    assert _strip_html(html_input) == "- Premier point\n- Deuxième point"

def test_strip_html_collapses_excessive_blank_lines():
    # Les <p></p> vides (fréquents chez OpenAgenda, utilisés pour l'espacement visuel) ne
    # doivent pas produire plus de 2 sauts de ligne consécutifs.
    html_input = "<p>Un</p><p></p><p></p><p>Deux</p>"
    assert _strip_html(html_input) == "Un\n\nDeux"


# --- openagenda_loader._months_ago ---

def test_months_ago_same_year():
    today = datetime.date.today()
    result = _months_ago(1)
    # Le mois doit reculer de 1 (avec retenue sur l'année si on est en janvier)
    expected_month = today.month - 1 or 12
    expected_year = today.year if today.month > 1 else today.year - 1
    assert result.month == expected_month
    assert result.year == expected_year

def test_months_ago_crosses_year_boundary():
    today = datetime.date.today()
    result = _months_ago(today.month + 12)  # recule largement avant l'année en cours
    assert result.year < today.year

def test_months_ago_avoids_end_of_month_overflow():
    # Le jour est toujours <= 28 pour éviter une erreur si le mois cible a moins de jours
    result = _months_ago(6)
    assert result.day <= 28


# --- openagenda_loader._build_where_clause ---

def test_build_where_clause_minimal():
    clause = _build_where_clause(region=None, city=None, months_back=12)
    assert "firstdate_begin >=" in clause
    assert "location_region" not in clause
    assert "location_city" not in clause

def test_build_where_clause_with_region_and_city():
    clause = _build_where_clause(region="Nouvelle-Aquitaine", city="Bordeaux", months_back=12)
    assert 'location_region="Nouvelle-Aquitaine"' in clause
    assert 'location_city="Bordeaux"' in clause

def test_build_where_clause_with_excluded_agendas():
    clause = _build_where_clause(region=None, city=None, months_back=12, excluded_agendas=["France Travail"])
    assert 'not(originagenda_title like "France Travail")' in clause


# --- QueryClassifier.needs_rag (chemins rapides, sans appel LLM) ---

@pytest.fixture
def classifier():
    return QueryClassifier()

def test_needs_rag_greeting_is_direct(classifier):
    needs_rag, confidence, reason = classifier.needs_rag("Bonjour !")
    assert needs_rag is False
    assert confidence > 0.9

def test_needs_rag_event_keyword_triggers_rag(classifier):
    needs_rag, confidence, reason = classifier.needs_rag("Quels concerts ce week-end ?")
    assert needs_rag is True
    assert "concert" in reason.lower()

def test_needs_rag_festival_keyword_triggers_rag(classifier):
    needs_rag, confidence, reason = classifier.needs_rag("Y a-t-il un festival prévu ?")
    assert needs_rag is True


# --- QueryClassifier._classify_with_llm (avec mock : pas de vrai appel réseau) ---

def test_classify_with_llm_calls_complete_not_chat(classifier, monkeypatch):
    """
    Test de non-régression : vérifie que gemini_complete() (branché temporairement à la place
    de Mistral, voir utils/query_classifier.py) est bien appelé avec des messages au format
    dict (via .format()), pas des objets ChatMessage bruts.
    """
    mock_gemini = MagicMock(return_value="DIRECT - question générale")
    monkeypatch.setattr("utils.query_classifier.gemini_complete", mock_gemini)

    needs_rag, confidence, reason = classifier._classify_with_llm("Explique-moi la photosynthèse")

    mock_gemini.assert_called_once()
    messages = mock_gemini.call_args[0][0]
    # Les messages doivent être des dicts (via .format()), pas des objets ChatMessage bruts
    assert all(isinstance(m, dict) and "role" in m and "content" in m for m in messages)
    assert needs_rag is False

def test_classify_with_llm_parses_rag_response(classifier, monkeypatch):
    monkeypatch.setattr(
        "utils.query_classifier.gemini_complete",
        MagicMock(return_value="RAG - événement précis demandé"),
    )

    needs_rag, confidence, reason = classifier._classify_with_llm("Il y a quoi ce soir ?")

    assert needs_rag is True
    assert confidence == 0.85
