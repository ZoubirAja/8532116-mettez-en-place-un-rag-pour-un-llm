# tests/api_test.py
"""
Tests fonctionnels de l'API Puls-Events, via TestClient (appelle l'app FastAPI directement,
sans avoir besoin de lancer uvicorn séparément).

Note importante sur /rebuild : ce script teste UNIQUEMENT le rejet (403) sans token valide.
Il ne teste jamais un rebuild avec un token valide, pour ne pas déclencher accidentellement
une reconstruction complète de l'index (~40 min, coûteux en appels API Mistral) en lançant
ce script de test.
"""

from fastapi.testclient import TestClient

from api.puls_events_api import app

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_ask_valid_question():
    response = client.post("/ask", json={"query": "Quels concerts à Bordeaux ?", "num_docs": 3})
    assert response.status_code == 200
    data = response.json()
    assert "response" in data
    assert "sources" in data
    assert data["mode"] in ("RAG", "DIRECT")


def test_ask_empty_query_returns_422():
    response = client.post("/ask", json={"query": ""})
    assert response.status_code == 422


def test_ask_missing_query_returns_422():
    response = client.post("/ask", json={})
    assert response.status_code == 422


def test_ask_conversation_history_is_accepted():
    history = [
        {"role": "user", "content": "Des événements de volley ?"},
        {"role": "assistant", "content": "Oui, plusieurs à venir."},
    ]
    response = client.post("/ask", json={"query": "Et à Paris ?", "conversation_history": history})
    assert response.status_code == 200


def test_rebuild_without_token_is_rejected():
    response = client.post("/rebuild")
    assert response.status_code == 403


def test_rebuild_with_wrong_token_is_rejected():
    response = client.post("/rebuild", headers={"X-Admin-Token": "mauvais-token"})
    assert response.status_code == 403


def test_rebuild_status_is_readable():
    response = client.get("/rebuild/status")
    assert response.status_code == 200
    assert "state" in response.json()


if __name__ == "__main__":
    import sys

    tests = [obj for name, obj in list(globals().items()) if name.startswith("test_") and callable(obj)]
    failures = 0
    for test in tests:
        try:
            test()
            print(f"OK   {test.__name__}")
        except AssertionError as e:
            failures += 1
            print(f"FAIL {test.__name__}: {e}")
    print(f"\n{len(tests) - failures}/{len(tests)} tests passés")
    sys.exit(1 if failures else 0)
