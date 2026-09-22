# Puls-Events — Chatbot RAG de recommandations d'événements culturels

POC pour Puls-Events : un chatbot capable de répondre à des questions sur des événements
culturels (concerts, expositions, festivals...), en s'appuyant sur un système RAG
(Retrieval-Augmented Generation) qui combine une recherche vectorielle hybride (Faiss + BM25)
sur des événements réels (API publique OpenAgenda) et un LLM pour la génération de réponses.

## Architecture en bref

```
API OpenAgenda ──indexation──> Faiss + BM25 (vector_db/) ──recherche──> LLM ──> réponse
                                                                          ↑
                                                   Streamlit (démo) ou API REST (intégration)
```

- **Indexation** (hors ligne, à relancer périodiquement) : télécharge des événements depuis
  l'API OpenAgenda, les découpe en chunks (LangChain `RecursiveCharacterTextSplitter`), génère
  leurs embeddings (Mistral `mistral-embed`) et les stocke localement dans `vector_db/`.
- **Recherche** (à chaque question) : recherche hybride Faiss (dense) + BM25 (mots-clés),
  fusionnées par Reciprocal Rank Fusion, avec filtres déterministes ville/région/mois/année et
  passé/à venir (pas laissés au LLM, peu fiable sur ce type de logique). Cherche uniquement dans
  les données déjà indexées localement — aucun appel à OpenAgenda pendant le chat.
- **Génération** : un LLM rédige la réponse à partir des événements retrouvés.

### Choix technique important : Mistral (embeddings) + Gemini (génération)

Le brief suppose Mistral de bout en bout. En pratique, le compte Mistral utilisé pour ce POC a
été bloqué par leur plateforme spécifiquement sur `chat/completions` (`x-ratelimit-limit-req-minute: 0`,
vérifié sur deux comptes différents, embeddings non affectés). La génération finale utilise donc
l'API Gemini (`google-genai`, modèle `gemini-flash-lite-latest`) à la place. Le code d'appel
Mistral pour la génération est conservé en commentaire dans `utils/chat_service.py` pour un
retour en arrière simple si le blocage est levé. Les embeddings restent sur `mistral-embed`.

### LangChain

Utilisé pour le découpage en chunks (`langchain_text_splitters.RecursiveCharacterTextSplitter`),
pas pour orchestrer la chaîne recherche→LLM : cette partie est écrite à la main dans
`utils/chat_service.py`. Choix assumé, pas un oubli — voir `docs/rapport_technique.md` (section
"Choix technologiques") pour la justification détaillée.

## Prérequis

- Python 3.10+
- Une clé API Mistral ([console.mistral.ai](https://console.mistral.ai/)) — embeddings
- Une clé API Gemini ([aistudio.google.com](https://aistudio.google.com/apikey)) — génération
- Une connexion internet (API OpenAgenda + API Mistral + API Gemini)

## Installation

1. **Cloner le dépôt et créer un environnement virtuel**

```bash
git clone <url-du-repo>
cd <nom-du-repo>
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
```

2. **Installer les dépendances**

```bash
pip install -r requirements.txt
```

3. **Configurer les variables d'environnement**

Créez un fichier `.env` à la racine (jamais versionné, voir `.gitignore`) :

```
MISTRAL_API_KEY=votre_clé_api_mistral
GEMINI_API_KEY=votre_clé_api_gemini
ADMIN_TOKEN=un_token_secret_de_votre_choix   # protège l'endpoint /rebuild de l'API
```

4. **Vérifier une installation "propre"**

```bash
pip install -r requirements.txt --no-cache-dir
python -c "import faiss; from langchain_text_splitters import RecursiveCharacterTextSplitter; from mistralai.client import Mistral; from google import genai; print('imports OK')"
```

## Structure du projet

```
.
├── api/
│   └── puls_events_api.py   # API REST (FastAPI) : /ask, /chat, /rebuild, /health
├── scripts/                  # CLI exécutables (python -m scripts.<nom>)
│   ├── index_by_region.py   # Indexation pondérée par région (script utilisé en production)
│   ├── indexer.py           # Indexation depuis fichiers locaux OU OpenAgenda (une région/ville)
│   ├── migrate_add_region.py # Migration ponctuelle : backfill du champ région sur l'index existant
│   └── evaluate_rag.py      # Évaluation automatique (similarité + juge LLM) sur un jeu annoté
├── tests/
│   ├── test_puls_events.py  # Tests unitaires des modules internes (pytest)
│   └── api_test.py          # Tests fonctionnels de l'API (TestClient, sans lancer de serveur)
├── utils/                     # Bibliothèque cœur, partagée entre API/Streamlit/scripts
│   ├── config.py             # Constantes (clés API, chemins, tailles de chunk, budget de contexte...)
│   ├── openagenda_loader.py  # Récupération et nettoyage des événements OpenAgenda
│   ├── vector_store.py       # Index Faiss + BM25, recherche hybride, filtres, retry sur rate-limit
│   ├── indexing.py           # Logique de (re)construction de l'index, pondérée par région
│   ├── query_classifier.py   # RAG ou réponse directe ? + extraction ville/région/mois/année
│   ├── chat_message.py       # Petit wrapper de message (role/content)
│   ├── chat_service.py       # Logique métier du chat, partagée entre Streamlit et l'API
│   ├── gemini_client.py      # Client Gemini (génération finale)
│   ├── formatting.py         # Formatage des dates en français
│   ├── database.py           # Historique des interactions (SQLite)
│   └── data_loader.py        # Extraction de texte PDF/DOCX/CSV (source "files" de indexer.py)
├── docs/
│   └── rapport_technique.md # Rapport technique (architecture, choix, résultats, limites)
├── pages/
│   └── 1_Feedback_Viewer.py # Dashboard Streamlit des interactions/feedbacks (doit rester sibling
│                              de puls_events_app.py : convention Streamlit multipage)
├── puls_events_app.py        # Interface de démo Streamlit (racine, convention Streamlit)
├── Dockerfile, docker-compose.yml
├── pytest.ini                # pythonpath=. pour que tests/ importe utils/ et api/ correctement
├── vector_db/                 # Index Faiss + chunks (généré par l'indexation, pas versionné)
├── database/                  # Base SQLite des interactions
└── coursework/                # Exercices et notebooks pédagogiques du cours (hors périmètre du POC)
    ├── correction_exercices/
    └── notebooks/
```

## Construire l'index vectoriel

**Option recommandée** — indexation pondérée par région (celle utilisée en pratique) :

```bash
python -m scripts.index_by_region
```

Récupère des événements sur 13 régions françaises, avec un budget plus élevé pour les régions
à grande métropole (Paris, Marseille, Lyon), filtre les agendas non-culturels connus (offres
d'emploi, hébergement touristique), et s'arrête automatiquement si la taille estimée de
l'index dépasse 1 Go (sécurité disque). Prend ~40 minutes.

```bash
python -m scripts.index_by_region --mode extend   # ajoute seulement les nouveaux événements
```

**Option alternative** — une seule région/ville, plus rapide pour tester :

```bash
python -m scripts.indexer --source openagenda --city "Bordeaux" --months-back 12 --max-records 500
```

**Migration ponctuelle** — remplit le champ région des chunks indexés avant son ajout (sans
ré-embedding, patch de métadonnées uniquement) :

```bash
python -m scripts.migrate_add_region
```

## Lancer l'application

**Démo interactive (Streamlit)**

```bash
streamlit run puls_events_app.py
```
→ http://localhost:8501

**API REST (FastAPI)**

```bash
uvicorn api.puls_events_api:app --reload
```
→ Doc interactive (Swagger) : http://localhost:8000/docs

Endpoints principaux :
- `POST /ask` (ou son alias `/chat`) — pose une question (`{"query": "..."}"`), reçoit une
  réponse augmentée
- `POST /rebuild` — relance l'indexation complète en arrière-plan (protégé par le header
  `X-Admin-Token`, doit correspondre à `ADMIN_TOKEN` dans `.env`)
- `GET /rebuild/status` — consulte l'état du dernier rebuild déclenché
- `GET /health` — vérification de disponibilité

## Docker

```bash
docker compose up --build
```

Construit l'image, monte `vector_db/` et `database/` en volumes (pas rebâtis à chaque build de
l'image), expose l'API sur http://localhost:8000. Testé de bout en bout (`/health`, `/docs`,
`/ask`).

## Lancer les tests

```bash
pytest tests/test_puls_events.py     # tests unitaires : fonctions pures, Mistral/Gemini mockés, pas d'appel réseau
pytest tests/api_test.py             # tests fonctionnels de l'API : appels RÉELS à Mistral/Gemini (clés requises)
```

`tests/api_test.py` ne déclenche jamais un vrai `/rebuild` (uniquement le rejet 403 sans token
valide), mais `/ask` et `/chat` appellent réellement le pipeline complet — comptez sur ces tests
pour consommer un peu de quota API à chaque exécution.

## Évaluer la qualité des réponses

```bash
python -m scripts.evaluate_rag
```

Exécute le RAG sur un jeu de 10 questions/réponses annotées manuellement (voir
`scripts/evaluate_rag.py::EVAL_DATASET`, contenu vérifié par recherche directe dans l'index avant
d'écrire chaque référence, pas inventé) et note chaque réponse selon deux signaux : score de
similarité d'embeddings, et verdict d'un juge LLM (Gemini). Alternative à Ragas, dont la chaîne
de dépendances (`langchain_community` → `instructor`) s'est révélée cassée avec les versions
actuellement publiées, y compris en environnement totalement isolé — le brief autorise
explicitement cette alternative ("score de similarité... ou classification manuelle"). Le script
se termine avec un code de sortie non nul si moins de 70 % des cas sont jugés "correcte" ou
"partiellement correcte" par le juge LLM (`QUALITY_GATE_THRESHOLD`), pour pouvoir être utilisé
comme porte de qualité en CI. Résultat le plus récent observé : 9/10 correctes selon le juge LLM.

## Intégration continue (GitHub Actions)

`.github/workflows/ci.yml` définit deux jobs volontairement séparés :

- **`unit-tests`** (à chaque push et pull request) : `pytest tests/test_puls_events.py`. Rapide,
  gratuit, aucun secret requis.
- **`integration-and-evaluation`** (push sur `main`, déclenchement manuel, et chaque lundi) :
  reconstruit l'index vectoriel (mis en cache entre les runs pour éviter les ~40 minutes de
  reconstruction à chaque fois), puis lance `tests/api_test.py` et `scripts/evaluate_rag.py`
  contre de vrais appels API. Volontairement pas déclenché sur chaque push/PR : le quota gratuit
  Gemini (15 requêtes/minute) a été atteint à plusieurs reprises en pratique pendant le
  développement, y compris pour une seule évaluation de 10 questions.

**Secrets GitHub à configurer** (Settings → Secrets and variables → Actions) pour que le second
job fonctionne : `MISTRAL_API_KEY`, `GEMINI_API_KEY`, `ADMIN_TOKEN`.

## Limites connues / prochaines étapes

- L'orchestration recherche→LLM est faite "à la main" en Python, pas via des chaînes LangChain
  (LangChain n'est utilisé que pour le découpage en chunks) — voir `docs/rapport_technique.md`.
- Le jeu de test annoté (`scripts/evaluate_rag.py::EVAL_DATASET`, 10 cas) reste modeste pour une
  mesure statistiquement robuste de la qualité globale — un cas est actuellement en échec (voir
  `docs/rapport_technique.md`, section Évaluation).
- Dépendance à deux APIs externes avec quotas gratuits limités (Gemini : 15 req/min sur le
  tier gratuit, déjà rencontré en pratique, y compris en CI) — d'où le retry avec délai réel
  (`utils/gemini_client.py::_extract_retry_delay_seconds`) et des déclencheurs CI espacés plutôt
  que sur chaque commit.
- `vector_db/` n'est pas versionné (dépasse la limite GitHub de 100 Mo/fichier) : à reconstruire
  après clonage, ou via le cache du job CI `integration-and-evaluation`.
