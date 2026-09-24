# Rapport technique — Puls-Events, assistant RAG d'événements culturels

**Projet** : POC RAG pour Puls-Events
**Source de données** : API publique OpenAgenda
**Statut** : POC fonctionnel — indexation, recherche, génération, API REST et Docker opérationnels

---

## 1. Contexte et objectifs

Puls-Events souhaite démontrer à ses équipes produit et marketing la faisabilité d'un chatbot
capable de répondre à des questions sur des événements culturels en s'appuyant sur les données
réelles de l'API OpenAgenda, plutôt que sur la connaissance générale d'un LLM (qui ne connaît ni
les événements ni leurs dates).

Le principe RAG (Retrieval-Augmented Generation) répond directement à ce besoin : au lieu de
demander au modèle de langage d'inventer une réponse, on lui fournit d'abord les événements réels
les plus pertinents pour la question posée, puis on lui demande de rédiger sa réponse à partir de
ces documents. Cela réduit fortement le risque d'invention ("hallucination") et permet de citer
des sources vérifiables (lien OpenAgenda de chaque événement).

## 2. Architecture générale

```
API OpenAgenda ──indexation──> Faiss + BM25 (vector_db/) ──recherche──> LLM ──> réponse
                                                                          ↑
                                                   Streamlit (démo) ou API REST (intégration)
```

Le système se décompose en trois étapes, détaillées ci-dessous : **indexation** (hors ligne),
**recherche** (à chaque question) et **génération** (à chaque question).

### 2.1 Indexation

- Récupération des événements via l'API publique OpenAgenda (`utils/openagenda_loader.py`),
  filtrés par ancienneté (12 mois par défaut, sans limite dans le futur) et par agendas connus
  comme non-culturels (offres d'emploi, hébergement touristique).
- Découpage en chunks avec `langchain_text_splitters.RecursiveCharacterTextSplitter`
  (1500 caractères, 150 de chevauchement).
- Génération des embeddings avec `mistral-embed` (API Mistral) et indexation dans Faiss
  (`IndexFlatIP`, similarité cosinus).
- Indexation pondérée par région (`scripts/index_by_region.py`) : 13 régions françaises, budget
  plus élevé pour les régions à grande métropole, avec une sécurité d'arrêt si l'index dépasse
  1 Go sur disque.
- **Volumétrie actuelle** : 149 284 chunks indexés, 131 210 événements uniques.

### 2.2 Recherche : hybride, filtrée, à budget adaptatif

**Recherche hybride (Faiss + BM25).** Une recherche purement dense (embeddings) peut diluer un mot-clé
ou un nom propre précis dans le sens global d'un document plus long, surtout si la requête ne
partage aucun mot exact avec le texte indexé. C'est le risque théorique qui justifie de ne pas se
reposer sur le dense seul : une recherche sparse (BM25, correspondance de mots-clés) reste
insensible à ce problème puisqu'elle ne regarde que les mots effectivement présents. Sur le jeu de
données OpenAgenda actuel, ce risque ne s'est pas matérialisé de façon flagrante en test manuel
(`mistral-embed` retrouve bien les noms propres testés) - la robustesse vient plutôt de la garantie
que les deux méthodes se complètent sans jamais se nuire : BM25 rattrape le dense sur les cas où
il se tromperait, sans coût dans les cas où le dense se débrouille déjà bien. Le système combine
donc une recherche dense (Faiss, similarité cosinus) et une recherche sparse (BM25, correspondance
de mots-clés), fusionnées par Reciprocal Rank Fusion (RRF) :

```
score(document) = 1/(60 + rang_dense + 1) + 1/(60 + rang_bm25 + 1)
```

**Filtres déterministes, pas laissés au LLM.** Ville, région, mois, année et exclusion des
événements passés sont extraits de la question (via un appel léger au LLM pour l'extraction
d'entités) puis appliqués comme filtres Python stricts sur les métadonnées — jamais laissés au
LLM pour comparer des dates lui-même. Ce choix vient d'un bug observé en pratique : le LLM
affirmait qu'une date de février était "après le 1er septembre". Les comparaisons de dates sont
systématiquement calculées en Python (`datetime`), jamais "raisonnées" par le modèle.

**Budget de contexte adaptatif plutôt qu'un nombre de documents fixe.** Un plafond fixe en nombre
de documents (`k`) s'est révélé arbitraire dans les deux sens : trop bas, il exclut des résultats
pertinents dès qu'un filtre ville/mois réduit le pool de candidats (mesuré : un concert réel classé
14e sur 52 candidats filtrés restait hors d'un top 5) ; trop haut ou illimité, il explose sur une
question large (mesuré : **13 242 résultats** franchissent le seuil de qualité minimal pour
"concerts de musique classique en France" sans aucun filtre ville/date, ce qui enverrait des
millions de tokens de contexte). Le système accumule donc les résultats par ordre de pertinence
jusqu'à un budget de ~24 000 caractères (`MAX_CONTEXT_CHARS`, `utils/config.py`), ce qui laisse
passer tous les résultats pertinents d'une recherche filtrée tout en bornant le coût et le temps
de réponse sur une question ouverte.

### 2.3 Génération

Le LLM reçoit le contexte documentaire (événements trouvés, avec ville/date/lien) et rédige une
réponse en langage naturel, avec citation des sources.

**Choix technique assumé — Mistral (embeddings) + Gemini (génération).** Le brief suppose Mistral
de bout en bout. En cours de développement, le compte Mistral utilisé s'est retrouvé bloqué par
leur plateforme spécifiquement sur l'endpoint `chat/completions` (`x-ratelimit-limit-req-minute: 0`
constaté dans les en-têtes de réponse HTTP, vérifié sur deux comptes Mistral différents avec des
niveaux d'usage différents — écartant une simple question de quota épuisé). Les embeddings
(`mistral-embed`) ne sont pas affectés et restent utilisés. La génération finale bascule donc sur
l'API Gemini (`google-genai`, modèle `gemini-flash-lite-latest`). Le code d'appel Mistral pour la
génération est conservé en commentaire dans `utils/chat_service.py` pour permettre un retour en
arrière simple si le blocage est levé.

## 3. Choix technologiques

| Composant | Choix | Justification |
|---|---|---|
| Recherche vectorielle | Faiss (`IndexFlatIP`) | Similarité cosinus, gratuit, local, pas de dépendance à un service tiers pour la recherche |
| Recherche mots-clés | `rank-bm25` | Complète Faiss sur les entités nommées et le vocabulaire exact |
| Découpage en chunks | LangChain (`RecursiveCharacterTextSplitter`) | Composant isolé et réutilisable, pas besoin de le réécrire |
| Embeddings | Mistral (`mistral-embed`) | Non affecté par le blocage du compte, qualité suffisante en pratique |
| Génération | Gemini (`gemini-flash-lite-latest`) | Pivot nécessaire (voir 2.3), quota gratuit correct hors pics d'usage intensif |
| Orchestration RAG | Python "à la main" (`utils/chat_service.py`), pas de chaînes LangChain | Voir ci-dessous |
| API | FastAPI | Documentation Swagger automatique, validation Pydantic native |
| Conteneurisation | Docker + Docker Compose | Portabilité, démo reproductible en local |

**Pourquoi pas de chaînes LangChain pour l'orchestration.** Le brief recommande d'utiliser
LangChain pour orchestrer les appels entre Faiss et le LLM. Ce projet utilise LangChain pour le
découpage en chunks, mais pas pour la chaîne recherche→prompt→génération, qui est écrite
directement en Python. Deux raisons concrètes rencontrées en pratique pendant le développement :

1. **Contrôle fin nécessaire.** Le système applique plusieurs étapes de logique métier successives
   et interdépendantes (décomposition d'une question en plusieurs thèmes, filtrage déterministe
   par date, repli sur les événements passés si aucun futur n'est trouvé, budget de contexte
   adaptatif) qui demandent une visibilité complète sur chaque étape intermédiaire pour les
   déboguer — plusieurs bugs réels de cette nature ont été trouvés et corrigés en observant
   directement les résultats intermédiaires, ce qu'une chaîne LangChain opaque aurait rendu plus
   difficile à diagnostiquer.
2. **Risque de dépendances.** Une tentative d'intégrer Ragas (qui s'appuie sur `langchain_community`)
   a révélé une chaîne de dépendances cassée avec les versions actuellement publiées sur PyPI
   (`instructor` importe un module qui n'existe plus dans `mistralai` v2, puis `ragas.llms.base`
   importe une classe déplacée hors de `langchain_community`) — reproduit dans un environnement
   totalement isolé, donc pas spécifique à ce projet. Cette expérience a pesé dans la décision de
   limiter la dépendance à l'écosystème LangChain au strict nécessaire.

C'est un choix d'architecture assumé, pas un oubli, mais c'est un écart réel par rapport à la
lettre du brief à mentionner clairement en soutenance.

## 4. API REST

FastAPI, avec documentation interactive automatique (`/docs`). Endpoints :

- `POST /ask` — pose une question, reçoit une réponse augmentée
- `POST /rebuild` — relance l'indexation complète en tâche de fond, protégé par un header
  `X-Admin-Token` (recommandation du brief suivie : endpoint sensible protégé)
- `GET /rebuild/status` — état du dernier rebuild déclenché
- `GET /health` — vérification de disponibilité

Testée via `tests/api_test.py` (TestClient, sans lancer de serveur réel) et manuellement via
conteneur Docker (voir section 6).

## 5. Évaluation

**Pourquoi pas Ragas.** Voir section 3 — chaîne de dépendances cassée, reproduite en environnement
isolé, indépendamment des choix de versions propres à ce projet. Le brief autorise explicitement
une alternative ("score de similarité, correspondance exacte, ou classification manuelle").

**Méthode retenue** (`scripts/evaluate_rag.py`) : deux signaux complémentaires par question, sur
un jeu de questions/réponses annotées manuellement à partir de contenu réellement présent dans
l'index (vérifié par recherche directe avant d'écrire le jeu de test, pas inventé) :

1. **Score de similarité cosinus** entre les embeddings de la réponse générée et de la référence.
   Rapide et gratuit, mais peut pénaliser une réponse correcte mais formulée différemment.
2. **Verdict d'un juge LLM** (Gemini) qui compare le sens des deux réponses, moins sensible à la
   formulation — équivalent fonctionnel à ce que Ragas aurait fait avec ses métriques
   "faithfulness"/"answer correctness".

Seuils de similarité calibrés empiriquement (pas choisis arbitrairement) en comparant une
référence à une reformulation quasi-identique (0.995), une paraphrase correcte (0.935), une
réponse partielle (0.909), une réponse fausse (0.814) et une réponse hors-sujet (0.772) : ≥ 0.92
= "correcte", ≥ 0.85 = "partiellement correcte", en dessous = "incorrecte".

**Jeu de test** (`scripts/evaluate_rag.py::EVAL_DATASET`) : 10 cas couvrant des angles différents
- question simple ville+thème, mois/année explicites, région seule (sans ville), multi-critères
et multi-thèmes (décomposition en facettes), question à formulation temporelle floue ("dans les
prochaines semaines" - non-régression sur un bug corrigé), question au passé avec inversion
interrogative (non-régression sur un second bug corrigé), cas piège hors-sujet, et question
générale hors-RAG (vérifie le routage `needs_rag()`).

**Résultats de la dernière exécution complète** (10/10 cas exécutés avec succès, retry automatique
sur les 429 du quota gratuit Gemini - voir section 8) :

| Signal | Correcte | Partiellement correcte | Incorrecte |
|---|---|---|---|
| Similarité cosinus | 2/10 | 6/10 | 2/10 |
| Juge LLM | 9/10 | 0/10 | 1/10 |

Score de similarité moyen : 0.885. Seuil de qualité CI (`QUALITY_GATE_THRESHOLD` = 70 % de cas non
"incorrecte" selon le juge LLM) : **90 % obtenu, seuil franchi**.

**Pourquoi les deux signaux ne sont pas toujours d'accord** — sur les 10 cas, ils divergent 3 fois.
Dans 2 cas, c'est la similarité qui se trompe (elle sous-note une réponse en réalité correcte) ;
dans 1 cas, c'est elle qui est trop indulgente et c'est le juge LLM qui détecte le vrai problème.

- **La similarité sous-note deux réponses pourtant correctes.** Sur le cas piège Antarctique
  (*"Quels événements de plongée sous-marine polaire y a-t-il en Antarctique ?"*), le système
  répond correctement qu'aucun événement de ce type n'existe — mais avec des mots différents de
  la référence écrite pour ce jeu de test. Résultat : similarité 0.796 ("incorrecte", sous le
  seuil de 0.85) alors que le juge LLM, qui lit et compare le sens des deux réponses plutôt que
  leur formulation, la juge à raison "correcte". Même chose sur la question de l'exposition photo
  à Paris en novembre 2026 : le système cite le bon événement (l'exposition de Sam Samore) mais
  développe davantage sa réponse que la référence, ce qui fait chuter la similarité à 0.820
  ("incorrecte") sans que ce soit justifié. Dans les deux cas, la similarité mesure une proximité
  de vocabulaire, pas si l'information est juste — c'est exactement sa limite connue (voir le
  point 1 ci-dessus), et ces deux cas le montrent concrètement plutôt que de rester théorique.

- **Le juge LLM détecte un vrai problème que la similarité avait laissé passer.** Sur la question
  *"Quand a lieu l'exposition du prix de Paris à Lyon ?"*, le système répond avec une exposition
  et une date différentes de celles attendues. La similarité ne le voit pas clairement (elle note
  quand même 0.890, "partiellement correcte", sans doute parce que la réponse ressemble à la
  référence dans sa forme : ville, type d'événement, structure de phrase) mais le juge LLM, en
  comparant le contenu factuel, repère l'erreur et note "incorrecte" à raison. C'est le seul des
  10 cas où le système donne une réponse réellement fausse. Cause probable, pas encore vérifiée
  en détail : plusieurs expositions similaires ("prix de Paris" ou proches) coexistent peut-être
  dans l'index, et rien dans le pipeline actuel ne garantit de choisir la bonne quand plusieurs se
  ressemblent — à investiguer, piste listée en section 10.

## 6. Conteneurisation Docker

`Dockerfile` (image `python:3.14-slim`, dépendances installées avant copie du code pour profiter
du cache Docker) + `docker-compose.yml` (montage de `vector_db/` et `database/` en volumes,
plutôt que de les intégrer à l'image — permet de mettre à jour les données sans rebuild).

Testé de bout en bout : build complet, démarrage du conteneur, `/health` et `/ask` répondent
correctement avec de vrais appels Faiss/BM25/Gemini depuis l'intérieur du conteneur.

## 7. Intégration continue (CI)

`.github/workflows/ci.yml`, deux jobs :

- **`unit-tests`** (chaque push et pull request) : `pytest tests/test_puls_events.py` uniquement
  — fonctions pures, Mistral/Gemini mockés, pas d'appel réseau, aucun secret requis.
- **`integration-and-evaluation`** (push sur `main`, déclenchement manuel, et chaque lundi via
  `schedule`) : reconstruit l'index vectoriel (mis en cache via `actions/cache` entre les runs),
  puis exécute `tests/api_test.py` et `scripts/evaluate_rag.py` avec de vrais appels API.

**Pourquoi deux déclencheurs différents plutôt qu'un seul job systématique.** Deux contraintes
concrètes rencontrées pendant le développement, pas une précaution générique :

1. `vector_db/` (l'index Faiss + les chunks) a été retiré du suivi git en cours de projet après
   avoir découvert que `faiss_index.idx` atteint plusieurs centaines de Mo — largement au-dessus
   de la limite de 100 Mo par fichier imposée par GitHub sans Git LFS (**le dernier commit local
   contenant ces fichiers n'avait d'ailleurs jamais pu être poussé sur le dépôt distant**). L'index
   doit donc être reconstruit pour toute exécution CI qui en a besoin, ce qui prend ~40 minutes et
   consomme des appels Mistral réels — d'où le cache.
2. Le quota gratuit Gemini (15 requêtes/minute) a été atteint à plusieurs reprises en pratique
   pendant le développement, y compris pour une seule évaluation de 10 questions (chaque question
   du pipeline déclenchant ~5-7 appels Gemini légers en plus de la génération finale) — le lancer
   sur chaque push/PR le rendrait systématiquement lent et parfois instable, sans bénéfice
   proportionné pour un simple relecture de code.

**Secrets requis** (GitHub → Settings → Secrets and variables → Actions) : `MISTRAL_API_KEY`,
`GEMINI_API_KEY`, `ADMIN_TOKEN`.

**Porte de qualité automatisée** : `scripts/evaluate_rag.py` se termine avec un code de sortie non
nul si moins de 70 % des cas ne sont pas jugés "incorrecte" par le juge LLM (`check_quality_gate()`,
`QUALITY_GATE_THRESHOLD`) — le job `integration-and-evaluation` échoue donc automatiquement si la
qualité des réponses se dégrade, satisfaisant "l'automatisation des métriques d'évaluation"
demandée par le brief.

## 8. Bugs réels trouvés et corrigés (méthodologie de diagnostic)

Plusieurs bugs non triviaux ont été trouvés en comparant le comportement du système à des données
vérifiées manuellement, plutôt qu'en supposant que le code fonctionnait :

- **Filtre mois insensible à l'année** : "en octobre" retournait des événements d'octobre de
  n'importe quelle année passée. Corrigé en donnant la date du jour au LLM d'extraction pour
  qu'il infère l'année la plus probable.
- **Décomposition de questions multi-thèmes dégradant les questions à thème unique** : le LLM de
  reformulation paraphrasait même les questions à un seul thème en une forme plus courte et moins
  bien classée par l'embedding (mesuré : 82.2% pour une question naturelle complète contre 74.8%
  pour sa version "mots-clés" compressée, sur le même événement cible). Corrigé en forçant le
  système à toujours garder la question originale inchangée quand un seul thème est détecté.
- **Détection des questions "au passé" cassée par l'inversion interrogative française** : la
  détection de mots-clés cherchait la sous-chaîne exacte "a eu lieu", qui ne correspond pas à sa
  forme interrogative "a-t-il eu lieu" (bug de sous-chaîne). Corrigé en utilisant des sous-chaînes
  plus robustes ("eu lieu", "déroulé") qui couvrent les deux formes sans introduire de faux
  positifs sur des questions au futur ("aura lieu", "se déroulera").
- **Repli passé/futur trop agressif** : un mécanisme de repli comparait les scores avec et sans
  exclusion du passé et bascule vers le passé dès que le score était ne serait-ce qu'un dixième de
  point plus haut — au point d'écarter 45 résultats futurs pertinents au profit de 2 événements
  déjà terminés pour un écart de seulement 3,8 points. Remplacé par une règle plus simple et plus
  sûre : ne basculer sur le passé que si aucun résultat futur n'est trouvé.
- **Les erreurs 429 (quota Gemini) n'étaient pas interceptées du tout** : le code de retry ne
  gérait que `ServerError` et les erreurs réseau bas niveau, alors qu'un 429 lève une `ClientError`
  distincte — tout dépassement de quota faisait donc planter la requête immédiatement, sans aucune
  tentative. Corrigé en interceptant `ClientError` sur le code 429 et en lisant le délai d'attente
  exact indiqué par Gemini lui-même (`RetryInfo.retryDelay` dans le corps de la réponse) plutôt que
  de deviner un backoff exponentiel arbitraire — vérifié en conditions réelles : un enchaînement de
  20 appels qui échouait avant intégralement se termine maintenant avec succès, en attendant
  automatiquement le temps exact nécessaire (jusqu'à ~56 secondes observées).

### 8.1 Réduction du code par tests destructifs

Méthode : pour chaque candidat suspecté redondant, suppression réelle du code puis exécution de
la suite de tests complète — si rien ne casse, c'est une preuve, pas une supposition. Trois
suppressions confirmées de cette façon (~282 lignes de Python retirées, 21/21 tests toujours au
vert après chaque suppression) :

- **Endpoint `/chat`** (`api/puls_events_api.py`) : alias de `/ask` conservé "pour compatibilité"
  mais sans aucun consommateur réel (ni la démo Streamlit, qui appelle `answer_query()` directement
  en Python, ni aucun autre code) - seul son propre test l'utilisait.
- **`get_db()`** (`utils/database.py`) : fonction de dépendance FastAPI classique (`yield` d'une
  session SQLAlchemy) jamais câblée à une route via `Depends()` - `log_interaction()` gère sa
  propre session directement. Zéro appel externe trouvé.
- **`utils/data_loader.py` et le mode `--source files` de `scripts/indexer.py`** : hérités du tout
  premier prototype (extraction PDF/DOCX/CSV pour le projet mairie, avant le pivot vers
  OpenAgenda) - le dossier `inputs/` qu'ils traitaient n'existe même plus. Suppression en cascade
  de 3 dépendances devenues orphelines (`PyPDF2`, `python-docx`, `openpyxl`) de `requirements.txt`.

## 9. Limites connues

- Pas d'orchestration LangChain pour la chaîne recherche→génération (voir section 3).
- Un cas du jeu de test échoue actuellement (question sur une exposition à Lyon, mauvaise
  exposition/date retournée) — non encore investigué en détail, voir section 5 et section 10.
- Dépendance à deux APIs externes avec quotas gratuits limités — le quota Gemini (15 req/min sur
  le tier gratuit) a été atteint très régulièrement pendant le développement et l'évaluation,
  chaque question du pipeline déclenchant ~5-7 appels LLM légers (classification, extraction
  ville/région/mois, décomposition de facettes, génération) ; désormais absorbé par un retry qui
  respecte le délai réel indiqué par Gemini (voir section 8), au prix d'un temps de réponse parfois
  long en cas de pic d'usage.
- Le mapping ville→région utilisé pour la migration de métadonnées (`scripts/migrate_add_region.py`)
  est un vote majoritaire par ville, pas une vérité absolue : 2 villes sur 12 545 n'ont pas été
  résolues (communes rares, absentes du référentiel OpenAgenda interrogé).
- `vector_db/` n'est plus versionné dans git (voir section 7) : un clone frais du dépôt nécessite
  de reconstruire l'index avant d'être testable (~40 minutes), en tension avec l'objectif du brief
  de "pouvoir tester rapidement" la solution. Le job CI en cache une copie ; un évaluateur humain
  clonant le dépôt doit reconstruire lui-même, ou récupérer l'index par un autre moyen (à discuter).

## 10. Pistes d'amélioration

- Investiguer et corriger le cas d'échec identifié en section 5 (exposition à Lyon) — vérifier si
  plusieurs expositions similaires coexistent dans l'index et si le score minimal ou le classement
  hybride doivent être ajustés pour ce type de question.
- Continuer à étoffer le jeu de test annoté au-delà de 10 cas à mesure que de nouveaux types de
  questions ou de bugs sont identifiés en production.
- Explorer un modèle Gemini payant (ou un second fournisseur en repli) pour la CI et un usage plus
  intensif, le tier gratuit (15 req/min) restant la principale source de lenteur et d'instabilité
  malgré le retry avec délai réel mis en place.
- Explorer une orchestration LangChain légère (ex: `LangGraph`, déjà présent comme dépendance
  transitive) pour les parties du pipeline les plus stables, en gardant le contrôle fin sur les
  étapes encore en évolution rapide.
- Décider d'une stratégie de distribution de l'index vectoriel pour un clone frais du dépôt
  (release GitHub avec l'archive en pièce jointe, Git LFS, ou stockage objet externe) plutôt que
  d'imposer une reconstruction de 40 minutes à chaque nouvel évaluateur.
