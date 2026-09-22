# utils/vector_store.py
import os
import re
import time
import datetime
import pickle
import faiss
import numpy as np
import logging
from typing import List, Dict, Tuple, Optional
from mistralai.client import Mistral
from mistralai.extra.exceptions import MistralClientException
from langchain_text_splitters.character import RecursiveCharacterTextSplitter
from langchain_core.documents import Document # Utilisé pour le format attendu par le splitter
from rank_bm25 import BM25Okapi

# Constante de la formule RRF (Reciprocal Rank Fusion) - 60 est la valeur standard de la littérature
RRF_K = 60

from .config import (
    MISTRAL_API_KEY, EMBEDDING_MODEL, EMBEDDING_BATCH_SIZE,
    FAISS_INDEX_FILE, DOCUMENT_CHUNKS_FILE, CHUNK_SIZE, CHUNK_OVERLAP
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class VectorStoreManager:
    """Gère la création, le chargement et la recherche dans un index Faiss."""

    def __init__(self):
        self.index: Optional[faiss.Index] = None
        self.document_chunks: List[Dict[str, any]] = []
        self.bm25: Optional[BM25Okapi] = None
        self.mistral_client = Mistral(api_key=MISTRAL_API_KEY)
        self._load_index_and_chunks()

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Découpe un texte en mots pour BM25 (minuscules, sans ponctuation)."""
        return re.findall(r"\w+", text.lower())

    def _build_bm25_index(self):
        """Construit l'index BM25 (recherche par mots-clés) à partir des chunks chargés."""
        if not self.document_chunks:
            self.bm25 = None
            return
        tokenized_corpus = [self._tokenize(chunk["text"]) for chunk in self.document_chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def _load_index_and_chunks(self):
        """Charge l'index Faiss et les chunks si les fichiers existent."""
        if os.path.exists(FAISS_INDEX_FILE) and os.path.exists(DOCUMENT_CHUNKS_FILE):
            try:
                logging.info(f"Chargement de l'index Faiss depuis {FAISS_INDEX_FILE}...")
                self.index = faiss.read_index(FAISS_INDEX_FILE)
                logging.info(f"Chargement des chunks depuis {DOCUMENT_CHUNKS_FILE}...")
                with open(DOCUMENT_CHUNKS_FILE, 'rb') as f:
                    self.document_chunks = pickle.load(f)
                logging.info(f"Index ({self.index.ntotal} vecteurs) et {len(self.document_chunks)} chunks chargés.")
                self._build_bm25_index()
            except Exception as e:
                logging.error(f"Erreur lors du chargement de l'index/chunks: {e}")
                self.index = None
                self.document_chunks = []
        else:
            logging.warning("Fichiers d'index Faiss ou de chunks non trouvés. L'index est vide.")

    def _split_documents_to_chunks(self, documents: List[Dict[str, any]]) -> List[Dict[str, any]]:
        """Découpe les documents en chunks avec métadonnées."""
        logging.info(f"Découpage de {len(documents)} documents en chunks (taille={CHUNK_SIZE}, chevauchement={CHUNK_OVERLAP})...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len, # Important: mesure en caractères
            add_start_index=True, # Ajoute la position de début du chunk dans le document original
        )

        all_chunks = []
        doc_counter = 0
        for doc in documents:
            # Convertit notre format de document en format Langchain Document pour le splitter
            langchain_doc = Document(page_content=doc["page_content"], metadata=doc["metadata"])
            chunks = text_splitter.split_documents([langchain_doc])
            logging.info(f"  Document '{doc['metadata'].get('filename', 'N/A')}' découpé en {len(chunks)} chunks.")

            # Enrichit chaque chunk avec des métadonnées supplémentaires
            for i, chunk in enumerate(chunks):
                chunk_dict = {
                    "id": f"{doc_counter}_{i}", # Identifiant unique du chunk (doc_index_chunk_index)
                    "text": chunk.page_content,
                    "metadata": {
                        **chunk.metadata, # Métadonnées héritées du document (source, category, etc.)
                        "chunk_id_in_doc": i, # Position du chunk dans son document d'origine
                        "start_index": chunk.metadata.get("start_index", -1) # Position de début (en caractères)
                    }
                }
                all_chunks.append(chunk_dict)
            doc_counter += 1

        logging.info(f"Total de {len(all_chunks)} chunks créés.")
        return all_chunks

    @staticmethod
    def _get_status_code(exception: Exception) -> Optional[int]:
        """Extrait le code HTTP d'une exception SDKError Mistral, si disponible."""
        for arg in exception.args:
            status = getattr(arg, "status_code", None)
            if status is not None:
                return status
        return None

    def _generate_embeddings(self, chunks: List[Dict[str, any]]) -> Optional[np.ndarray]:
        """Génère les embeddings pour une liste de chunks via l'API Mistral."""
        if not MISTRAL_API_KEY:
            logging.error("Impossible de générer les embeddings: MISTRAL_API_KEY manquante.")
            return None
        if not chunks:
            logging.warning("Aucun chunk fourni pour générer les embeddings.")
            return None

        logging.info(f"Génération des embeddings pour {len(chunks)} chunks (modèle: {EMBEDDING_MODEL})...")
        all_embeddings = []
        total_batches = (len(chunks) + EMBEDDING_BATCH_SIZE - 1) // EMBEDDING_BATCH_SIZE
        max_retries = 5

        for i in range(0, len(chunks), EMBEDDING_BATCH_SIZE):
            batch_num = (i // EMBEDDING_BATCH_SIZE) + 1
            batch_chunks = chunks[i:i + EMBEDDING_BATCH_SIZE]
            texts_to_embed = [chunk["text"] for chunk in batch_chunks]

            logging.info(f"  Traitement du lot {batch_num}/{total_batches} ({len(texts_to_embed)} chunks)")

            batch_embeddings = None
            for attempt in range(max_retries):
                try:
                    response = self.mistral_client.embeddings.create(
                        model=EMBEDDING_MODEL,
                        inputs=texts_to_embed
                    )
                    batch_embeddings = [data.embedding for data in response.data]
                    break  # succès
                except Exception as e:
                    status_code = self._get_status_code(e)
                    if status_code == 429 and attempt < max_retries - 1:
                        wait_time = 2 ** (attempt + 1)  # 2s, 4s, 8s, 16s, 32s
                        logging.warning(
                            f"  Rate limit (429) sur le lot {batch_num}, nouvelle tentative dans "
                            f"{wait_time}s (essai {attempt + 1}/{max_retries})..."
                        )
                        time.sleep(wait_time)
                        continue
                    logging.error(f"Erreur lors de la génération d'embeddings (lot {batch_num}, tentative {attempt + 1}): {e}")
                    break  # échec définitif : pas un 429, ou plus de tentatives disponibles

            if batch_embeddings is not None:
                all_embeddings.extend(batch_embeddings)
            else:
                # Échec définitif du lot après épuisement des tentatives : vecteurs nuls pour ne pas bloquer
                if all_embeddings:
                    dim = len(all_embeddings[0])
                    logging.warning(f"Ajout de {len(texts_to_embed)} vecteurs nuls de dimension {dim} pour le lot {batch_num} (échec définitif).")
                    all_embeddings.extend([np.zeros(dim, dtype='float32')] * len(texts_to_embed))
                else:
                    logging.error(f"Impossible de déterminer la dimension des embeddings, lot {batch_num} perdu.")


        if not all_embeddings:
             logging.error("Aucun embedding n'a pu être généré.")
             return None

        embeddings_array = np.array(all_embeddings).astype('float32')
        logging.info(f"Embeddings générés avec succès. Shape: {embeddings_array.shape}")
        return embeddings_array

    def build_index(self, documents: List[Dict[str, any]]):
        """Construit l'index Faiss à partir des documents."""
        if not documents:
            logging.warning("Aucun document fourni pour construire l'index.")
            return

        # 1. Découper en chunks
        self.document_chunks = self._split_documents_to_chunks(documents)
        if not self.document_chunks:
            logging.error("Le découpage n'a produit aucun chunk. Impossible de construire l'index.")
            return

        # 2. Générer les embeddings
        embeddings = self._generate_embeddings(self.document_chunks)
        if embeddings is None or embeddings.shape[0] != len(self.document_chunks):
            logging.error("Problème de génération d'embeddings. Le nombre d'embeddings ne correspond pas au nombre de chunks.")
            # Nettoyer pour éviter un état incohérent
            self.document_chunks = []
            self.index = None
            # Supprimer les fichiers potentiellement corrompus
            if os.path.exists(FAISS_INDEX_FILE): os.remove(FAISS_INDEX_FILE)
            if os.path.exists(DOCUMENT_CHUNKS_FILE): os.remove(DOCUMENT_CHUNKS_FILE)
            return


        # 3. Créer l'index Faiss optimisé pour la similarité cosinus
        dimension = embeddings.shape[1]
        logging.info(f"Création de l'index Faiss optimisé pour la similarité cosinus avec dimension {dimension}...")

        # Normaliser les embeddings pour la similarité cosinus
        faiss.normalize_L2(embeddings)

        # Créer un index pour la similarité cosinus (IndexFlatIP = produit scalaire)
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(embeddings)
        logging.info(f"Index Faiss créé avec {self.index.ntotal} vecteurs.")

        # 4. Construire l'index BM25 (recherche par mots-clés) en complément du Faiss
        self._build_bm25_index()

        # 5. Sauvegarder l'index et les chunks
        self._save_index_and_chunks()

    def add_chunks(self, new_chunks: List[Dict[str, any]]):
        """
        Ajoute des chunks à l'index EXISTANT, sans réembeder ce qui y est déjà.
        Contrairement à build_index(), ne repart pas de zéro : n'appelle l'API
        d'embeddings que sur `new_chunks`.

        Args:
            new_chunks: chunks déjà découpés (format identique à _split_documents_to_chunks),
                        pas encore présents dans self.document_chunks.
        """
        if not new_chunks:
            logging.warning("Aucun nouveau chunk à ajouter.")
            return

        # 1. Embeddings UNIQUEMENT pour les nouveaux chunks (coûteux en API, donc pas sur l'existant)
        new_embeddings = self._generate_embeddings(new_chunks)
        if new_embeddings is None or new_embeddings.shape[0] != len(new_chunks):
            logging.error("Problème de génération d'embeddings pour les nouveaux chunks. Ajout annulé.")
            return

        faiss.normalize_L2(new_embeddings)

        # 2. Ajout incrémental à l'index Faiss existant (IndexFlatIP supporte .add() répété)
        if self.index is None:
            dimension = new_embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dimension)
        self.index.add(new_embeddings)
        self.document_chunks.extend(new_chunks)
        logging.info(f"{len(new_chunks)} chunks ajoutés. Index Faiss: {self.index.ntotal} vecteurs au total.")

        # 3. BM25 doit être reconstruit en entier (pas d'ajout incrémental dans rank_bm25),
        # mais c'est gratuit : pas d'appel API, juste retokeniser le texte.
        self._build_bm25_index()

        # 4. Sauvegarde
        self._save_index_and_chunks()

    def _save_index_and_chunks(self):
        """Sauvegarde l'index Faiss et la liste des chunks."""
        if self.index is None or not self.document_chunks:
            logging.warning("Tentative de sauvegarde d'un index ou de chunks vides.")
            return

        os.makedirs(os.path.dirname(FAISS_INDEX_FILE), exist_ok=True)
        os.makedirs(os.path.dirname(DOCUMENT_CHUNKS_FILE), exist_ok=True)

        try:
            logging.info(f"Sauvegarde de l'index Faiss dans {FAISS_INDEX_FILE}...")
            faiss.write_index(self.index, FAISS_INDEX_FILE)
            logging.info(f"Sauvegarde des chunks dans {DOCUMENT_CHUNKS_FILE}...")
            with open(DOCUMENT_CHUNKS_FILE, 'wb') as f:
                pickle.dump(self.document_chunks, f)
            logging.info("Index et chunks sauvegardés avec succès.")
        except Exception as e:
            logging.error(f"Erreur lors de la sauvegarde de l'index/chunks: {e}")

    @staticmethod
    def _is_past(date_str: Optional[str], today: datetime.date, default_if_unparseable: bool = True) -> bool:
        """
        Détermine si une date d'événement (format ISO, ex: "2026-02-14T18:00:00+00:00") est
        antérieure à `today`. Filtre déterministe (pas de raisonnement LLM sur les dates,
        qui s'est montré peu fiable en pratique).

        Args:
            default_if_unparseable: valeur renvoyée si la date est absente/non-parsable.
                True par défaut = on exclut par prudence (mieux vaut rater un résultat
                qu'afficher une date qu'on n'a pas pu vérifier).
        """
        if not date_str or date_str == "N/A":
            return default_if_unparseable
        try:
            event_date = datetime.datetime.fromisoformat(date_str).date()
        except (ValueError, TypeError):
            return default_if_unparseable
        return event_date < today

    def search(
        self,
        query_text: str,
        k: int = 5,
        min_score: float = None,
        city: Optional[str] = None,
        region: Optional[str] = None,
        month: Optional[int] = None,
        year: Optional[int] = None,
        include_past: bool = True,
    ) -> List[Dict[str, any]]:
        """
        Recherche les k chunks les plus pertinents pour une requête.

        Args:
            query_text: Texte de la requête
            k: Nombre de résultats à retourner
            min_score: Score minimum (entre 0 et 1) pour inclure un résultat
            city: si fourni, ne garde que les chunks dont la métadonnée 'ville' correspond
                  (comparaison insensible à la casse)
            region: si fourni, ne garde que les chunks dont la métadonnée 'region' correspond
                  (comparaison insensible à la casse). Note : seuls les chunks indexés/migrés
                  après l'ajout de ce champ ont une région renseignée.
            month: si fourni (1-12), ne garde que les chunks dont le mois de la date correspond
            year: si fourni avec `month`, restreint aussi à cette année précise (sinon "octobre"
                  attrape n'importe quelle année - vérifié en pratique, mauvaise surprise sans ça)
            include_past: si False, exclut (filtre déterministe, pas de LLM) les chunks dont
                  la date est antérieure à aujourd'hui, ou dont la date est absente/non-parsable.

        Returns:
            Liste des chunks pertinents avec leurs scores
        """
        if self.index is None or not self.document_chunks:
            logging.warning("Recherche impossible: l'index Faiss n'est pas chargé ou est vide.")
            return []
        if not MISTRAL_API_KEY:
             logging.error("Recherche impossible: MISTRAL_API_KEY manquante pour générer l'embedding de la requête.")
             return []

        logging.info(f"Recherche hybride (dense+BM25) des {k} chunks les plus pertinents pour: '{query_text}'")
        try:
            n_total = self.index.ntotal

            # 1. Recherche dense (Faiss) : classement de TOUS les chunks par similarité cosinus
            response = self.mistral_client.embeddings.create(
                model=EMBEDDING_MODEL,
                inputs=[query_text] # La requête doit être une liste
            )
            query_embedding = np.array([response.data[0].embedding]).astype('float32')
            faiss.normalize_L2(query_embedding) # Pour la similarité cosinus

            dense_scores, dense_indices = self.index.search(query_embedding, n_total)
            # rang (0 = meilleur) et score brut par index de chunk
            dense_rank = {int(idx): rank for rank, idx in enumerate(dense_indices[0])}
            dense_score_map = {int(idx): float(dense_scores[0][rank]) for rank, idx in enumerate(dense_indices[0])}

            # 2. Recherche sparse (BM25) : classement de TOUS les chunks par correspondance de mots-clés
            if self.bm25 is not None:
                bm25_scores = self.bm25.get_scores(self._tokenize(query_text))
                bm25_order = np.argsort(-bm25_scores) # du meilleur score au pire
                bm25_rank = {int(idx): rank for rank, idx in enumerate(bm25_order)}
            else:
                bm25_rank = {}

            # 3. Fusion des deux classements par Reciprocal Rank Fusion (RRF)
            # Chaque retrieveur "vote" en fonction du RANG (pas du score brut, non comparable entre les deux méthodes)
            fused_scores = {}
            for idx in range(n_total):
                r_dense = dense_rank.get(idx, n_total)
                r_bm25 = bm25_rank.get(idx, n_total)
                fused_scores[idx] = 1.0 / (RRF_K + r_dense + 1) + 1.0 / (RRF_K + r_bm25 + 1)

            # Classement final : score fusionné décroissant
            sorted_idx = sorted(fused_scores, key=lambda i: fused_scores[i], reverse=True)

            # 4. Formater les résultats (le score affiché reste la similarité cosinus dense, pour rester lisible)
            results = []
            min_score_percent = min_score * 100 if min_score is not None else 0
            for idx in sorted_idx:
                if not (0 <= idx < len(self.document_chunks)):
                    logging.warning(f"Index {idx} hors limites (taille des chunks: {len(self.document_chunks)}).")
                    continue

                raw_score = dense_score_map.get(idx, 0.0)
                similarity = raw_score * 100

                if min_score is not None and similarity < min_score_percent:
                    logging.debug(f"Document filtré (score {similarity:.2f}% < minimum {min_score_percent:.2f}%)")
                    continue

                chunk = self.document_chunks[idx]

                if city is not None:
                    chunk_ville = (chunk["metadata"].get("ville") or "").strip().lower()
                    if chunk_ville != city.strip().lower():
                        continue

                if region is not None:
                    chunk_region = (chunk["metadata"].get("region") or "").strip().lower()
                    if chunk_region != region.strip().lower():
                        continue

                if month is not None:
                    chunk_date = chunk["metadata"].get("date")
                    try:
                        parsed_date = datetime.datetime.fromisoformat(chunk_date)
                    except (ValueError, TypeError):
                        continue  # date absente/non-parsable : on exclut par prudence, comme _is_past
                    if parsed_date.month != month:
                        continue
                    if year is not None and parsed_date.year != year:
                        continue

                if not include_past and self._is_past(chunk["metadata"].get("date"), datetime.date.today()):
                    continue

                results.append({
                    "score": similarity, # Score de similarité dense en pourcentage (affichage)
                    "raw_score": raw_score, # Score brut dense pour débogage
                    "rrf_score": fused_scores[idx], # Score de fusion utilisé pour le classement
                    "text": chunk["text"],
                    "metadata": chunk["metadata"] # Contient source, category, chunk_id_in_doc, start_index etc.
                })

                if len(results) >= k:
                    break

            if min_score is not None:
                min_score_percent = min_score * 100
                logging.info(f"{len(results)} chunks pertinents trouvés (score minimum: {min_score_percent:.2f}%).")
            else:
                logging.info(f"{len(results)} chunks pertinents trouvés.")

            return results

        except MistralClientException as e:
            logging.error(f"Erreur API Mistral lors de la génération de l'embedding de la requête: {e}")
            logging.error(f"  Détails: Status Code={e.status_code}, Message={e.message}")
            return []
        except Exception as e:
            logging.error(f"Erreur inattendue lors de la recherche: {e}")
            return []