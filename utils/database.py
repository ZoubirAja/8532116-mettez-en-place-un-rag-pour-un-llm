# utils/database.py
import os
import datetime
import logging
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, Float, JSON
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.exc import SQLAlchemyError

from .config import DATABASE_URL, DATABASE_DIR

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Crée le dossier de la base de données s'il n'existe pas
os.makedirs(DATABASE_DIR, exist_ok=True)

# Crée l'engine SQLAlchemy pour la base de données SQLite
# `check_same_thread=False` est nécessaire pour SQLite avec Streamlit/multithreading
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False}, echo=False) # echo=True pour voir les requêtes SQL

# Crée une base de déclaration pour les modèles ORM
Base = declarative_base()

# Définit le modèle ORM pour la table des interactions
class Interaction(Base):
    __tablename__ = 'interactions'

    id = Column(Integer, primary_key=True)
    # `default=lambda: ...` (pas `default=datetime.datetime.now(...)`) : sans le lambda, la
    # valeur n'était calculée qu'UNE fois au chargement du module, pas à chaque insertion - toutes
    # les interactions d'un même processus (ex: une session Streamlit sans redémarrage) se
    # retrouvaient avec le même horodatage figé (bug réel trouvé : plusieurs lignes de la table
    # partageant l'horodatage à la microseconde près, présent depuis le tout premier commit).
    timestamp = Column(DateTime, default=lambda: datetime.datetime.now(datetime.timezone.utc))
    query = Column(Text, nullable=False)
    response = Column(Text)
    sources = Column(JSON) # Stocke la liste des dictionnaires de sources en JSON
    query_metadata = Column(JSON) # Stocke les métadonnées (mode, confiance, etc.)
    feedback = Column(String(20)) # ex: "👍", "👎"
    feedback_value = Column(Integer) # 1 pour positif, 0 pour négatif, NULL pour aucun
    feedback_comment = Column(Text) # Optionnel: commentaire de feedback

# Crée la table dans la base de données si elle n'existe pas déjà
try:
    Base.metadata.create_all(engine)
    logging.info("Table 'interactions' vérifiée/créée dans la base de données.")
except SQLAlchemyError as e:
    logging.error(f"Erreur lors de la création/vérification de la table: {e}")

# Crée une factory de session pour interagir avec la base de données
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def log_interaction(query: str, response: str, sources: list, metadata: dict = None, feedback: str = None, feedback_comment: str = None):
    """Enregistre une interaction dans la base de données.

    Args:
        query: Question de l'utilisateur
        response: Réponse générée
        sources: Liste des sources utilisées
        metadata: Métadonnées (mode, confiance, etc.)
        feedback: Feedback utilisateur
        feedback_comment: Commentaire de feedback

    Returns:
        ID de l'interaction enregistrée
    """
    db_session = SessionLocal()
    try:
        interaction = Interaction(
            query=query,
            response=response,
            sources=sources, # SQLAlchemy gère la sérialisation JSON
            query_metadata=metadata, # Métadonnées (mode, confiance, etc.)
            feedback=feedback,
            feedback_comment=feedback_comment
        )
        db_session.add(interaction)
        db_session.commit()

        # Journaliser avec des informations sur le mode utilisé
        mode_info = ""
        if metadata and "mode" in metadata:
            mode_info = f", Mode: {metadata['mode']}"

        logging.info(f"Interaction enregistrée (Query: '{query[:50]}...'{mode_info}, Feedback: {feedback})")
        return interaction.id # Retourne l'ID de l'interaction enregistrée
    except SQLAlchemyError as e:
        logging.error(f"Erreur lors de l'enregistrement de l'interaction: {e}")
        db_session.rollback() # Annule les changements en cas d'erreur
        return None
    finally:
        db_session.close() # Ferme toujours la session

def get_all_interactions(limit: int = 100):
    """Récupère les dernières interactions de la base de données."""
    db_session = SessionLocal()
    try:
        interactions = db_session.query(Interaction).order_by(Interaction.timestamp.desc()).limit(limit).all()
        logging.info(f"{len(interactions)} interactions récupérées.")
        # Convertit les objets Interaction en dictionnaires pour une manipulation plus facile (ex: Pandas)
        return [
            {
                "id": inter.id,
                "timestamp": inter.timestamp,
                "query": inter.query,
                "response": inter.response,
                "sources": inter.sources, # Déjà une liste de dicts (ou None)
                "metadata": inter.query_metadata, # Métadonnées (mode, confiance, etc.)
                "feedback": inter.feedback,
                "feedback_comment": inter.feedback_comment,
            }
            for inter in interactions
        ]
    except SQLAlchemyError as e:
        logging.error(f"Erreur lors de la récupération des interactions: {e}")
        return []
    finally:
        db_session.close()

def update_feedback(interaction_id: int, feedback: str, feedback_comment: str = None, feedback_value: int = None):
    """Met à jour le feedback pour une interaction spécifique.

    Args:
        interaction_id: ID de l'interaction à mettre à jour
        feedback: Texte du feedback (emoji)
        feedback_comment: Commentaire optionnel
        feedback_value: Valeur numérique (1 pour positif, 0 pour négatif)

    Returns:
        True si la mise à jour a réussi, False sinon
    """
    db_session = SessionLocal()
    try:
        interaction = db_session.query(Interaction).filter(Interaction.id == interaction_id).first()
        if interaction:
            # Mise à jour des valeurs
            interaction.feedback = feedback
            interaction.feedback_value = feedback_value
            interaction.feedback_comment = feedback_comment

            # Enregistrer les modifications
            db_session.commit()
            logging.info(f"Feedback mis à jour pour l'interaction ID {interaction_id}")
            return True
        else:
            logging.warning(f"Interaction ID {interaction_id} non trouvée pour la mise à jour du feedback.")
            return False
    except SQLAlchemyError as e:
        logging.error(f"Erreur lors de la mise à jour du feedback pour l'interaction {interaction_id}: {e}")
        db_session.rollback()
        return False
    finally:
        db_session.close()