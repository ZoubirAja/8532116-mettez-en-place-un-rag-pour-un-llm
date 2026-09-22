# utils/chat_message.py
"""Petit wrapper pour représenter un message de conversation (role + contenu)."""


class ChatMessage:
    """
    Représente un message envoyé à/reçu de l'API Mistral (system, user ou assistant).
    L'API Mistral attend des dictionnaires simples {"role": ..., "content": ...} —
    cette classe sert juste à manipuler des objets plus lisibles dans le code, à
    convertir en dict via .format() juste avant l'appel API.
    """

    def __init__(self, role: str, content: str):
        self.role = role
        self.content = content

    def format(self) -> dict:
        """Convertit le message au format dict attendu par le client Mistral."""
        return {"role": self.role, "content": self.content}
