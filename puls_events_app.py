# puls_events_app.py
"""Interface Streamlit du chatbot Puls-Events : recommandations d'événements culturels."""

import streamlit as st
import datetime
from streamlit_feedback import streamlit_feedback

from utils.config import APP_TITLE, APP_NAME
from utils.chat_service import answer_query
from utils.database import update_feedback
from utils.formatting import format_date_fr

st.set_page_config(page_title=APP_TITLE, page_icon="🎭", layout="wide")

# --- Initialisation de l'état de session ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_interaction_id" not in st.session_state:
    st.session_state.last_interaction_id = None

# --- Barre latérale : réglages ---
with st.sidebar:
    st.title(f"🎭 {APP_NAME}")
    st.caption("Assistant de recommandations d'événements culturels")

    if st.button("🔄 Nouvelle conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.last_interaction_id = None
        st.rerun()

    st.divider()
    st.subheader("⚙️ Paramètres")

    # Modèle unique fixé en dur : mistral-large-latest renvoie une erreur 403 (non inclus
    # dans l'abonnement Mistral utilisé), et entre les deux modèles restants testés,
    # mistral-medium-latest donne des réponses plus complètes que mistral-small-latest
    # (ajoute systématiquement lieu + description, pas seulement small).
    selected_model = "mistral-medium-latest"

    # Pas de curseur "nombre d'événements" ici : ce n'est pas un réglage de qualité (le classement
    # Faiss+BM25+RRF porte toujours sur l'index entier, quel que soit ce plafond) - seul le budget
    # de contexte (MAX_CONTEXT_CHARS, utils/config.py) décide réellement combien de résultats
    # arrivent jusqu'au LLM, de façon adaptative. Un plafond bas ici ne ferait qu'appauvrir les
    # questions multi-thèmes (risque de faire disparaître un thème entier) sans rien améliorer.
    # SEARCH_K (utils/config.py) reste un garde-fou interne, pas exposé côté client.

    min_score_percent = st.slider(
        "Score minimum (filtrer les résultats faibles)", min_value=0, max_value=100, value=75, step=5, format="%d%%",
        help="Score de similarité sémantique, pas un niveau de confiance classique : même un très "
             "bon résultat dépasse rarement 80% avec notre modèle d'embeddings. Au-delà de 85%, "
             "il est fréquent qu'aucun événement ne remonte, même pertinent.",
    )
    min_score = min_score_percent / 100.0

    st.divider()
    if st.session_state.messages:
        st.info(f"{len(st.session_state.messages) // 2} échanges dans cette conversation")

# --- Titre principal ---
st.title(f"🎭 {APP_TITLE}")
st.caption("Posez vos questions sur les événements culturels à venir")

# --- Affichage de l'historique ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and message.get("sources"):
            with st.expander("Sources utilisées"):
                for i, source in enumerate(message["sources"]):
                    meta = source.get("metadata", {})
                    st.markdown(f"**{meta.get('filename', 'N/A')}** — {meta.get('ville', 'N/A')} ({format_date_fr(meta.get('date'))})")
                    if meta.get("url") and meta["url"] != "N/A":
                        st.markdown(f"[Voir l'événement]({meta['url']})")
                    st.markdown(f"*Score de similarité:* {source.get('score', 0.0):.1f}%")
                    st.text_area(
                        f"Extrait {i+1}", value=source.get("text", "")[:500] + "...",
                        height=100, disabled=True, key=f"src_{message['timestamp']}_{i}",
                    )

# --- Zone de saisie ---
if prompt := st.chat_input("Posez votre question ici..."):
    st.session_state.messages.append({"role": "user", "content": prompt, "timestamp": datetime.datetime.now().isoformat()})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        placeholder.markdown("🔎 Recherche d'événements et génération de la réponse...")

        try:
            # Historique = tous les messages sauf celui qu'on vient d'ajouter (le prompt actuel),
            # réduit à role/content (on retire sources/timestamp/interaction_id, inutiles ici)
            conversation_history = [
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages[:-1]
            ]
            result = answer_query(
                prompt,
                conversation_history=conversation_history,
                min_score=min_score,
                model=selected_model,
            )
            placeholder.markdown(result["response"])

            if result["sources"]:
                with st.expander("Sources utilisées"):
                    for i, source in enumerate(result["sources"]):
                        meta = source.get("metadata", {})
                        st.markdown(f"**{meta.get('filename', 'N/A')}** — {meta.get('ville', 'N/A')} ({format_date_fr(meta.get('date'))})")
                        if meta.get("url") and meta["url"] != "N/A":
                            st.markdown(f"[Voir l'événement]({meta['url']})")
                        st.markdown(f"*Score de similarité:* {source.get('score', 0.0):.1f}%")
                        st.text_area(
                            f"Extrait {i+1}", value=source.get("text", "")[:500] + "...",
                            height=100, disabled=True, key=f"src_new_{i}",
                        )
            else:
                st.info("Mode direct : réponse générée sans base d'événements." if result["mode"] == "DIRECT"
                         else "Aucun événement pertinent trouvé pour cette question.")

            st.session_state.last_interaction_id = result["interaction_id"]
            st.session_state.messages.append({
                "role": "assistant",
                "content": result["response"],
                "sources": result["sources"],
                "timestamp": datetime.datetime.now().isoformat(),
                "interaction_id": result["interaction_id"],
            })

        except Exception as e:
            placeholder.error(f"Une erreur s'est produite: {e}")
            st.session_state.messages.append({
                "role": "assistant", "content": f"Erreur: {e}", "sources": [],
                "timestamp": datetime.datetime.now().isoformat(), "interaction_id": None,
            })
            st.session_state.last_interaction_id = None

# --- Feedback sur la dernière réponse ---
last_assistant_message = next((m for m in reversed(st.session_state.messages) if m["role"] == "assistant"), None)
current_interaction_id = last_assistant_message.get("interaction_id") if last_assistant_message else None

if current_interaction_id:
    feedback = streamlit_feedback(
        feedback_type="thumbs",
        optional_text_label="[Optionnel] Commentaires :",
        key=f"feedback_{current_interaction_id}",
        align="flex-start",
    )
    if feedback:
        score = feedback.get("score")
        feedback_score = "positive" if score in ("👍", "thumbs_up") else "negative" if score in ("👎", "thumbs_down") else None
        feedback_text = "positif" if feedback_score == "positive" else "négatif" if feedback_score == "negative" else "N/A"
        feedback_value = 1 if feedback_score == "positive" else 0 if feedback_score == "negative" else None
        comment = feedback.get("text")

        success = update_feedback(current_interaction_id, feedback_text, comment, feedback_value)
        if success:
            st.toast("Merci pour votre retour !", icon="✅")
        else:
            st.toast("Erreur lors de l'enregistrement du retour.", icon="❌")
else:
    st.write("Posez une question pour pouvoir donner votre avis sur la réponse.")
