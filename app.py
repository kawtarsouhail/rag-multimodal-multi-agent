import streamlit as st
import tempfile
import os
from PIL import Image

from ragproject import extract_text_from_file, run_multimodal_analysis

# ---------- PAGE CONFIG ----------
st.set_page_config(
    page_title="Multimodal AI Analyzer",
    page_icon="🧠",
    layout="wide"
)

# ---------- STYLE CSS ----------
st.markdown("""
<style>
.big-title {
font-size:40px;
font-weight:bold;
color:#4A90E2;
}

.card {
background-color:#f8f9fa;
padding:20px;
border-radius:10px;
box-shadow:0 2px 6px rgba(0,0,0,0.1);
}
</style>
""", unsafe_allow_html=True)

# ---------- HEADER ----------
st.markdown('<p class="big-title">🧠 Multimodal AI Analyzer</p>', unsafe_allow_html=True)
st.write(
    """
    Analyse **documents, images et graphiques** avec un système multimodal  
    basé sur **RAG + Vision + CrewAI + LangGraph**.
    """
)

# ---------- FILE UPLOAD ----------
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### 📄 Documents")
    docs = st.file_uploader(
        "Upload PDF / TXT",
        type=["pdf","txt"],
        accept_multiple_files=True
    )

with col2:
    st.markdown("### 🖼 Images")
    images = st.file_uploader(
        "Upload Images",
        type=["png","jpg","jpeg"],
        accept_multiple_files=True
    )

with col3:
    st.markdown("### 📊 Graphiques")
    graphs = st.file_uploader(
        "Upload Graphs",
        type=["png","jpg","jpeg"],
        accept_multiple_files=True
    )

# ---------- QUESTION ----------
# ---------- QUESTION ----------
st.markdown("### ❓ Question")
user_input = st.text_area(
    "Que voulez-vous analyser ?",
    height=120,
    placeholder="Laissez vide pour une analyse complète automatique..."
)

# Si l'utilisateur ne tape rien, on définit une question par défaut
if not user_input.strip():
    user_question = "Effectue une analyse complète et croisée de tous les documents, images et graphiques fournis. Donne une conclusion générale."
else:
    user_question = user_input

# ---------- ANALYZE BUTTON ----------
if st.button("🚀 Lancer Analyse", use_container_width=True):

    with st.spinner("Analyse multimodale en cours..."):

        # ---------- SAVE FILES TEMP ----------
        documents, image_paths, graph_paths = [], [], []
        temp_dir = tempfile.mkdtemp()

        # documents
        if docs:
            for file in docs:
                path = os.path.join(temp_dir, file.name)
                with open(path, "wb") as f:
                    f.write(file.getbuffer())
                text = extract_text_from_file(path)
                if text.strip():
                    documents.append(text)

        # images
        if images:
            for file in images:
                path = os.path.join(temp_dir, file.name)
                with open(path, "wb") as f:
                    f.write(file.read())
                image_paths.append(path)

        # graphs
        if graphs:
            for file in graphs:
                path = os.path.join(temp_dir, file.name)
                with open(path, "wb") as f:
                    f.write(file.read())
                graph_paths.append(path)

        # ---------- RUN MULTIMODAL ANALYSIS ----------
        final_answer = run_multimodal_analysis(
            documents,
            image_paths,
            graph_paths,
            user_question
        )

    # ---------- DISPLAY IMAGES ----------
    if image_paths:
        st.subheader("🖼 Images analysées")
        for p in image_paths:
            st.image(p, caption=os.path.basename(p), width=250)

    if graph_paths:
        st.subheader("📊 Graphiques analysés")
        for p in graph_paths:
            st.image(p, caption=os.path.basename(p), width=250)

    # ---------- DISPLAY FINAL SYNTHESIS ----------
    st.subheader("📋 Synthèse finale")
    st.markdown(final_answer)

# ---------- FILE PREVIEW ----------
st.markdown("## 🔍 Aperçu des fichiers")

col1, col2 = st.columns(2)

with col1:
    if images:
        for img in images:
            image = Image.open(img)
            st.image(image, caption=img.name, width=250)

with col2:
    if graphs:
        for g in graphs:
            image = Image.open(g)
            st.image(image, caption=g.name, width=250)