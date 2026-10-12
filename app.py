import streamlit as st
import tempfile
import os
from PIL import Image
from ragproject import extract_text_from_file, run_multimodal_analysis
import base64

# ---------- PAGE CONFIG ----------
st.set_page_config(
    page_title="Multimodal AI Analyzer - Projet SMA",
    page_icon="🧠",
    layout="wide"
)

# ---------- STYLE CSS PRO ----------
st.markdown("""
<style>
    .stApp { background-color: #f4f7f6; }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #003366 !important;
        color: white !important;
    }
    section[data-testid="stSidebar"] * { color: white !important; }

    /* Custom Cards */
    .css-card {
        background: white;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
        margin-bottom: 20px;
        border: 1px solid #e2e8f0;
    }

    /* Result Box Style */
    .result-box {
        background-color: #ffffff;
        padding: 25px;
        border-radius: 15px;
        border-left: 6px solid #4A90E2;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        color: #1e293b;
        line-height: 1.6;
    }

    /* Titles */
    .section-title {
        color: #003366;
        font-weight: 700;
        margin-bottom: 15px;
        border-bottom: 2px solid #4A90E2;
        display: inline-block;
    }

    /* Analyze Button */
    div.stButton > button {
        background: linear-gradient(90deg, #4A90E2 0%, #357ABD 100%);
        color: white; border-radius: 10px; padding: 15px; font-weight: bold; 
        border: none; width: 100%; font-size: 18px; transition: 0.3s;
    }
    div.stButton > button:hover {
        transform: scale(1.01);
        box-shadow: 0 5px 15px rgba(74,144,226,0.3);
    }
    /* Styling the Upload Box (Drag and Drop) */
    [data-testid="stFileUploadDropzone"] {
        border: 2px dashed #4A90E2 !important;
        background-color: #f0f7ff !important;
        border-radius: 12px !important;
        transition: 0.3s;
    }
    [data-testid="stFileUploadDropzone"]:hover {
        border-color: #003366 !important;
        background-color: #e6f2ff !important;
    }

    /* Custom labels for uploaders */
    .upload-info {
        font-size: 12px;
        color: #64748b;
        margin-top: -10px;
        margin-bottom: 15px;
    }
    
    .stTextArea textarea {
        background-color: #ffffff !important;
        border: 1px solid #cbd5e1 !important;
        border-radius: 10px !important;
    }
</style>
""", unsafe_allow_html=True)

# ---------- SIDEBAR (TEAM) ----------
def get_image_base64(path):
    if os.path.exists(path):
        with open(path, "rb") as f:
            data = f.read()
        return base64.b64encode(data).decode()
    return None

with st.sidebar:
    if os.path.exists("docs/Projet.jpeg"):
        st.image("docs/Projet.jpeg")
    else:
        st.markdown("<h2 style='text-align:center; color:white;'>🚀 PROJET SMA</h2>", unsafe_allow_html=True)
    
    st.markdown("<hr style='border: 0.5px solid #4A90E2;'>", unsafe_allow_html=True)
    st.markdown("<h3 style='color:white; text-align:center;'>👥 Équipe Projet</h3>", unsafe_allow_html=True)

    students = [
        {"name": "Kawtar Souhail", "role": "Responsable RAG & Texte", "img": "docs/Kawtar.jpeg"},
    ]

    for s in students:
        img_base64 = get_image_base64(s['img'])
        if img_base64:
            st.markdown(f"""
            <div style="display: flex; align-items: center; background: rgba(255,255,255,0.1); padding: 10px; border-radius: 15px; margin-bottom: 10px; border: 1px solid rgba(255,255,255,0.2);">
                <img src="data:image/jpeg;base64,{img_base64}" style="width: 50px; height: 50px; border-radius: 50%; object-fit: cover; border: 2px solid #4A90E2; margin-right: 12px;">
                <div>
                    <b style="color: white; font-size: 14px; display: block;">{s['name']}</b>
                    <span style="color: #cbd5e1; font-size: 11px;">{s['role']}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.info("💡 **Tech Stack:** Llama 3.3, CrewAI, LangGraph, FAISS")

# ---------- HEADER (UNIVERSITÉ & PROJET) ----------
header_html = """
<div style="background-color: #003366; padding: 20px; border-radius: 15px; display: flex; align-items: center; justify-content: space-between; margin-bottom: 30px; border-bottom: 4px solid #4A90E2;">
    <div style="flex: 1;"><h2 style="color: white; margin:0;">🏫 UIT</h2></div>
    <div style="flex: 2; text-align: center;">
        <h1 style="color: white; margin: 0; font-family: 'Arial'; font-weight: 800;">MULTIMODAL AI ANALYZER</h1>
        <p style="color: #cbd5e1; margin: 0; font-size: 14px;">Master Systèmes Intelligents & SMA</p>
    </div>
    <div style="flex: 1; text-align: right;">
        <div style="border: 2px solid #4A90E2; padding: 8px; border-radius: 10px; display: inline-block; background: rgba(255,255,255,0.05);">
            <p style="color: #4A90E2; margin: 0; font-size: 11px; font-weight: bold;">Encadré par :</p>
            <p style="color: white; margin: 0; font-size: 15px; font-weight: bold;">Pr. Salma Azzouzi</p>
        </div>
    </div>
</div>
"""
st.markdown(header_html, unsafe_allow_html=True)

# ---------- MAIN CONTENT (FILE UPLOAD) ----------
st.markdown('<p class="section-title">📂 Chargement des Données</p>', unsafe_allow_html=True)
st.markdown('<div class="css-card">', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("<b>📄 Documents (PDF/TXT)</b>", unsafe_allow_html=True)
    docs = st.file_uploader(
        "Drag and drop files here", 
        type=["pdf","txt"], 
        accept_multiple_files=True, 
        key="docs_up",
        label_visibility="collapsed" # Khayidna l-label l-qdim bach n-diro dyalna
    )
    st.markdown('<p class="upload-info">Limit 200MB per file • PDF, TXT</p>', unsafe_allow_html=True)

with col2:
    st.markdown("<b>🖼 Images (Vision)</b>", unsafe_allow_html=True)
    imgs = st.file_uploader(
        "Drag and drop files here", 
        type=["png","jpg","jpeg"], 
        accept_multiple_files=True, 
        key="imgs_up",
        label_visibility="collapsed"
    )
    st.markdown('<p class="upload-info">Limit 200MB per file • PNG, JPG, JPEG</p>', unsafe_allow_html=True)

with col3:
    st.markdown("<b>📊 Graphiques (Data)</b>", unsafe_allow_html=True)
    grphs = st.file_uploader(
        "Drag and drop files here", 
        type=["png","jpg","jpeg"], 
        accept_multiple_files=True, 
        key="grphs_up",
        label_visibility="collapsed"
    )
    st.markdown('<p class="upload-info">Limit 200MB per file • PNG, JPG, JPEG</p>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# ---------- QUESTION AREA ----------
st.markdown('<p class="section-title">❓ Question & Analyse Spécifique</p>', unsafe_allow_html=True)
user_input = st.text_area(
    "", 
    placeholder="Ex: Analyse la corrélation entre les graphiques et le texte du PDF...", 
    height=100
)

# ---------- BUTTON & LOGIC ----------
if st.button("🚀 LANCER L'ANALYSE MULTIMODALE"):
    if not (docs or imgs or grphs):
        st.error("⚠️ Veuillez uploader au moins un fichier pour commencer.")
    else:
        with st.spinner("🧠 Intelligence Artificielle en cours de réflexion..."):
            documents, image_paths, graph_paths = [], [], []
            temp_dir = tempfile.mkdtemp()
            
            # (Hna l-logic dyal l-processing dialk t-ibqa hwa hwa...)
            if docs:
                for f in docs:
                    p = os.path.join(temp_dir, f.name)
                    with open(p, "wb") as file: file.write(f.getbuffer())
                    txt = extract_text_from_file(p)
                    if txt.strip(): documents.append(txt)
            
            if imgs:
                for f in imgs:
                    p = os.path.join(temp_dir, f.name)
                    with open(p, "wb") as file: file.write(f.read())
                    image_paths.append(p)

            if grphs:
                for f in grphs:
                    p = os.path.join(temp_dir, f.name)
                    with open(p, "wb") as file: file.write(f.read())
                    graph_paths.append(p)

            user_q = user_input if user_input.strip() else "Analyse complète."
            final_answer = run_multimodal_analysis(documents, image_paths, graph_paths, user_q)

            # --- DISPLAY RESULTS ---
            st.markdown('<p class="section-title">📋 Rapport d\'Analyse Final</p>', unsafe_allow_html=True)
            
            tab_res, tab_media = st.tabs(["📑 Synthèse de l'IA", "📸 Fichiers Source"])
            
            with tab_res:
                st.markdown(f'<div class="result-box">{final_answer}</div>', unsafe_allow_html=True)
            
            with tab_media:
                c_i, c_g = st.columns(2)
                with c_i:
                    if image_paths:
                        st.markdown("#### 🖼 Images")
                        for p in image_paths: st.image(p, use_container_width=True)
                with c_g:
                    if graph_paths: 
                        st.markdown("#### 📊 Graphiques")
                        for p in graph_paths: st.image(p, use_container_width=True)