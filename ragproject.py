#!/usr/bin/env python
# coding: utf-8

# In[28]:


# ---------- (2) Imports ----------
import os
import base64
import json
import re
from typing import List, TypedDict, Annotated
import operator
from IPython.display import display, Image as IPImage
#from google.colab import files

# Groq client
from groq import Groq

# LangChain / embeddings / vectorstore
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# CrewAI & LangGraph
from crewai import LLM, Agent, Task, Crew, Process
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

# PDF parsing
import PyPDF2
from tkinter import Tk
from tkinter.filedialog import askopenfilenames
from IPython.display import display, Image as IPImage


# In[29]:


# ---------- Configuration ----------
import os

from dotenv import load_dotenv

load_dotenv() 

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

os.environ["GROQ_API_KEY"] = GROQ_API_KEY

from groq import Groq
from crewai import LLM

# récupérer la clé
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# client groq
client = Groq(api_key=GROQ_API_KEY)

# modèles
MODEL_VISION = "meta-llama/llama-4-scout-17b-16e-instruct"
MODEL_TEXT = "llama-3.3-70b-versatile"

# LLM CrewAI
llm = LLM(
    model="groq/llama-3.3-70b-versatile",
    api_key=GROQ_API_KEY
)

VALID_GROQ_MODELS = {
    "small": "llama-3.1-8b-instant",
    "large": "llama-3.3-70b-versatile",
    "mix": "mixtral-8x7b-32768",
    "gemma": "gemma2-9b-it"
}


# In[30]:


# ---------- (4) Helper Functions ----------
def to_base64(path: str) -> str:
    """Convertit une image en base64 data URI."""
    ext = path.split('.')[-1].lower()
    mime = "image/jpeg" if ext in ("jpg", "jpeg") else f"image/{ext}"
    with open(path, "rb") as f:
        return f"data:{mime};base64," + base64.b64encode(f.read()).decode()

def extract_text_from_file(path: str) -> str:
    """Extrait texte de .txt ou .pdf."""
    ext = path.split('.')[-1].lower()
    try:
        if ext == "txt":
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()
        elif ext == "pdf":
            text = []
            with open(path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    text.append(page.extract_text() or "")
            return "\n".join(text)
        else:
            return ""
    except Exception as e:
        print(f"Erreur extraction {path}: {e}")
        return ""

def analyse_visuelle(path, question, role):
    try:
        resp = client.chat.completions.create(
            model=MODEL_VISION,
            messages=[
                {"role": "system", "content": role},
                {"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": to_base64(path)}},
                    {"type": "text", "text": question}
                ]}
            ],
            max_tokens=1000
        )

        return resp.choices[0].message.content

    except Exception as e:
        print("Erreur vision:", e)
        return "Analyse visuelle indisponible."


# In[31]:


def synthese_finale(analyse_image: str, analyse_graph: str, analyse_texts: List[str], question: str) -> str:
    """Crée une synthèse finale combinant toutes les analyses."""
    combined_text = "\n\n".join(analyse_texts) if analyse_texts else "Aucune donnée textuelle."

    prompt = f"""
Tu es un expert en synthèse multimodale.
Question de l'utilisateur : {question}

--- Analyse des images ---
{analyse_image if analyse_image else "Aucune image analysée."}

--- Analyse des graphiques ---
{analyse_graph if analyse_graph else "Aucun graphique analysé."}

--- Preuves textuelles ---
{combined_text}

Produis une synthèse finale structurée en sections claires:
1. **Résumé Exécutif**: Synthèse en 5-4 phrases
2. **Analyse des Images**: Ce que montrent les images (si disponible)
3. **Analyse des Graphiques**: Données quantitatives et tendances (si disponible)
4. **Insights Documentaires**: Informations clés des textes (si disponible)
5. **Conclusion Intégrée**: Liens entre les différentes sources et recommandations
"""
    try:
        resp = client.chat.completions.create(
            model=MODEL_TEXT,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1500
        )
        return resp.choices[0].message.content
    except Exception as e:
        return f"Erreur synthèse finale: {e}"



# In[32]:


# ---------- (5) DocumentStore ----------
class DocumentStore:
    def __init__(self, documents: List[str]):
        if not documents:
            print(" Aucun document fourni")
            self.vectorstore = None
            return

        print(" Chargement des embeddings...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        docs = [Document(page_content=doc) for doc in documents]
        self.vectorstore = FAISS.from_documents(docs, self.embeddings)
        print(f"{len(documents)} documents indexés")

    def similarity_search(self, query: str, k: int = 3) -> List[str]:
        if self.vectorstore is None:
            return []
        results = self.vectorstore.similarity_search(query, k=k)
        return [doc.page_content for doc in results]

# ---------- (6) Agent State ----------
class AgentState(TypedDict):
    user_query: str
    routing_decision: List[str]
    text_evidence: Annotated[List[str], operator.add]
    graph_evidence: str
    image_evidence: str
    final_answer: str
    iteration_count: int

# ---------- (7) LLM Setup ----------
def create_llm(model_name: str, temperature: float = 0.0):
    try:
        llm = ChatGroq(
            groq_api_key=GROQ_API_KEY,
            model_name=model_name,
            temperature=temperature,
            max_retries=2,
            timeout=30
        )
        llm.invoke("test")
        return llm
    except Exception as e:
        print(f"Erreur avec {model_name}: {e}. Fallback.")
        return ChatGroq(
            groq_api_key=GROQ_API_KEY,
            model_name=VALID_GROQ_MODELS["small"],
            temperature=temperature
        )



# In[33]:


# ---------- (8) Workflow Nodes ----------

def router_node(state: AgentState) -> AgentState:
    """Router intelligent qui détermine quels agents activer."""
    query = state["user_query"].lower()
    llm = create_llm(VALID_GROQ_MODELS["small"], temperature=0)

    prompt = ChatPromptTemplate.from_template("""
Tu es un routeur intelligent. Analyse cette requête et retourne un JSON valide.
Format: ["retriever"] ou ["retriever", "graph"] ou ["retriever", "image"] ou ["retriever", "graph", "image"]

Règles:
- Inclure "retriever" si des documents textuels sont disponibles
- Ajouter "graph" si: graphique, courbe, tendance, évolution, statistique, données
- Ajouter "image" si: image, photo, schéma, figure, diagramme, visuel

Requête: {query}
Réponds UNIQUEMENT avec le JSON:
""")

    try:
        chain = prompt | llm
        response = chain.invoke({"query": query})
        content = response.content.strip()
        content = re.sub(r'```json\s*', '', content)
        content = re.sub(r'```\s*', '', content).strip()
        routing = json.loads(content)

        if not isinstance(routing, list):
            raise ValueError("Format invalide")
    except Exception as e:
        print(f" Erreur parsing JSON: {e}")
        routing = []
        if "document" in query or "texte" in query:
            routing.append("retriever")
        if any(w in query for w in ["graph", "graphique", "courbe", "tendance", "statistique"]):
            routing.append("graph")
        if any(w in query for w in ["image", "photo", "schéma", "figure", "diagramme"]):
            routing.append("image")
        if not routing:
            routing = ["retriever"]

    state["routing_decision"] = routing
    print(f"ROUTER: {routing}")
    return state


def retriever_node(state: AgentState, vector_store: DocumentStore) -> AgentState:
    """Recherche dans les documents textuels."""
    if "retriever" not in state["routing_decision"] or vector_store.vectorstore is None:
        print("RETRIEVER: Skippé")
        return state

    try:
        results = vector_store.similarity_search(state["user_query"], k=3)
        state["text_evidence"] = results
        print(f" RETRIEVER: {len(results)} documents trouvés")
    except Exception as e:
        print(f" Erreur RETRIEVER: {e}")
        state["text_evidence"] = []

    return state


def image_node(state: AgentState, image_paths: List[str]) -> AgentState:
    """Analyse les images fournies."""
    if not image_paths or "image" not in state["routing_decision"]:
        print("IMAGE NODE: Skippé")
        state["image_evidence"] = ""
        return state

    analyses = []
    print(" Analyse des images...")
    for p in image_paths:
        res = analyse_visuelle(
            p,
            state["user_query"],
        """Tu es un expert en vision par ordinateur.
        
        Analyse uniquement le contenu visuel :
        
        - objets présents
        - texte visible
        - environnement
        - contexte
        
        Ne fais aucune supposition basée sur d'autres documents.
        Réponds de manière structurée."""        )
        analyses.append(f"**{os.path.basename(p)}**:\n{res}")

    state["image_evidence"] = "\n\n".join(analyses)
    print(f" {len(image_paths)} image(s) analysée(s)")
    return state


def graph_node(state: AgentState, graph_paths: List[str]) -> AgentState:
    """Analyse les graphiques fournis."""
    if not graph_paths or "graph" not in state["routing_decision"]:
        print("GRAPH NODE: Skippé")
        state["graph_evidence"] = ""
        return state

    analyses = []
    print("📊 Analyse des graphiques...")
    for p in graph_paths:
        res = analyse_visuelle(
            p,
            state["user_query"],
            """Tu es un expert en théorie des graphes et visualisation de données.

                Analyse ce graphique :
                
                1. type de graphique
                2. sommets / noeuds
                3. arêtes / connexions
                4. propriétés (degré, centralité, connexité)
                
                Ne fais aucune supposition basée sur d'autres documents."""      )
        analyses.append(f"**{os.path.basename(p)}**:\n{res}")

    state["graph_evidence"] = "\n\n".join(analyses)
    print(f" {len(graph_paths)} graphique(s) analysé(s)")
    return state


def synthesis_node(state: AgentState, vector_store: DocumentStore) -> AgentState:
    """Synthèse finale avec CrewAI."""
    print("\n Synthèse finale avec CrewAI...")

    llm = LLM(
        model=VALID_GROQ_MODELS["large"],
        api_key=GROQ_API_KEY,
        base_url="https://api.groq.com/openai/v1"
    )

    # Préparation du contexte
    text_evidence = "\n".join(state["text_evidence"]) if state["text_evidence"] else "Aucun document textuel."
    image_evidence = state.get("image_evidence", "")
    graph_evidence = state.get("graph_evidence", "")
    
    if image_evidence.strip() == "":
        image_evidence = "Aucune analyse d'image disponible."
    
    if graph_evidence.strip() == "":
        graph_evidence = "Aucune analyse de graphique disponible."

    full_context = f"""
      Question utilisateur: {state['user_query']}

      === DOCUMENTS TEXTUELS ===
      {text_evidence}

      === ANALYSES IMAGES ===
      {image_evidence}

      === ANALYSES GRAPHIQUES ===
      {graph_evidence}
"""

    # Agents CrewAI
    analyst = Agent(
        role="Analyste Multi-Modal Senior",
        goal="Extraire insights clés de données textuelles, visuelles et graphiques",
        backstory="Expert en analyse multimodale avec 15 ans d'expérience.",
        llm=llm,
        verbose=False
    )

    synthesizer = Agent(
      role="Synthétiseur Expert",
      goal="Créer une synthèse structurée et actionnable",
      backstory="Spécialiste en communication analytique.",
      llm=llm,
      verbose=False
   )

    # Tasks
    task1 = Task(
        description=f"""
    Analyse séparément chaque source.

    1️⃣ DOCUMENTS TEXTUELS
    - extraire chiffres
    - faits importants
    
    2️⃣ IMAGES
    - décrire uniquement ce qui est visible
    - ne pas inventer de relation avec les documents
    
    3️⃣ GRAPHIQUES
    - identifier type
    - extraire structure ou données
    
    Ne mélange pas les sources si elles sont indépendantes.

    CONTEXTE:
    {full_context}
    """,
        expected_output="Liste structurée des insights multimodaux avec explications.",
        agent=analyst
    )

    task2 = Task(
        description="""
      À partir de l'analyse précédente, produis une synthèse finale structurée:

      1. Résumé exécutif (3 phrases max)
      2. Analyse détaillée par source (Texte / Image / Graphique)
      3. Corrélations et insights croisés
      4. Conclusion générale

      Format Markdown structuré.
      """,
        expected_output="Synthèse multimodale complète en markdown.",
        agent=synthesizer,
        context=[task1]
    )

    crew = Crew(
        agents=[analyst, synthesizer],
        tasks=[task1, task2],
        process=Process.sequential,
        verbose=False,
        memory=False
    )

    try:
        result = crew.kickoff()
        state["final_answer"] = str(result)
        print("Synthèse terminée")
    except Exception as e:
        state["final_answer"] = f" Erreur CrewAI: {e}"

    return state


# In[34]:


def create_workflow(vector_store: DocumentStore, image_paths: List[str], graph_paths: List[str]):
    workflow = StateGraph(AgentState)

    workflow.add_node("router", router_node)
    workflow.add_node("retriever", lambda s: retriever_node(s, vector_store))
    workflow.add_node("graph", lambda s: graph_node(s, graph_paths))
    workflow.add_node("image", lambda s: image_node(s, image_paths))
    workflow.add_node("synthesis", lambda s: synthesis_node(s, vector_store))

    workflow.set_entry_point("router")
    workflow.add_edge("router", "retriever")
    workflow.add_edge("retriever", "graph")
    workflow.add_edge("graph", "image")
    workflow.add_edge("image", "synthesis")
    workflow.add_edge("synthesis", END)

    return workflow.compile()


# In[35]:
from PIL import Image
import os

def run_multimodal_analysis(documents, image_paths, graph_paths, user_question):
    """
    Analyse multimodale avec gestion automatique de la question.
    """
    if not user_question or not user_question.strip():
        user_question = "Analyse complète des données : documents, images et graphiques."

    vector_store = DocumentStore(documents) if documents else DocumentStore([])

    initial_state = {
        "user_query": user_question, 
        "routing_decision": [],
        "text_evidence": [],
        "graph_evidence": "",
        "image_evidence": "",
        "final_answer": "",
        "iteration_count": 0
    }
    

    app = create_workflow(vector_store, image_paths, graph_paths)

    try:
        final_state = app.invoke(initial_state)
    except Exception as e:
        final_state = initial_state
        final_state["final_answer"] = f"Erreur workflow : {e}"

    # fallback si vide
    if not final_state["final_answer"].strip():
        result = ""
        if documents:
            result += f"Documents uploadés : {len(documents)}\n"
        if image_paths:
            result += f"Images uploadées : {', '.join([os.path.basename(p) for p in image_paths])}\n"
        if graph_paths:
            result += f"Graphiques uploadés : {', '.join([os.path.basename(p) for p in graph_paths])}\n"
        final_state["final_answer"] = result or "Aucune donnée détectée."

    return final_state["final_answer"]

# In[37]:


# In[ ]:




