"""
Multimodal multi-agent RAG pipeline.

Flow (LangGraph): router -> retriever -> graph -> image -> synthesis (CrewAI)
Vision: Groq Llama 4 Scout | Router: Llama 3.1 8B | Synthesis: gpt-oss-120b
"""

import base64
import json
import operator
import os
import re
from functools import lru_cache
from typing import Annotated, List, TypedDict

import PyPDF2
from crewai import LLM, Agent, Crew, Process, Task
from dotenv import load_dotenv
from groq import Groq
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import END, StateGraph

# ---------- Configuration ----------
load_dotenv()  # reads GROQ_API_KEY from the .env file

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY is missing: add it to your .env file.")

client = Groq(api_key=GROQ_API_KEY)

MODEL_VISION = "meta-llama/llama-4-scout-17b-16e-instruct"

# Defined BEFORE any use (this was the NameError)
VALID_GROQ_MODELS = {
    "small": "llama-3.1-8b-instant",
    "large": "openai/gpt-oss-120b",
}


# ---------- Helper functions ----------
def to_base64(path: str) -> str:
    """Convert an image file to a base64 data URI."""
    ext = path.split(".")[-1].lower()
    mime = "image/jpeg" if ext in ("jpg", "jpeg") else f"image/{ext}"
    with open(path, "rb") as f:
        return f"data:{mime};base64," + base64.b64encode(f.read()).decode()


def extract_text_from_file(path: str) -> str:
    """Extract text from a .txt or .pdf file."""
    ext = path.split(".")[-1].lower()
    try:
        if ext == "txt":
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()
        if ext == "pdf":
            pages = []
            with open(path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    pages.append(page.extract_text() or "")
            return "\n".join(pages)
        return ""
    except Exception as e:
        print(f"Extraction error {path}: {e}")
        return ""


def analyse_visuelle(path: str, question: str, role: str) -> str:
    """Send one image to the Groq vision model."""
    try:
        resp = client.chat.completions.create(
            model=MODEL_VISION,
            messages=[
                {"role": "system", "content": role},
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": to_base64(path)}},
                        {"type": "text", "text": question},
                    ],
                },
            ],
            max_tokens=1000,
        )
        return resp.choices[0].message.content
    except Exception as e:
        print("Vision error:", e)
        return "Analyse visuelle indisponible."


# ---------- Document store (FAISS) ----------
@lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:
    """Load the embedding model once, not on every request."""
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
    )


class DocumentStore:
    def __init__(self, documents: List[str]):
        self.vectorstore = None
        if not documents:
            print("No documents provided")
            return

        print("Indexing documents...")
        # Split long documents so retrieval returns relevant passages,
        # not whole PDFs.
        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
        chunks = [
            Document(page_content=chunk)
            for doc in documents
            for chunk in splitter.split_text(doc)
        ]
        if not chunks:
            return
        self.vectorstore = FAISS.from_documents(chunks, get_embeddings())
        print(f"{len(chunks)} chunks indexed from {len(documents)} document(s)")

    def similarity_search(self, query: str, k: int = 3) -> List[str]:
        if self.vectorstore is None:
            return []
        return [d.page_content for d in self.vectorstore.similarity_search(query, k=k)]


# ---------- Agent state ----------
class AgentState(TypedDict):
    user_query: str
    routing_decision: List[str]
    text_evidence: Annotated[List[str], operator.add]
    graph_evidence: str
    image_evidence: str
    final_answer: str
    iteration_count: int


# ---------- LLM setup ----------
def create_llm(model_name: str, temperature: float = 0.0) -> ChatGroq:
    return ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name=model_name,
        temperature=temperature,
        max_retries=2,
        timeout=30,
    )


# ---------- Workflow nodes ----------
# Each node returns ONLY the keys it changes. Returning the full state would
# make the `operator.add` reducer duplicate text_evidence at every step.

def router_node(state: AgentState, image_paths: List[str], graph_paths: List[str]) -> dict:
    """Decide which agents to activate."""
    query = state["user_query"].lower()
    llm = create_llm(VALID_GROQ_MODELS["small"], temperature=0)

    prompt = ChatPromptTemplate.from_template(
        """
Tu es un routeur intelligent. Analyse cette requête et retourne un JSON valide.
Format: ["retriever"] ou ["retriever", "graph"] ou ["retriever", "image"] ou ["retriever", "graph", "image"]

Règles:
- Inclure "retriever" si des documents textuels sont disponibles
- Ajouter "graph" si: graphique, courbe, tendance, évolution, statistique, données
- Ajouter "image" si: image, photo, schéma, figure, diagramme, visuel

Requête: {query}
Réponds UNIQUEMENT avec le JSON:
"""
    )

    try:
        response = (prompt | llm).invoke({"query": query})
        content = re.sub(r"```(?:json)?", "", response.content.strip()).strip()
        routing = json.loads(content)
        if not isinstance(routing, list):
            raise ValueError("Invalid format")
        routing = [r for r in routing if r in ("retriever", "graph", "image")]
    except Exception as e:
        print(f"Router parsing error: {e}. Using keyword fallback.")
        routing = []
        if any(w in query for w in ["document", "texte", "pdf"]):
            routing.append("retriever")
        if any(w in query for w in ["graph", "graphique", "courbe", "tendance", "statistique"]):
            routing.append("graph")
        if any(w in query for w in ["image", "photo", "schéma", "figure", "diagramme"]):
            routing.append("image")

    if not routing:
        routing = ["retriever"]

    # Never ignore files the user actually uploaded
    if image_paths and "image" not in routing:
        routing.append("image")
    if graph_paths and "graph" not in routing:
        routing.append("graph")

    print(f"ROUTER: {routing}")
    return {"routing_decision": routing}


def retriever_node(state: AgentState, vector_store: DocumentStore) -> dict:
    """Search the text documents."""
    if "retriever" not in state["routing_decision"] or vector_store.vectorstore is None:
        print("RETRIEVER: skipped")
        return {"text_evidence": []}
    try:
        results = vector_store.similarity_search(state["user_query"], k=3)
        print(f"RETRIEVER: {len(results)} passages found")
        return {"text_evidence": results}
    except Exception as e:
        print(f"RETRIEVER error: {e}")
        return {"text_evidence": []}


def image_node(state: AgentState, image_paths: List[str]) -> dict:
    """Analyse the uploaded images."""
    if not image_paths or "image" not in state["routing_decision"]:
        print("IMAGE NODE: skipped")
        return {"image_evidence": ""}

    role = """Tu es un expert en vision par ordinateur.

Analyse uniquement le contenu visuel :
- objets présents
- texte visible
- environnement
- contexte

Ne fais aucune supposition basée sur d'autres documents.
Réponds de manière structurée."""

    print("Analysing images...")
    analyses = []
    for p in image_paths:
        res = analyse_visuelle(p, state["user_query"], role)
        analyses.append(f"**{os.path.basename(p)}**:\n{res}")

    print(f"{len(image_paths)} image(s) analysed")
    return {"image_evidence": "\n\n".join(analyses)}


def graph_node(state: AgentState, graph_paths: List[str]) -> dict:
    """Analyse the uploaded charts and graphs."""
    if not graph_paths or "graph" not in state["routing_decision"]:
        print("GRAPH NODE: skipped")
        return {"graph_evidence": ""}

    role = """Tu es un expert en visualisation de données et en graphes.

Analyse ce graphique :
1. Type de graphique (barres, courbes, camembert, réseau...)
2. Axes, légendes et unités
3. Valeurs clés et tendances (hausse, baisse, extrêmes)
4. S'il s'agit d'un graphe de réseau : sommets, arêtes, degrés, connexité

Ne fais aucune supposition basée sur d'autres documents."""

    print("Analysing graphs...")
    analyses = []
    for p in graph_paths:
        res = analyse_visuelle(p, state["user_query"], role)
        analyses.append(f"**{os.path.basename(p)}**:\n{res}")

    print(f"{len(graph_paths)} graph(s) analysed")
    return {"graph_evidence": "\n\n".join(analyses)}


def synthesis_node(state: AgentState) -> dict:
    """Final synthesis with two CrewAI agents."""
    print("Final synthesis with CrewAI...")

    # "groq/" tells litellm which provider to use. Without it, "openai/..."
    # is read as the OpenAI provider and the model is not found.
    llm = LLM(
        model="groq/" + VALID_GROQ_MODELS["large"],
        api_key=GROQ_API_KEY,
    )

    text_evidence = "\n".join(state["text_evidence"]) or "Aucun document textuel."
    image_evidence = state.get("image_evidence", "").strip() or "Aucune analyse d'image disponible."
    graph_evidence = state.get("graph_evidence", "").strip() or "Aucune analyse de graphique disponible."

    full_context = f"""
Question utilisateur: {state['user_query']}

=== DOCUMENTS TEXTUELS ===
{text_evidence}

=== ANALYSES IMAGES ===
{image_evidence}

=== ANALYSES GRAPHIQUES ===
{graph_evidence}
"""

    analyst = Agent(
        role="Analyste Multi-Modal Senior",
        goal="Extraire les insights clés de données textuelles, visuelles et graphiques",
        backstory="Expert en analyse multimodale.",
        llm=llm,
        verbose=False,
    )

    synthesizer = Agent(
        role="Synthétiseur Expert",
        goal="Créer une synthèse structurée et actionnable",
        backstory="Spécialiste en communication analytique.",
        llm=llm,
        verbose=False,
    )

    task1 = Task(
        description=f"""
Analyse séparément chaque source.

1. DOCUMENTS TEXTUELS
- extraire chiffres et faits importants

2. IMAGES
- décrire uniquement ce qui est visible
- ne pas inventer de relation avec les documents

3. GRAPHIQUES
- identifier le type
- extraire la structure ou les données

Ne mélange pas les sources si elles sont indépendantes.

CONTEXTE:
{full_context}
""",
        expected_output="Liste structurée des insights multimodaux avec explications.",
        agent=analyst,
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
        context=[task1],
    )

    crew = Crew(
        agents=[analyst, synthesizer],
        tasks=[task1, task2],
        process=Process.sequential,
        verbose=False,
        memory=False,
    )

    try:
        result = crew.kickoff()
        print("Synthesis done")
        return {"final_answer": str(result)}
    except Exception as e:
        return {"final_answer": f"Erreur CrewAI: {e}"}


# ---------- Workflow ----------
def create_workflow(vector_store: DocumentStore, image_paths: List[str], graph_paths: List[str]):
    workflow = StateGraph(AgentState)

    workflow.add_node("router", lambda s: router_node(s, image_paths, graph_paths))
    workflow.add_node("retriever", lambda s: retriever_node(s, vector_store))
    workflow.add_node("graph", lambda s: graph_node(s, graph_paths))
    workflow.add_node("image", lambda s: image_node(s, image_paths))
    workflow.add_node("synthesis", synthesis_node)

    workflow.set_entry_point("router")
    workflow.add_edge("router", "retriever")
    workflow.add_edge("retriever", "graph")
    workflow.add_edge("graph", "image")
    workflow.add_edge("image", "synthesis")
    workflow.add_edge("synthesis", END)

    return workflow.compile()


def run_multimodal_analysis(documents, image_paths, graph_paths, user_question):
    """Run the full multimodal analysis and return the final answer (Markdown)."""
    if not user_question or not user_question.strip():
        user_question = "Analyse complète des données : documents, images et graphiques."

    vector_store = DocumentStore(documents or [])

    initial_state = {
        "user_query": user_question,
        "routing_decision": [],
        "text_evidence": [],
        "graph_evidence": "",
        "image_evidence": "",
        "final_answer": "",
        "iteration_count": 0,
    }

    app = create_workflow(vector_store, image_paths or [], graph_paths or [])

    try:
        final_state = app.invoke(initial_state)
    except Exception as e:
        final_state = dict(initial_state)
        final_state["final_answer"] = f"Erreur workflow : {e}"

    if not final_state["final_answer"].strip():
        parts = []
        if documents:
            parts.append(f"Documents uploadés : {len(documents)}")
        if image_paths:
            parts.append("Images uploadées : " + ", ".join(os.path.basename(p) for p in image_paths))
        if graph_paths:
            parts.append("Graphiques uploadés : " + ", ".join(os.path.basename(p) for p in graph_paths))
        final_state["final_answer"] = "\n".join(parts) or "Aucune donnée détectée."

    return final_state["final_answer"]