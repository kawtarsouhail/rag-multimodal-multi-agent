# RAG Multimodal Multi-Agent

## 📝 Description

Ce projet met en place un **système multi-agents avancé** basé sur le framework **LangGraph** et **CrewAI**, combiné à un moteur de **Récupération Augmentée par Génération (RAG)**.  
Le système permet de répondre à des requêtes utilisateur en analysant des **documents textuels**, **images**, et **graphes**, puis de générer une synthèse structurée.

Le workflow se compose de plusieurs étapes :  
1. **Router Node** : Détermine quels agents et outils sont nécessaires selon la requête de l’utilisateur (texte, graphique, image).  
2. **Retriever Node** : Recherche et récupère les documents pertinents dans le vector store.  
3. **CrewAI Node** : Plusieurs agents collaborent pour analyser les données et produire une réponse finale.

---

## ⚙️ Technologies utilisées

- **LangChain / LangGraph** : pour la construction et l'orchestration du workflow.  
- **CrewAI** : pour la synthèse multi-agent et l’analyse avancée des données.  
- **FAISS** : pour l’indexation vectorielle des documents.  
- **HuggingFace Embeddings** : pour encoder les documents textuels.  
- **Groq API** : pour l’exécution des LLM (modèles `llama3` et `mixtral`).  
- **Python 3.12+**

---
# 🚀 Système Multi-Modal Intelligent  
### Analyse Automatisée de Documents, Images et Graphiques avec Agents IA

---


## 🏗 Architecture du Système

```text
User Input
     ↓
Router (LLM-based decision)
     ↓
Retriever (RAG - FAISS)
     ↓
Graph Analyzer (Vision LLM)
     ↓
Image Analyzer (Vision LLM)
     ↓
Multi-Agent Synthesis (CrewAI)
     ↓
Final Multimodal Report ```

## 📂 Structure du projet
```markdown
 rag-multimodal-multi-agent/
├── sma.ipynb # Notebook principal
├── documents/ # Dossier contenant les documents à analyser
├── .env # Variables d'environnement (API keys)
├── README.md # Ce fichier ```
