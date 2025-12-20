<img width="1914" height="908" alt="Monitor Tracking" src="https://github.com/user-attachments/assets/27a1878c-3ec4-438c-993f-361f1b7461d0" />

# 🔗 Adaptive RAG System with LangGraph, Groq & Web Search

An end-to-end production-ready Adaptive Retrieval-Augmented Generation (RAG) system that dynamically routes user queries between local vectorstore retrieval and live web search, applies multi-stage relevance filtering, query rewriting, and hallucination detection, and guarantees fact-grounded, question-aligned answers.

This project demonstrates agentic decision-making using LangGraph, making it suitable for enterprise RAG systems, SCADA / technical documentation QA, and reliable AI assistants.

## 🧠 Key Features

### 🔀 Intelligent Query Routing
Routes queries to:
- **Vectorstore** (FAISS + embeddings) for document-grounded questions
- **Web Search** (Tavily) for open-world or fresh information

### 📚 PDF-based Knowledge Ingestion
Chunked, embedded, and indexed technical documents (SCADA specs, manuals, etc.)

### 🧪 Document Relevance Grading
Filters irrelevant chunks using structured LLM evaluation

### ✍️ Query Rewriting Agent
Improves retrieval quality when documents are insufficient

### 🧠 Hallucination Detection
Ensures answers are strictly grounded in retrieved facts

### ✅ Answer–Question Alignment Check
Verifies the final answer actually resolves the user's question

### 🕸️ Agentic Workflow with LangGraph
Deterministic, debuggable, production-grade control flow

## 🏗️ Architecture Overview

```mermaid
flowchart TD
    START --> Router
    Router -->|Vectorstore| Retrieve
    Router -->|Web Search| WebSearch

    Retrieve --> GradeDocs
    GradeDocs -->|Relevant| Generate
    GradeDocs -->|Irrelevant| RewriteQuery

    RewriteQuery --> Router

    WebSearch --> Generate

    Generate --> HallucinationCheck
    HallucinationCheck -->|Grounded| AnswerCheck
    HallucinationCheck -->|Not Grounded| Generate

    AnswerCheck -->|Useful| END
    AnswerCheck -->|Not Useful| RewriteQuery
