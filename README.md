<img width="1914" height="908" alt="Monitor Tracking" src="https://github.com/user-attachments/assets/27a1878c-3ec4-438c-993f-361f1b7461d0" />



## 📋 Project Overview
A sophisticated Retrieval-Augmented Generation (RAG) system built with LangGraph that intelligently routes questions between document retrieval and web search, with quality control mechanisms for hallucinations and relevance scoring.

## 🎯 Features
Intelligent Question Routing: Automatically determines whether to search the SCADA FAT document or use web search

Multi-stage Quality Control:

Document relevance grading

Hallucination detection

Answer quality assessment

Query Optimization: Rewrites queries for better retrieval performance

Modular Architecture: Clear separation of concerns with stateful workflow management

Multi-source Retrieval: Combines local document knowledge with web search capabilities

## 🏗️ Architecture
####Core Components
Document Processing Pipeline

PDF loading with PyPDFLoader

Text splitting with RecursiveCharacterTextSplitter

Vector embeddings using Ollama (nomic-embed-text)

FAISS vector store for efficient similarity search

Intelligent Router

Uses structured LLM output to classify questions

Routes to vectorstore for SCADA FAT document questions

Routes to web search for general questions

Quality Control Layers

Document Grader: Filters irrelevant retrieved documents

Hallucination Grader: Ensures answers are factually grounded

Answer Grader: Verifies answers address the original question

Query Optimization

Question rewriter improves retrieval performance

Dynamic query transformation based on retrieval results

#### Workflow Graph
The system implements a state machine with conditional edges:

START → Route Question → {Web Search or Vectorstore}

Vectorstore path includes: Retrieve → Grade Documents → {Generate or Transform Query}

Quality checks: Generate → Grade Hallucinations/Answer → {END, Transform Query, or Regenerate}

## 🛠️ Installation & Setup
Prerequisites
Python 3.8+

Ollama installed and running with nomic-embed-text model

API keys for:

Groq API

Tavily Search API

Hugging Face API (optional)

Environment Setup
Clone the repository and install dependencies:

bash
pip install -r requirements.txt
Create a .env file with your API keys:

env
groq_api_key=your_groq_api_key
TAVILY_API_KEY=your_tavily_api_key
hf_api_key=your_huggingface_key
Ensure Ollama is running and has the required models:

bash
ollama pull nomic-embed-text:latest
ollama pull llama-3.1-8b-instant  # or ensure Groq API access
Required Libraries
Install with:

bash
pip install langchain langchain-community langchain-groq langchain-ollama langgraph faiss-cpu pypdf python-dotenv
## 📁 File Structure
text
├── SCADA_FAT_RAG.py          # Main implementation file
├── FAT_SCADA_AL KUS.pdf      # Primary document for RAG
├── .env                      # Environment variables
├── requirements.txt          # Python dependencies
└── README.md                 # This file
🚀 Usage
Basic Usage
python
# Initialize and run the workflow
result = graph.invoke({"question": "What are the compliance requirements?"})
print(result["generation"])
Custom Questions
The system handles two types of questions:

SCADA FAT Document Questions (routes to vectorstore):

Compliance requirements

Architecture details

FAT/SAT procedures

Any content from the 14 annexures

General Questions (routes to web search):

General technical questions

Current events

Information not in the document

Example Queries
python
# Document-specific questions
graph.invoke({"question": "Explain the backup procedures mentioned in Annexure 8"})
graph.invoke({"question": "What are the warranty terms?"})

# General questions (will use web search)
graph.invoke({"question": "What is the latest version of SCADA technology?"})
⚙️ Configuration
Document Processing
Chunk Size: 900 characters

Chunk Overlap: 100 characters

Separators: Hierarchical text splitting with \n#, \n##, \n###, \n\n, \n, space

Model Configuration
Embeddings: Ollama nomic-embed-text:latest

LLM: Groq llama-3.1-8b-instant (for all LLM operations)

Web Search: Tavily API with top 3 results

Quality Thresholds
Document relevance grading: Binary (yes/no)

Hallucination detection: Binary (yes/no)

Answer quality: Binary (yes/no)

🔍 How It Works
Step-by-Step Process
Question Reception: User submits a question

Routing Decision: LLM classifies question as document-specific or general

Retrieval Phase:

Document path: Retrieve from FAISS → Grade relevance → Filter irrelevant docs

Web path: Search using Tavily API

Generation Phase: Generate answer using RAG prompt

Quality Assurance:

Check for hallucinations

Verify answer addresses question

Final Output or Recursion: Return answer or refine query and retry

State Management
The system uses a GraphState dictionary that maintains:

question: Current question being processed

generation: LLM-generated answer

documents: Retrieved documents for context

## 📚 Dependencies
Core: langchain, langgraph, pydantic

Models: langchain-groq, langchain-ollama

Document Processing: pypdf, faiss-cpu

Search: tavily-python

Utilities: python-dotenv, typing-extensions
