# 🛂 SwiftVisa – AI-Based Smart Visa Eligibility Screening Agent

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python">
  <img src="https://img.shields.io/badge/Streamlit-Deployed-red?style=for-the-badge&logo=streamlit">
  <img src="https://img.shields.io/badge/FAISS-Vector%20Search-success?style=for-the-badge">
  <img src="https://img.shields.io/badge/HuggingFace-Transformers-yellow?style=for-the-badge">
  <img src="https://img.shields.io/badge/RAG-Powered-green?style=for-the-badge">
</p>

<p align="center">
An AI-powered Retrieval-Augmented Generation (RAG) application that helps users understand UK Student Visa requirements by answering questions directly from official PDF documents using semantic search and Large Language Models.
</p>

---

## 🌐 Live Demo

**Application:**  
[ai-visa-eligibility-screening-agent](https://ai-visa-eligibility-screening-agent-shpurnlsbpc5jju86lj4ah.streamlit.app/)

---

# 📌 Overview

SwiftVisa is an AI-powered Visa Eligibility Screening Assistant developed using Retrieval-Augmented Generation (RAG).

Instead of relying only on an LLM's internal knowledge, the system retrieves the most relevant information from uploaded visa policy PDFs using semantic search and then generates an accurate answer grounded in those documents.

This approach significantly reduces hallucinations while providing source-backed responses.

---

# ✨ Features

✅ AI-powered Question Answering

✅ Retrieval-Augmented Generation (RAG)

✅ PDF Document Processing

✅ Automatic Text Extraction

✅ Token-Based Chunking

✅ SentenceTransformer Embeddings

✅ FAISS Vector Database

✅ Semantic Search

✅ Source Citation

✅ Multi-document Support

✅ Interactive Chat Interface

✅ Modern Streamlit UI

---

# 🖥️ Application Preview

### Sidebar

- Chat Assistant
- Document Processing
- System Status
- Indexed Documents
- Chunk Statistics

### Chat Assistant

- Conversational AI
- Context-aware responses
- Source references
- Chat history

### Document Processing

- Upload PDFs
- Generate embeddings
- Build vector database
- Monitor processing status

---

# ⚙️ Tech Stack

| Category | Technology |
|-----------|------------|
| Language | Python |
| Frontend | Streamlit |
| Embeddings | Sentence Transformers (all-MiniLM-L6-v2) |
| Vector Database | FAISS |
| LLM | Google FLAN-T5 |
| NLP | Hugging Face Transformers |
| PDF Parsing | PyPDF2 |
| Numerical Computing | NumPy |

---

# 🏗️ System Architecture

```
                  PDF Documents
                        │
                        ▼
             Text Extraction (PyPDF2)
                        │
                        ▼
            Token Based Chunking
                        │
                        ▼
      Sentence Transformer Embeddings
                        │
                        ▼
            FAISS Vector Database
                        │
                User Query
                        │
                        ▼
          Semantic Similarity Search
                        │
                        ▼
             Top Relevant Chunks
                        │
                        ▼
         FLAN-T5 Language Model
                        │
                        ▼
          AI Generated Response
```

---

# 🚀 Workflow

### Step 1

Upload Visa PDF Documents.

---

### Step 2

Extract text from every page.

---

### Step 3

Split documents into overlapping chunks.

---

### Step 4

Generate embeddings using:

```
sentence-transformers/all-MiniLM-L6-v2
```

---

### Step 5

Store embeddings inside FAISS.

---

### Step 6

User asks a question.

---

### Step 7

Semantic Search retrieves Top-K relevant chunks.

---

### Step 8

Retrieved context is passed to the LLM.

---

### Step 9

The AI generates a grounded response with source citations.

---

# 📂 Project Structure

```
SwiftVisa/
│
├── .devcontainer/
├── .streamlit/
├── data/
│   ├── UK student visa.pdf
│
├── output/
│   ├── all_chunks.jsonl
│   ├── chunks_with_embeddings.jsonl
│
├── complete-swift-visa.py
├── requirements.txt
├── README.md
└── screenshots/
```

---

# 🧠 AI Models Used

### Embedding Model

```
sentence-transformers/all-MiniLM-L6-v2
```

Purpose

- Semantic Embedding
- Similarity Search

---

### Language Model

```
google/flan-t5-small
```

Purpose

- Context-aware Answer Generation

---

# 📊 RAG Pipeline

```
User Question
      │
      ▼
Embedding Generation
      │
      ▼
FAISS Similarity Search
      │
      ▼
Top K Chunks Retrieved
      │
      ▼
Prompt Construction
      │
      ▼
FLAN-T5
      │
      ▼
Generated Answer
```

---

# 📈 Advantages

- Fast semantic retrieval
- Reduced hallucinations
- Source-backed responses
- Handles multiple PDFs
- Easy deployment
- Lightweight architecture
- Scalable RAG pipeline

---

# 💻 Installation

Clone the repository

```bash
git clone https://github.com/yourusername/SwiftVisa.git
```

Move into project

```bash
cd SwiftVisa
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run application

```bash
streamlit run app.py
```

---

# 📦 Requirements

```
streamlit
sentence-transformers
transformers
faiss-cpu
PyPDF2
numpy
torch
```

---

# 🎯 Future Improvements

- GPT-based answer generation
- OCR support for scanned PDFs
- Multi-language support
- Conversation memory
- User authentication
- Cloud vector database
- Document summarization
- Voice interaction

---

# 👨‍💻 Author

**Anuj Kumar**

B.Tech Artificial Intelligence & Machine Learning

---

# ⭐ If you like this project

Please consider giving it a ⭐ on GitHub!
