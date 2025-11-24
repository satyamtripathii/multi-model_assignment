# 📘 Multi-Modal RAG System on Qatar IMF Report

A production-grade **Multi-Modal Retrieval-Augmented Generation (RAG)** system built on the **Qatar IMF Article IV Report (2024)**.

This system extracts **text, tables, and OCR from images**, generates semantic embeddings, stores them in a FAISS vector index, and answers natural-language queries with **grounded, citation-backed responses** through a **Streamlit chat interface**.

---

## 🔗 Live Demo

👉 **https://multi-modelassignment-tdaedowjdfzir2qjuwwtwv.streamlit.app/#multi-modal-rag**

---


## 🖼 Screenshot (UI)

<img width="1920" height="1080" alt="Screenshot (151)" src="https://github.com/user-attachments/assets/559f44c9-13e1-4b2f-bb48-dd805bf20af7" />


---

## 🚀 Features

- Extracts **text, tables, and OCR from images** using PyMuPDF + Tesseract  
- Multi-modal chunking with metadata (page number, type, source)  
- Embedding generation using **Sentence Transformers – all-MiniLM-L6-v2**  
- FAISS vector similarity search (fast & accurate)  
- Optional BM25/keyword boosting  
- FLAN-T5-based LLM Question Answering  
- Citation-backed responses  
- Full Streamlit UI deployed on Streamlit Cloud  
- Accuracy enhancements (fraction normalization, soft validation, reranking)

---

## 🧠 System Architecture Diagram

```
                ┌────────────────────────┐
                │    PDF Document        │
                └───────────┬────────────┘
                            │
      ┌───────────────────────────────────────────────────┐
      │         Multi-Modal Document Processor            │
      │  (Text Extractor • Table Extractor • OCR Engine) │
      └───────────┬──────────────┬──────────────┬────────┘
                  │              │              │
              Text Chunks   Table Chunks   Image/OCR Chunks
                  │              │              │
                  └──────────────┬──────────────┘
                                 │
        ┌──────────────────────────────────────────────┐
        │   Embedding Generator (all-MiniLM-L6-v2)     │
        └───────────────────────┬──────────────────────┘
                                │
                        Vector Embeddings
                                │
      ┌────────────────────────────────────────────────────┐
      │   FAISS Vector Store (Semantic Similarity Search) │
      └───────────────────────┬────────────────────────────┘
                              │
                        Top-k Relevant Chunks
                              │
        ┌──────────────────────────────────────────────┐
        │    LLM QA Engine (FLAN-T5 + Post Processing) │
        └───────────────────────┬──────────────────────┘
                                │
                      Grounded Answer + Citations
                                │
        ┌──────────────────────────────────────────────┐
        │     Streamlit User Interface (Chatbot UI)     │
        └──────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
multi-modal_assignment/
│
├── app.py                     # Streamlit web app
├── config.py                  # Config & directory setup
├── process_document.py        # Extract text/tables/images
├── document_processor.py      # Multi-modal PDF parser
├── create_embeddings.py       # Build FAISS embeddings index
├── vector_store.py            # FAISS + embedding logic
├── llm_qa.py                  # LLM QA pipeline
├── run_pipeline.py            # Full pipeline runner
├── requirements.txt           # Dependencies
│
└── data/
    ├── raw/qatar_test_doc.pdf
    ├── processed/extracted_chunks.json
    └── vector_store/faiss_index/
```

---

## ⚙️ Setup & Local Run

### 1️⃣ Create virtual environment
```bash
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 2️⃣ Install dependencies
```bash
pip install -r requirements.txt
```

### 3️⃣ Add the Qatar IMF PDF
Place file at:
```
data/raw/qatar_test_doc.pdf
```

### 4️⃣ Run entire pipeline
```bash
python run_pipeline.py
```

Or step-by-step:
```bash
python config.py
python process_document.py
python create_embeddings.py
```

### 5️⃣ Start Streamlit app
```bash
streamlit run app.py
```

---

## 🛠 Accuracy Improvements Implemented

### ✔ Fraction & Numeric Normalization
Handled in `document_processor.py`:
- ½ → 0.5  
- 5½ → 5.5  
- ¼ → 0.25  
- ¾ → 0.75  

### ✔ LLM Typo Correction  
Handled in `llm_qa.py`:
- “512 percent” → “5.5 percent”

### ✔ Soft Validation & Relevance Scoring
- LLM labels chunks as **YES / PARTIAL / NO**  
- Scores converted into **1.0 / 0.5 / 0.0**  
- Keyword matching boosts relevance  
- Final score = max(llm_score, keyword_score)

### ✔ Keyword-aware Boosting
Boost chunks containing:
- “GDP”, “growth”, “inflation”, “fiscal”, “projection”  
- Deprioritize tables for conceptual questions

---

## 🔄 Updated Pipeline Instructions

Whenever you modify the PDF or extraction logic:

```bash
python process_document.py
python create_embeddings.py
```

Then restart app:

```bash
streamlit run app.py
```

---

## 🧪 Smoke Test

Run:

```bash
python smoke_test_gdp.py
```

This quickly checks:
- GDP forecast retrieval  
- Chunk selection  
- Answer grounding  

---

## ⚠ Known Issues

| Issue | Reason | Fix |
|------|--------|------|
| OCR missing | Tesseract not installed | Install Tesseract & rerun processing |
| Long context warnings | FLAN-T5 limit 512 tokens | Reduce chunk size |
| HF model download errors | Internet / Access issue | Falls back to rank-based scoring |

---

## ☁ Streamlit Cloud Deployment

Steps:

1. Push repo to GitHub  
2. Open https://share.streamlit.io  
3. Select repo:

```
Repository: satyamtripathii/multi-model_assignment
Branch: main
Main file: app.py
```

4. Deploy 🎉

---

## 🧑‍💻 Author

**Satyam Tripathi**  
B.Tech CSE  
Pranveer Singh Institute of Technology  
2022–26

---



   - **Main file path:** `multi-model_assignment/app.py`.
4. Deploy – after the build finishes, Streamlit will provide a public URL you can use as the hosted link.
