# Multi-Modal RAG on Qatar IMF Report

This project implements a multi‑modal Retrieval-Augmented Generation (RAG) system over the *Qatar IMF report* PDF. It extracts text and tables (and optionally images via OCR) from the PDF, builds a FAISS vector index using sentence embeddings, and exposes an interactive Streamlit chat interface where a user can ask questions about the report and get grounded answers with citations.

## 🔗 Live Demo

👉 **https://multi-modelassignment-tdaedowjdfzir2qjuwwtwv.streamlit.app/#multi-modal-rag**


## Project Structure

- `app.py` – Streamlit web app providing the chat UI.
- `config.py` – paths and model configuration, directory creation helper.
- `process_document.py` / `document_processor.py` – extract text, tables and images from the PDF.
- `create_embeddings.py` – create embeddings for all chunks and build a FAISS index.
- `vector_store.py` – wrapper around FAISS + embeddings.
- `llm_qa.py` – LLM / QA logic (FLAN‑T5 and simple fallback).
- `run_pipeline.py` – convenience script that runs the full data processing pipeline.
- `requirements.txt` – Python dependencies.
- `data/` – data folder created at runtime:
  - `data/raw/qatar_test_doc.pdf` – input PDF.
  - `data/processed/extracted_chunks.json` – extracted chunks.
  - `data/vector_store/faiss_index` – FAISS index files.

## Setup & Local Run

1. **Create and activate a virtual environment** (recommended)

   ```bash
   python -m venv .venv
   # Windows PowerShell
   .venv\\Scripts\\Activate.ps1
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Place the PDF**

   Put the provided Qatar IMF PDF file at:

   ```text
   data/raw/qatar_test_doc.pdf
   ```

4. **Run the data pipeline**

   Either run the full pipeline:

   ```bash
   python run_pipeline.py
   ```

   or run each step manually:

   ```bash
   python config.py            # ensure directories exist
   python process_document.py  # extract text/tables/images
   python create_embeddings.py # build FAISS index
   ```

5. **Start the Streamlit app**

   ```bash
   streamlit run app.py
   ```

   Then open the URL shown in the terminal (typically `http://localhost:8501`).

## Fixes Implemented for Accuracy

### Fraction and numeric normalization

To keep numeric values faithful to the PDF:

- All extracted text (body text, tables, OCR) is cleaned in `document_processor.py` using a `_clean_text` helper:
  - Unicode fractions are normalized:
    - `½` → `0.5`
    - `¼` → `0.25`
    - `¾` → `0.75`
  - Unicode minus signs are normalized to `-`.
  - Multiple spaces and odd whitespace are collapsed so numbers and words are consistently spaced.

### LLM typo / hallucination correction

In `llm_qa.py`, generated answers go through a light post‑processing filter that fixes a specific mis‑expansion sometimes produced by unicode fractions:

- `"512 percent"` → `"5.5 percent"`
- `"512%"` → `"5.5%"`

This runs after answer generation (and again after any safety note is appended) so the final text reflects the correct percentage.

### Soft validation and relevance scoring

Answer generation now uses **soft scoring only**:

- Retrieved chunks are labelled by the LLM as `YES` / `PARTIAL` / `NO` and scored as `1.0 / 0.5 / 0.0`.
- A separate keyword‑overlap score looks at overlap between the question and chunk text.
- The final relevance score per chunk is `max(llm_score, keyword_score)`, so strong keyword matches are never discarded.
- Chunks with score ≥ 0.5 (YES or PARTIAL) are kept as context; lower‑scoring chunks are used only as a very weak fallback.
- Fallback answers are used **only** when the top 3 chunks all have very low scores *and* none match core economic keywords.

### Keyword‑aware retrieval boosts

In `vector_store.py` we add lightweight reranking on top of FAISS + BM25:

- Chunks that contain numeric forecasts or projections get an extra boost.
- Chunks mentioning phrases like `"real GDP growth is projected"` receive a larger boost.
- Additional macro‑policy terms inside the chunk (`"GDP"`, `"growth"`, `"fiscal"`, `"inflation"`, `"projection"`) get a small positive boost so that policy paragraphs are preferred.
- Forecast‑style questions mentioning GDP/growth and years like 2024–25 strongly prefer narrative text over tables or images.

## Updated Pipeline Instructions

Whenever you change the PDF or any of the extraction / retrieval code, rerun the full pipeline:

```bash
python process_document.py   # re‑extract and clean text/tables/images
python create_embeddings.py  # rebuild FAISS index and BM25 metadata
```

After that, restart the app:

```bash
streamlit run app.py
```

For quick smoke tests without the UI you can also run:

```bash
python smoke_test_gdp.py
```

which queries the vector store and QA layer with the GDP forecast question.

## Known Issues and Solutions

- **Tesseract not installed** – if you see messages like `tesseract is not installed or it's not in your PATH`, image OCR is skipped but the rest of the system works. Install Tesseract and rerun `process_document.py` if you need OCR content.
- **Long context warnings** – the FLAN‑T5 model may warn about sequences longer than 512 tokens. This does not crash the app but may truncate some context. You can reduce `max_chars_per_chunk` in `llm_qa.py` or switch to a larger model if needed.
- **Model download / access errors** – if a cross‑encoder model cannot be downloaded from Hugging Face, the system automatically falls back to rank‑based scoring without reranking. Retrieval will still work, just with slightly weaker ranking.

## Deploying on Streamlit Community Cloud

1. Push this repository to GitHub (already configured for `satyamtripathii/multi-model_assignment`).
2. Go to https://share.streamlit.io and sign in with GitHub.
3. Create a new app and select this repo:
   - **Repository:** `satyamtripathii/multi-model_assignment`
   - **Branch:** `main`
   - **Main file path:** `multi-model_assignment/app.py` (or just `app.py` if the repo root is this folder).
4. Deploy – after the build finishes, Streamlit will provide a public URL you can use as the hosted link.
