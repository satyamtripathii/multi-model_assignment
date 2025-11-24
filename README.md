# Multi-Modal RAG on Qatar IMF Report

This project implements a multi‑modal Retrieval-Augmented Generation (RAG) system over the *Qatar IMF report* PDF. It extracts text and tables (and optionally images via OCR) from the PDF, builds a FAISS vector index using sentence embeddings, and exposes an interactive Streamlit chat interface where a user can ask questions about the report and get grounded answers with citations.

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

## Notes

- Embeddings are computed with `sentence-transformers/all-MiniLM-L6-v2`.
- The LLM used is `google/flan-t5-base` via `transformers` and LangChain.
- Image OCR uses `pytesseract` and requires the Tesseract binary to be installed on the system. If Tesseract is missing, image OCR is skipped but the rest of the pipeline still works.

## Deploying on Streamlit Community Cloud

1. Push this repository to GitHub (already configured for `satyamtripathii/multi-model_assignment`).
2. Go to https://share.streamlit.io and sign in with GitHub.
3. Create a new app and select this repo:
   - **Repository:** `satyamtripathii/multi-model_assignment`
   - **Branch:** `main`
   - **Main file path:** `multi-model_assignment/app.py` (or just `app.py` if the repo root is this folder).
4. Deploy – after the build finishes, Streamlit will provide a public URL you can use as the hosted link.
