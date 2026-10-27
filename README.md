# PDF Question Answering with LangChain, Gemini & Qdrant

A simple Python-based **Retrieval-Augmented Generation (RAG)** project that allows users to ask questions about PDF documents using LangChain, Google Gemini, and Qdrant Vector Database.

## How It Works

### 1. PDF Indexing
- Load PDF documents using `PyPDFLoader`.
- Split documents into smaller chunks using `RecursiveCharacterTextSplitter`.
- Generate vector embeddings using `gemini-embedding-001`.
- Store embeddings and document metadata in Qdrant.

### 2. Question Answering
- Accept user questions through the terminal.
- Retrieve relevant PDF content using Qdrant similarity search.
- Build context using retrieved text, page numbers, and file sources.
- Generate answers using `gemini-3.5-flash-lite`.

## Tech Stack

- **Python** – Core programming language
- **LangChain** – Document loading, chunking, and retrieval
- **Google Gemini Embeddings** – Vector embeddings
- **Qdrant** – Vector database

## Installation

Install the required dependencies:

```bash
pip install python-dotenv pypdf langchain-community langchain-text-splitters langchain-google-genai langchain-qdrant openai
```

## Environment Setup

Create a `.env` file and configure your API credentials:

```env
GOOGLE_API_KEY=your_gemini_api_key
OPENAI_API_KEY=your_gemini_api_key
OPENAI_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
```

## Qdrant Setup

Start Qdrant locally using Docker:

```bash
docker run -d --name qdrant ```

Qdrant URL: `http://localhost:`

## Run the Project

**Step 1: Index PDF Documents**

Save the first script as `index.py`, and place `rohit1.pdf` in the same folder.

```bash
python index.py
```

**Step 2: Ask Questions**

Save the second script as `CHAT.py`.

```bash
python query.py
```

Enter a question to retrieve relevant information from the indexed PDF and generate an AI response.

## Important Notes

- Both scripts must use the same Qdrant collection name.
- Use a single collection name string, such as `"newdata"`, rather than a list.
- Keep API keys and private PDF files out of public repositories.
- Ensure Qdrant is running before executing the scripts.
- Adjust the example Python filenames to match your actual files.

## Project Overview

This project demonstrates the basic workflow of a RAG application, including document ingestion, text chunking, vector embeddings, semantic retrieval, and context-based AI question answering.
