# Local RAG on *AI Engineering*

A lightweight, fully local Retrieval-Augmented Generation (RAG) pipeline built around Chip Huyen’s *AI Engineering: Building Applications With Foundation Models*. The project retrieves relevant passages from the book and uses a local LLM to generate answers grounded in the retrieved context.

## Tech Stack

- **PDF Extraction:** PyMuPDF
- **Custom Chunking:** Structure-aware, paragraph-based chunker with heading detection and a token budget
- **Embedding Model:** [`BAAI/bge-small-en-v1.5`](https://huggingface.co/BAAI/bge-small-en-v1.5)
- **Vector Database:** ChromaDB with persistent local storage
- **LLM Runtime:** Ollama
- **Generation Model:** `qwen2.5:3b`

## How It Works

1. **Extract** text and page information from the PDF using PyMuPDF.
2. **Chunk** the text using a custom Python chunker that detects headings, groups lines into paragraphs, and respects a target token budget.
3. **Embed** each chunk using `BAAI/bge-small-en-v1.5`.
4. **Index** embeddings and metadata in a persistent ChromaDB collection.
5. **Retrieve** the top-k relevant chunks for a user query using vector similarity search.
6. **Generate** a context-grounded answer with `qwen2.5:3b` through Ollama.

## Custom Chunker

implemented custom structure-aware chunking logic.

- Detects headings using PDF font sizes and chapter-heading patterns.
- Groups extracted lines into paragraphs.
- Splits content around headings and a configurable token budget.
- Preserves metadata such as chapter, section, and page range for retrieved sources.

**Current configuration:** 400-token target; 730 chunks indexed in the initial dataset.

## Project Structure

```text
AIEngineering_RAG/
├── data/              # Source PDF (not committed)
├── chroma_db/         # Persistent vector database (generated)
├── src/
│   ├── chunk.py       # Custom PDF chunking
│   ├── config.py      # Environment-based configuration
│   ├── build_vectordb.py
│   ├── tokenizer.py
│   ├── retrieve.py    # Embedding and vector retrieval
│   ├── generate.py    # Ollama answer generation
│   └── rag.py         # End-to-end interactive RAG pipeline
├── .env.example
├── requirements.txt
└── README.md
```

## Configuration

```env
PDF_PATH=data/AI Engineering_Building Applications With Foundation Models by Chip Huyen.pdf
EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
CHROMA_PATH=chroma_db
COLLECTION_NAME=ai_engineering
TARGET_TOKENS=400
MODEL_NAME=qwen2.5:3b
```

## Current Features

- Local document ingestion and persistent vector storage
- Custom, structure-aware chunking
- Semantic retrieval using embedding similarity
- Context-grounded answer generation with Ollama
- Interactive question-answering loop
- Retrieved-source metadata for inspection

## Limitations

- Retrieval uses vector similarity search; hybrid search and reranking are not implemented.
- Conversation history is maintained only during the current session.
- Follow-up questions are not rewritten using conversation history before retrieval.
- Answer quality depends on the retrieved context and the local model.

## Future Improvements

- Add retrieval and answer-quality evaluation.
- Experiment with chunk sizes, overlap, and embedding models.
- Improve follow-up query handling and source citations.
- Add a lightweight web interface.

## Acknowledgements

Built as a hands-on project to explore the core components of RAG and local LLM applications while studying *AI Engineering* by Chip Huyen.
