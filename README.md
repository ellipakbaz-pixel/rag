# Graph-Enhanced RAG System

A sophisticated Retrieval-Augmented Generation (RAG) system designed for codebases. This system leverages static code analysis and knowledge graphs to understand the structure and relationships within code (like function calls and definitions), enabling more accurate and context-aware answers to user queries.

## Key Features

- **Code Knowledge Graph**: Builds a directed graph of files, classes, and functions to understand dependencies.
- **Context-Aware Indexing**: Enriches code chunks with information about their context (where they are defined, what they call, what calls them).
- **Hybrid Retrieval**: Combines semantic vector search with graph traversal to find relevant code that might not be semantically similar but is structurally related.
- **Reranking**: Uses specialized models to rerank search results for high precision.
- **Detailed Reporting**: Generates Markdown reports for every query, showing the reasoning, retrieved chunks, and the final answer.

## Prerequisites

- Python 3.8+
- **Voyage AI API Key**: For embedding (`voyage-code-3`) and reranking (`rerank-2.5`).
- **Google Gemini API Key**: For LLM generation (`gemini-2.0-flash`).

## Installation

1. Clone the repository.
2. Install the required dependencies:

```bash
pip install networkx voyageai google-genai lancedb pyarrow tree-sitter
```

## Quick Start

### 1. Configure API Keys

Set your API keys as environment variables:

```bash
export VOYAGE_API_KEY="your_voyage_api_key"
export GOOGLE_API_KEY="your_google_api_key"
```

> **Important Note**: The provided scripts (`run_index.py` and `run_query.py`) currently contain placeholder API keys that will override your environment variables. You must edit these files to remove the `os.environ` assignments or replace the values with your actual keys before running them.

### 2. Index Your Codebase

Run the indexing script to parse your code, build the graph, and generate embeddings. By default, it indexes the current directory.

```bash
python run_index.py /path/to/your/project
```

This will create:
- `lancedb_store/`: Vector database storage.
- `rag_data/`: Knowledge graph and metadata files.

### 3. Ask Questions

Once indexed, you can query the system:

```bash
python run_query.py "How does the authentication system work?"
```

The system will output the answer to the console and generate a detailed report in `query_result.md`.

## Documentation

For more detailed information, please refer to:

- [**Usage Guide**](usage.md): Detailed API usage, configuration options, and advanced features.
- [**Workflow**](workflow.md): In-depth explanation of the indexing and querying architecture.

## Project Structure

- `pipeline.py`: Main entry point class `GraphRAGPipeline`.
- `run_index.py` / `run_query.py`: CLI scripts for easy usage.
- `index_codebase.py` / `query_codebase.py`: Core logic implementations.
- `graph_builder.py`: Constructs the NetworkX knowledge graph.
- `document_chunker.py` / `chunk_enricher.py`: Handles code chunking and context enrichment.
- `embedding_service.py` / `llm_service.py`: Interfaces for AI models.
- `vector_store.py`: LanceDB interface.
