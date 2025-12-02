# Graph-Enhanced RAG Usage Guide

This guide explains how to use the Graph-Enhanced RAG system for indexing your codebase and performing queries.

## Prerequisites

1.  **Python 3.8+**
2.  **API Keys**:
    *   **Voyage AI API Key**: For embeddings (`voyage-code-3`) and reranking (`rerank-2.5`).
    *   **Google Gemini API Key**: For LLM generation (`gemini-2.0-flash`).

## Installation

Ensure you have the required dependencies installed:

```bash
pip install networkx voyageai google-genai lancedb pyarrow tree-sitter
```

## Configuration

Set your API keys as environment variables or pass them directly to the pipeline:

```bash
export VOYAGE_API_KEY="your_voyage_api_key"
export GOOGLE_API_KEY="your_google_api_key"
```

## Usage

The main entry point is the `GraphRAGPipeline` class in `graph_rag.pipeline`.

### 1. Initialization

```python
from graph_rag.pipeline import GraphRAGPipeline

# Initialize the pipeline
pipeline = GraphRAGPipeline(
    db_path="./lancedb_store",      # Path to LanceDB database directory
    data_dir="./rag_data",          # Path to store graph JSON and other data
    voyage_api_key="your_key",      # Optional if VOYAGE_API_KEY env var is set
    gemini_api_key="your_key"       # Optional if GOOGLE_API_KEY env var is set
)
```

### 2. Indexing a Codebase

Indexing parses the code, builds the knowledge graph, generates embeddings, and stores everything. Run this once or when code changes significantly.

```python
from graph_rag.exceptions import (
    ExtractionError, GraphBuildError, EnrichmentError,
    EmbeddingError, StorageError
)

project_path = "/path/to/your/project"

try:
    result = pipeline.index(project_path)
    
    if result.success:
        print("Indexing successful!")
        print(f"Functions extracted: {result.function_count}")
        print(f"Graph nodes: {result.node_count}")
        print(f"Graph edges: {result.edge_count}")
        print(f"Enriched chunks: {result.chunk_count}")
        print(f"Embeddings generated: {result.embedding_count}")
    else:
        print(f"Indexing failed: {result.message}")
        
except ExtractionError as e:
    print(f"Content extraction failed: {e}")
except GraphBuildError as e:
    print(f"Graph building failed: {e}")
except EnrichmentError as e:
    print(f"Chunk enrichment failed: {e}")
except EmbeddingError as e:
    print(f"Embedding generation failed: {e}")
except StorageError as e:
    print(f"Storage operation failed: {e}")
```

### 3. Querying

Once indexed, you can ask questions about the codebase:

```python
from graph_rag.exceptions import IndexNotFoundError, RetrievalError, GenerationError

query = "How does the authentication system work?"

try:
    result = pipeline.query(
        query=query,
        top_k=5,            # Number of final chunks to use
        expand=True,        # Enable graph expansion (recommended)
        rerank_top_k=3,     # Number of chunks after reranking
        search_limit=10     # Initial vector search results
    )
    
    print("--- Answer ---")
    print(result.gemini_response)
    
    print(f"\n--- Retrieved {result.retrieved_count} chunks ---")
    
    # Inspect primary matches (from vector search + rerank)
    print("\nPrimary Matches:")
    for i, doc in enumerate(result.reranked_docs, 1):
        print(f"[{i}] {doc[:100]}...")
        
    # Inspect expanded context (from graph traversal)
    print("\nExpanded Context:")
    for i, context in enumerate(result.expanded_context, 1):
        print(f"[{i}] {context[:100]}...")

except IndexNotFoundError as e:
    print(f"No index found: {e}")
except RetrievalError as e:
    print(f"Retrieval failed: {e}")
except GenerationError as e:
    print(f"LLM generation failed: {e}")
```

## Advanced Usage

### Customizing Components

Access individual components for granular control:

```python
# Access the content extractor
functions = pipeline.content_extractor.extract("/path/to/project")

# Access the graph builder
graph = pipeline.graph_builder.build("/path/to/project")
pipeline.graph_builder.export_json(graph, "my_graph.json")

# Access the embedding service
query_embedding = pipeline.embedding_service.embed_query("some query")
doc_embeddings = pipeline.embedding_service.embed_documents(["code1", "code2"])
reranked = pipeline.embedding_service.rerank("query", ["doc1", "doc2"], top_k=2)

# Access the vector store
results = pipeline.vector_store.search(query_embedding, limit=10)
text = pipeline.vector_store.get_by_node_id("Function:/path:name:42")

# Access the LLM service
response = pipeline.llm_service.generate("prompt", stream=False)

# Access the context assembler
prompt = pipeline.context_assembler.assemble(
    primary_results=["chunk1", "chunk2"],
    secondary_results=["related1"],
    query="How does X work?"
)
```

### Using Individual Components

You can use components independently without the full pipeline:

```python
from graph_rag import (
    ContentExtractor, GraphBuilder, ChunkEnricher,
    EmbeddingService, VectorStore, GraphExpander,
    ContextAssembler, LLMService
)

# Extract content
extractor = ContentExtractor()
functions = extractor.extract("/path/to/project")

# Build graph
builder = GraphBuilder()
graph = builder.build("/path/to/project")

# Enrich chunks
enricher = ChunkEnricher(graph, functions, token_limit=1000)
chunks = enricher.enrich_all(split_large_chunks=True)

# Generate embeddings
embedding_service = EmbeddingService()
texts = [chunk.to_text() for chunk in chunks]
embeddings = embedding_service.embed_documents(texts)

# Store in vector database
store = VectorStore(db_path="./my_lancedb")
store.store(chunks, embeddings)

# Search
query_vec = embedding_service.embed_query("find authentication")
results = store.search(query_vec, limit=5)
```

### Check Index Status

```python
# Check if an index exists
if pipeline.is_indexed():
    print("Index found, ready to query")
else:
    print("No index found, please run index() first")

# Clear existing index
pipeline.clear_index()
```

### Data Persistence

The pipeline stores data in two locations:

1.  **LanceDB (`db_path`)**: Vector embeddings and enriched chunk text
    - Table: `code_embeddings`
    - Fields: `vector`, `text`, `node_id`, `filename`

2.  **Data Directory (`data_dir`)**: Serialized JSON files
    - `knowledge_graph.json`: NetworkX graph in node-link format
    - `master_list.json`: Extracted function metadata
    - `enriched_chunks.json`: Enriched chunks with graph context

Keep these directories to persist the index between runs.

## Data Models

### EnrichedChunk

Each chunk contains three sections when serialized:

```
[METADATA]
File: /path/to/file.py
Function: ClassName.method_name
Type: Function
Node_ID: Function:/path:name:line

[CONTEXT]
Defined_In: ClassName
Calls: helper_func, utils.process
Called_By: main, test_method

[CODE]
def method_name(self, arg1, arg2):
    ...
```

### Result Types

- `IndexResult`: `function_count`, `node_count`, `edge_count`, `chunk_count`, `embedding_count`, `success`, `message`
- `QueryResult`: `query`, `retrieved_count`, `reranked_docs`, `expanded_context`, `gemini_response`, `success`, `message`
- `SearchResult`: `text`, `node_id`, `distance`, `filename`
- `RerankResult`: `index`, `relevance_score`, `document`
- `ExpandedResult`: `node_id`, `relationship`, `content`
