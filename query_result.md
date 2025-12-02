# Graph-Enhanced RAG Query Report

**Generated:** 2025-12-01 19:14:47

---

## Query

```
"what is job of this code graph_rag"
```

---

## Summary

| Metric | Value |
|--------|-------|
| Vector Search Results | 3 |
| Reranked Results | 3 |
| Graph-Expanded Context | 16 |
| Total Context Chunks | 19 |

---

## LLM Response

Okay, I will answer your question about the job of the `graph_rag` code based on the provided context.

The `graph_rag` code implements a retrieval-augmented generation (RAG) pipeline that leverages a knowledge graph to enhance code understanding and querying. It allows users to ask questions about a codebase and receive answers grounded in the code's structure and content.

Here's a breakdown of the key components and their roles:

**1. Indexing Pipeline (`pipeline.GraphRAGPipeline.index`)**

*   **Purpose:**  Processes a project's source code to build an index for efficient querying.
*   **Steps:**
    *   **Content Extraction:**  Uses `content_extractor.ContentExtractor.extract` (Dep 1) to parse the project and extract function metadata (name, code, file path, etc.). The extracted data is stored in `self._master_list`.
    *   **Graph Building:**  Uses `graph_builder.GraphBuilder.build` (Dep 2) to construct a knowledge graph representing the relationships between code elements (files, classes, functions, calls). The graph is stored in `self._graph`.
    *   **Chunk Enrichment:**  Uses `chunk_enricher.ChunkEnricher.enrich_all` (Dep 3) to create enriched chunks of code, adding context to the extracted functions. The enriched chunks are stored in `self._enriched_chunks`.
    *   **Embedding Generation:**  Uses `embedding_service.EmbeddingService.embed_documents` (Dep 5) to generate vector embeddings for the enriched code chunks. These embeddings capture the semantic meaning of the code.
    *   **Vector Storage:**  Uses `vector_store.VectorStore.store` (Dep 6) to store the enriched chunks and their embeddings in a vector database (LanceDB). This allows for efficient similarity search.
    *   **Graph Export and Index Building:** Exports the knowledge graph to a JSON file using `self.graph_builder.export_json` and persists the master list and enriched chunks to JSON files using `self._save_master_list` (Dep 7) and `self._save_enriched_chunks` (Dep 8).  It also builds in-memory indices for faster lookups using `self._build_indices` (Dep 9).
*   **Output:** An `IndexResult` object containing statistics about the indexed project (number of functions, nodes, edges, chunks, embeddings).

**2. Query Pipeline (`pipeline.GraphRAGPipeline.query`)**

*   **Purpose:**  Answers user questions about the codebase using the indexed data.
*   **Steps:**
    *   **Index Loading:** Ensures the indexed data (graph, master list, enriched chunks) is loaded into memory using `self._ensure_index_loaded` (Dep 10). If the data is not already loaded, it loads it from the persisted JSON files and rebuilds the indices.
    *   **Query Embedding:**  Uses `embedding_service.EmbeddingService.embed_query` (Dep 11) to generate a vector embedding for the user's query.
    *   **Vector Search:**  Uses `vector_store.VectorStore.search` (Dep 12) to search the vector database for code chunks that are semantically similar to the query.
    *   **Reranking:** Uses `embedding_service.EmbeddingService.rerank` (Dep 13) to rerank the initial search results, improving the quality of the retrieved chunks.
    *   **Graph Expansion (Optional):**  Uses `pipeline.GraphRAGPipeline._expand_results` (Chunk 3) and `graph_expander.GraphExpander.expand` (Dep 16) to expand the search context by traversing the knowledge graph and finding related code elements (e.g., functions called by the retrieved chunks).
    *   **Context Assembly:**  Uses `context_assembler.ContextAssembler.assemble` (Dep 14) to assemble a prompt containing the retrieved code chunks, related code elements, and the user's query.  This prompt is designed to provide the LLM with the necessary context to answer the question.
    *   **LLM Generation:**  Uses `llm_service.LLMService.generate` (Dep 15) to generate a response to the user's query using a large language model (LLM).
*   **Output:** A `QueryResult` object containing the retrieved code chunks, expanded context, and the LLM's response.

**3. Graph Expansion (`pipeline.GraphRAGPipeline._expand_results`)**

*   **Purpose:** Expands the context of the search results by finding related code elements in the knowledge graph.
*   **Mechanism:** Uses `graph_expander.GraphExpander.expand` to traverse the graph, starting from the nodes identified in the initial vector search. It retrieves neighboring nodes (e.g., functions called by the initial results) and adds their code to the context.

**In Summary**

The `graph_rag` code provides a system for:

*   Indexing a codebase by extracting code, building a knowledge graph, and generating vector embeddings.
*   Answering questions about the codebase by searching for relevant code chunks, expanding the context using the knowledge graph, and generating a response using an LLM.

This approach combines the strengths of vector search (finding semantically similar code) with the structured knowledge provided by a graph, leading to more accurate and informative answers.


---

## Detailed Results

### Primary Matches (Vector Search + Reranking)

These are the most relevant code chunks found via semantic search and reranked for relevance.

#### [1] pipeline.GraphRAGPipeline.index

**File:** `E:\PYTHON_CODE _DESKTOP\finite-monkey\finite-monkey-engine\src\tree_sitter_parsing\graph_rag\pipeline.py`

**Node ID:** `Function:E:\PYTHON_CODE _DESKTOP\finite-monkey\finite-monkey-engine\src\tree_sitter_parsing\graph_rag\pipeline.py:pipeline.GraphRAGPipeline.index:155`

**Distance:** 0.9597

**Context:**
- Defined In: `pipeline.GraphRAGPipeline`
- Calls: `logger.info`, `os.path.exists`, `os.makedirs`, `content_extractor.ContentExtractor.extract`, `len`, `ExtractionError`, `graph_builder.GraphBuilder.build`, `self._graph.number_of_nodes`, `self._graph.number_of_edges`, `GraphBuildError`, `ChunkEnricher`, `chunk_enricher.ChunkEnricher.enrich_all`, `EnrichmentError`, `models.EnrichedChunk.to_text`, `embedding_service.EmbeddingService.embed_documents`, `EmbeddingError`, `vector_store.VectorStore.store`, `StorageError`, `self.graph_builder.export_json`, `pipeline.GraphRAGPipeline._save_master_list`, `pipeline.GraphRAGPipeline._save_enriched_chunks`, `pipeline.GraphRAGPipeline._build_indices`, `IndexResult`
- Called By: None

**Code:**
```python
def index(self, project_path: str) -> IndexResult:
        """
        Index a project for RAG.
        
        Executes the 6-step indexing pipeline:
        1. Extract content (project_parser)
        2. Build graph (kg_implementation)
        3. Enrich chunks
        4. Generate embeddings
        5. Store in LanceDB
        6. Export graph JSON
        
        Args:
            project_path: Path to the project directory to index.
            
        Returns:
            IndexResult with stats (function_count, node_count, etc.)
            
        Raises:
            ExtractionError: If content extraction fails.
            GraphBuildError: If graph building fails.
            EnrichmentError: If chunk enrichment fails.
            EmbeddingError: If embedding generation fails.
            StorageError: If storage operations fail.
        """
        logger.info(f"Starting indexing for project: {project_path}")
        
        # Ensure data directory exists
        if self._data_dir and not os.path.exists(self._data_dir):
            os.makedirs(self._data_dir, exist_ok=True)
        
        # Step 1: Content Extraction
        try:
            logger.info("Step 1: Extracting content...")
            self._master_list = self.content_extractor.extract(project_path)
            function_count = len(self._master_list)
            logger.info(f"Extracted {function_count} functions")
        except Exception as e:
            raise ExtractionError(
                f"Failed to extract content from {project_path}: {e}",
                original_error=e
            )
        
        # Step 2: Graph Building
        try:
            logger.info("Step 2: Building knowledge graph...")
            self._graph = self.graph_builder.build(project_path)
            node_count = self._graph.number_of_nodes()
            edge_count = self._graph.number_of_edges()
            logger.info(f"Built graph with {node_count} nodes and {edge_count} edges")
        except Exception as e:
            raise GraphBuildError(
                f"Failed to build knowledge graph: {e}",
                original_error=e
            )
        
        # Step 3: Chunk Enrichment
        try:
            logger.info("Step 3: Enriching chunks...")
            enricher = ChunkEnricher(self._graph, self._master_list)
            self._enriched_chunks = enricher.enrich_all()
            chunk_count = len(self._enriched_chunks)
            logger.info(f"Created {chunk_count} enriched chunks")
        except Exception as e:
            raise EnrichmentError(
                f"Failed to enrich chunks: {e}",
                original_error=e
            )
        
        # Step 4: Embedding Generation
        try:
            logger.info("Step 4: Generating embeddings...")
            texts = [chunk.to_text() for chunk in self._enriched_chunks]
            embeddings = self.embedding_service.embed_documents(texts)
            embedding_count = len(embeddings)
            logger.info(f"Generated {embedding_count} embeddings")
        except Exception as e:
            raise EmbeddingError(
                f"Failed to generate embeddings: {e}",
                original_error=e
            )
        
        # Step 5: LanceDB Storage
        try:
            logger.info("Step 5: Storing in LanceDB...")
            self.vector_store.store(self._enriched_chunks, embeddings)
            logger.info(f"Stored {len(self._enriched_chunks)} chunks in LanceDB")
        except Exception as e:
            raise StorageError(
                f"Failed to store in LanceDB: {e}",
                original_error=e
            )
        
        # Step 6: Export Graph JSON and persist master list
        try:
            logger.info("Step 6: Exporting graph and master list...")
            self.graph_builder.export_json(self._graph, self._graph_path)
            self._save_master_list()
            self._save_enriched_chunks()
            logger.info(f"Exported graph to {self._graph_path}")
        except Exception as e:
            raise StorageError(
                f"Failed to export graph JSON: {e}",
                original_error=e
            )
        
        # Build lookup indices
        self._build_indices()
        
        logger.info("Indexing complete!")
        
        return IndexResult(
            function_count=function_count,
            node_count=node_count,
            edge_count=edge_count,
            chunk_count=chunk_count,
            embedding_count=embedding_count,
            success=True,
            message=f"Successfully indexed {project_path}"
        )
```

---

#### [2] pipeline.GraphRAGPipeline.query

**File:** `E:\PYTHON_CODE _DESKTOP\finite-monkey\finite-monkey-engine\src\tree_sitter_parsing\graph_rag\pipeline.py`

**Node ID:** `Function:E:\PYTHON_CODE _DESKTOP\finite-monkey\finite-monkey-engine\src\tree_sitter_parsing\graph_rag\pipeline.py:pipeline.GraphRAGPipeline.query:276`

**Distance:** 0.9723

**Context:**
- Defined In: `pipeline.GraphRAGPipeline`
- Calls: `logger.info`, `pipeline.GraphRAGPipeline._ensure_index_loaded`, `embedding_service.EmbeddingService.embed_query`, `RetrievalError`, `vector_store.VectorStore.search`, `len`, `embedding_service.EmbeddingService.rerank`, `logger.warning`, `pipeline.GraphRAGPipeline._expand_results`, `set`, `context_assembler.ContextAssembler.assemble`, `llm_service.LLMService.generate`, `QueryResult`
- Called By: None

**Code:**
```python
def query(
        self,
        query: str,
        top_k: int = 5,
        expand: bool = True,
        rerank_top_k: int = 3,
        search_limit: int = 10,
    ) -> QueryResult:
        """
        Query the indexed codebase.
        
        Executes the 6-step query pipeline:
        1. Embed query
        2. Vector search in LanceDB
        3. Rerank results with Voyage
        4. Graph expansion (if enabled)
        5. Assemble context
        6. Generate response with Gemini
        
        Args:
            query: Natural language query about the codebase.
            top_k: Number of final results to use (default: 5).
            expand: Whether to perform graph expansion (default: True).
            rerank_top_k: Number of results after reranking (default: 3).
            search_limit: Number of initial vector search results (default: 10).
            
        Returns:
            QueryResult with retrieved_chunks, expanded_context, response.
            
        Raises:
            IndexNotFoundError: If no index exists.
            RetrievalError: If retrieval operations fail.
            GenerationError: If LLM generation fails.
        """
        logger.info(f"Processing query: {query[:100]}...")
        
        # Ensure we have indexed data loaded
        self._ensure_index_loaded()
        
        # Step 1: Query Embedding
        try:
            logger.info("Step 1: Embedding query...")
            query_embedding = self.embedding_service.embed_query(query)
        except Exception as e:
            raise RetrievalError(
                f"Failed to embed query: {e}",
                original_error=e
            )
        
        # Step 2: Vector Search
        try:
            logger.info("Step 2: Searching LanceDB...")
            search_results = self.vector_store.search(
                query_embedding, limit=search_limit
            )
            logger.info(f"Found {len(search_results)} initial results")
        except Exception as e:
            raise RetrievalError(
                f"Failed to search vector store: {e}",
                original_error=e
            )
        
        # Step 3: Reranking
        try:
            logger.info("Step 3: Reranking results...")
            if search_results:
                documents = [r.text for r in search_results]
                reranked = self.embedding_service.rerank(
                    query, documents, top_k=rerank_top_k
                )
                reranked_docs = [r.document for r in reranked]
                reranked_node_ids = [
                    search_results[r.index].node_id for r in reranked
                ]
                logger.info(f"Reranked to {len(reranked_docs)} results")
            else:
                reranked_docs = []
                reranked_node_ids = []
        except Exception as e:
            # Graceful degradation: use raw search results
            logger.warning(f"Reranking failed, using raw results: {e}")
            reranked_docs = [r.text for r in search_results[:rerank_top_k]]
            reranked_node_ids = [r.node_id for r in search_results[:rerank_top_k]]
        
        # Step 4: Graph Expansion
        expanded_context: List[str] = []
        if expand and reranked_node_ids:
            try:
                logger.info("Step 4: Expanding via graph...")
                expanded_results = self._expand_results(
                    reranked_node_ids, set(reranked_node_ids)
                )
                expanded_context = [r.content for r in expanded_results]
                logger.info(f"Expanded to {len(expanded_context)} related chunks")
            except Exception as e:
                # Graceful degradation: continue without expansion
                logger.warning(f"Graph expansion failed: {e}")
                expanded_context = []
        
        # Step 5: Context Assembly
        try:
            logger.info("Step 5: Assembling context...")
            prompt = self.context_assembler.assemble(
                primary_results=reranked_docs,
                secondary_results=expanded_context,
                query=query
            )
        except Exception as e:
            raise RetrievalError(
                f"Failed to assemble context: {e}",
                original_error=e
            )
        
        # Step 6: LLM Generation
        try:
            logger.info("Step 6: Generating response...")
            response = self.llm_service.generate(prompt, stream=False)
            logger.info("Response generated successfully")
        except Exception as e:
            # Graceful degradation: return context without response
            logger.warning(f"LLM generation failed: {e}")
            return QueryResult(
                query=query,
                retrieved_count=len(reranked_docs),
                reranked_docs=reranked_docs,
                expanded_context=expanded_context,
                gemini_response="",
                success=False,
                message=f"LLM generation failed: {e}"
            )
        
        return QueryResult(
            query=query,
            retrieved_count=len(reranked_docs),
            reranked_docs=reranked_docs,
            expanded_context=expanded_context,
            gemini_response=response,
            success=True,
            message="Query completed successfully"
        )
```

---

#### [3] pipeline.GraphRAGPipeline._expand_results

**File:** `E:\PYTHON_CODE _DESKTOP\finite-monkey\finite-monkey-engine\src\tree_sitter_parsing\graph_rag\pipeline.py`

**Node ID:** `Function:E:\PYTHON_CODE _DESKTOP\finite-monkey\finite-monkey-engine\src\tree_sitter_parsing\graph_rag\pipeline.py:pipeline.GraphRAGPipeline._expand_results:532`

**Distance:** 1.0115

**Context:**
- Defined In: `pipeline.GraphRAGPipeline`
- Calls: `GraphExpander`, `graph_expander.GraphExpander.expand`
- Called By: `pipeline.GraphRAGPipeline.query`

**Code:**
```python
def _expand_results(
        self,
        node_ids: List[str],
        exclude_ids: Set[str]
    ) -> List[ExpandedResult]:
        """
        Expand search results using graph relationships.
        
        Args:
            node_ids: Node IDs from primary search results.
            exclude_ids: Node IDs to exclude from expansion.
            
        Returns:
            List of ExpandedResult with related code.
        """
        if self._graph is None:
            return []
        
        expander = GraphExpander(
            graph=self._graph,
            master_list=self._master_list_by_node_id or {},
            enriched_chunks=self._enriched_chunks_by_node_id
        )
        
        return expander.expand(node_ids, exclude_ids)
```

---

### Graph-Expanded Context (Related Code)

These are related code chunks discovered by traversing the knowledge graph (CALLS/CALLED_BY relationships).

#### [Dep 1] content_extractor.ContentExtractor.extract

**File:** `E:\PYTHON_CODE _DESKTOP\finite-monkey\finite-monkey-engine\src\tree_sitter_parsing\graph_rag\content_extractor.py`

**Relationship:** Graph neighbor (CALLS/CALLED_BY)

**Context:**
- Defined In: `content_extractor.ContentExtractor`
- Calls: `os.path.exists`, `FileNotFoundError`, `os.path.isdir`, `ValueError`, `project_parser.parse_project`, `logger.info`, `len`, `logger.error`
- Called By: `pipeline.GraphRAGPipeline.index`

**Code:**
```python
def extract(self, root_path: str) -> List[FunctionDict]:
        """
        Extract all functions from the project.
        
        Args:
            root_path: Path to the project root directory.
            
        Returns:
            List of function dictionaries with keys:
            - name: str (qualified name like "Contract.method")
            - content: str (full source code)
            - file_path: str
            - start_line: int
            - end_line: int
            - calls: List[str] (called function names)
            - contract_name: str (parent class/contract)
            - visibility: str
            
        Raises:
            FileNotFoundError: If root_path does not exist.
            ValueError: If root_path is not a directory.
        """
        # Validate input path
        if not os.path.exists(root_path):
            raise FileNotFoundError(f"Project path does not exist: {root_path}")
        
        if not os.path.isdir(root_path):
            raise ValueError(f"Project path is not a directory: {root_path}")
        
        try:
            # Call parse_project with the filter
            # parse_project returns: (functions, functions_to_check, chunks)
            functions, functions_to_check, _ = parse_project(
                root_path, 
                project_filter=self._filter
            )
            
            logger.info(
                f"Extracted {len(functions)} total functions, "
                f"{len(functions_to_check)} after filtering from {root_path}"
            )
            
            # Return the filtered functions (functions_to_check)
            # These have passed the filter criteria
            return functions_to_check
            
        except Exception as e:
            logger.error(f"Error extracting content from {root_path}: {e}")
            raise
```

---

#### [Dep 2] graph_builder.GraphBuilder.build

**File:** `E:\PYTHON_CODE _DESKTOP\finite-monkey\finite-monkey-engine\src\tree_sitter_parsing\graph_rag\graph_builder.py`

**Relationship:** Graph neighbor (CALLS/CALLED_BY)

**Context:**
- Defined In: `graph_builder.GraphBuilder`
- Calls: `os.path.exists`, `ValueError`, `os.path.isdir`, `KnowledgeGraphBuilder`, `kg_implementation.KnowledgeGraphBuilder.process_project`
- Called By: `pipeline.GraphRAGPipeline.index`

**Code:**
```python
def build(self, root_path: str) -> nx.DiGraph:
        """
        Build knowledge graph for the project.
        
        Args:
            root_path: Path to the project root directory.
            
        Returns:
            NetworkX DiGraph with:
            - File nodes: type="File", name, path
            - Class nodes: type="Class/Contract", name, file_path
            - Function nodes: type="Function", all function attributes
            - External nodes: type="External", name
            - DEFINES edges: containment relationships
            - CALLS edges: invocation relationships
            
        Raises:
            ValueError: If root_path does not exist or is not a directory.
        """
        if not os.path.exists(root_path):
            raise ValueError(f"Path does not exist: {root_path}")
        if not os.path.isdir(root_path):
            raise ValueError(f"Path is not a directory: {root_path}")
        
        self._builder = KnowledgeGraphBuilder()
        self._builder.process_project(root_path)
        return self._builder.graph
```

---

#### [Dep 3] chunk_enricher.ChunkEnricher.enrich_all

**File:** `E:\PYTHON_CODE _DESKTOP\finite-monkey\finite-monkey-engine\src\tree_sitter_parsing\graph_rag\chunk_enricher.py`

**Relationship:** Graph neighbor (CALLS/CALLED_BY)

**Context:**
- Defined In: `chunk_enricher.ChunkEnricher`
- Calls: `chunk_enricher.ChunkEnricher.enrich`, `chunk_enricher.ChunkEnricher._split_large_chunk`, `len`, `enriched_chunks.extend`, `enriched_chunks.append`, `logger.info`
- Called By: `pipeline.GraphRAGPipeline.index`

**Code:**
```python
def enrich_all(self, split_large_chunks: bool = True) -> List[EnrichedChunk]:
        """
        Create enriched chunks for all functions in master list.
        
        Args:
            split_large_chunks: If True, split chunks exceeding token_limit.
            
        Returns:
            List of EnrichedChunk objects. May include sub-chunks for large functions.
        """
        enriched_chunks = []
        split_count = 0
        
        for func in self._master_list:
            chunk = self.enrich(func)
            if chunk is not None:
                if split_large_chunks:
                    sub_chunks = self._split_large_chunk(chunk)
                    if len(sub_chunks) > 1:
                        split_count += 1
                    enriched_chunks.extend(sub_chunks)
                else:
                    enriched_chunks.append(chunk)
        
        logger.info(
            f"Created {len(enriched_chunks)} enriched chunks from {len(self._master_list)} functions "
            f"({split_count} large functions were split)"
        )
        
        return enriched_chunks
```

---

#### [Dep 4] models.EnrichedChunk.to_text

**File:** `E:\PYTHON_CODE _DESKTOP\finite-monkey\finite-monkey-engine\src\tree_sitter_parsing\graph_rag\models.py`

**Relationship:** Graph neighbor (CALLS/CALLED_BY)

**Context:**
- Defined In: `models.EnrichedChunk`
- Calls: `lines.append`, `'`, `'.join`, `"\n".join`
- Called By: `graph_expander.GraphExpander._get_content_for_node`, `pipeline.GraphRAGPipeline.index`, `vector_store.VectorStore.store`

**Code:**
```python
def to_text(self) -> str:
        """
        Serialize to structured text format for embedding.
        
        Format:
        
```

---

#### [Dep 5] embedding_service.EmbeddingService.embed_documents

**File:** `E:\PYTHON_CODE _DESKTOP\finite-monkey\finite-monkey-engine\src\tree_sitter_parsing\graph_rag\embedding_service.py`

**Relationship:** Graph neighbor (CALLS/CALLED_BY)

**Context:**
- Defined In: `embedding_service.EmbeddingService`
- Calls: `math.ceil`, `len`, `range`, `min`, `embedding_service.EmbeddingService._embed_batch`, `all_embeddings.extend`
- Called By: `pipeline.GraphRAGPipeline.index`

**Code:**
```python
def embed_documents(
        self, 
        texts: List[str], 
        batch_size: int = DEFAULT_BATCH_SIZE
    ) -> List[List[float]]:
        """
        Generate embeddings for documents with batch processing.
        
        Processes texts in batches to respect API rate limits.
        Each batch is retried with exponential backoff on rate limit errors.
        
        Args:
            texts: List of enriched chunk text blocks to embed
            batch_size: Number of texts per API call (default: 100)
            
        Returns:
            List of 2048-dimensional embedding vectors
            
        Raises:
            Exception: If embedding fails after all retries
        """
        if not texts:
            return []
        
        all_embeddings: List[List[float]] = []
        num_batches = math.ceil(len(texts) / batch_size)
        
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(texts))
            batch = texts[start_idx:end_idx]
            
            batch_embeddings = self._embed_batch(batch, input_type="document")
            all_embeddings.extend(batch_embeddings)
        
        return all_embeddings
```

---

#### [Dep 6] vector_store.VectorStore.store

**File:** `E:\PYTHON_CODE _DESKTOP\finite-monkey\finite-monkey-engine\src\tree_sitter_parsing\graph_rag\vector_store.py`

**Relationship:** Graph neighbor (CALLS/CALLED_BY)

**Context:**
- Defined In: `vector_store.VectorStore`
- Calls: `len`, `ValueError`, `zip`, `os.path.basename`, `data.append`, `models.EnrichedChunk.to_text`, `self._db.create_table`
- Called By: `pipeline.GraphRAGPipeline.index`

**Code:**
```python
def store(
        self, 
        chunks: List[EnrichedChunk], 
        embeddings: List[List[float]], 
        table_name: str = None
    ) -> None:
        """
        Store enriched chunks with their embeddings.
        
        Each record contains:
        - vector: embedding vector (2048 dimensions)
        - text: enriched chunk text (serialized via to_text())
        - node_id: graph node ID
        - filename: source file name
        
        Uses mode="overwrite" to replace existing data when creating the table.
        
        Args:
            chunks: List of EnrichedChunk objects to store
            embeddings: List of embedding vectors corresponding to chunks
            table_name: Optional table name (defaults to "code_embeddings")
            
        Raises:
            ValueError: If chunks and embeddings have different lengths
        """
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Number of chunks ({len(chunks)}) must match "
                f"number of embeddings ({len(embeddings)})"
            )
        
        if not chunks:
            return
        
        table_name = table_name or self._table_name
        
        # Prepare data for storage
        data = []
        for chunk, embedding in zip(chunks, embeddings):
            # Extract filename from file_path
            filename = os.path.basename(chunk.file_path) if chunk.file_path else ""
            
            data.append({
                "vector": embedding,
                "text": chunk.to_text(),
                "node_id": chunk.node_id,
                "filename": filename
            })
        
        # Create or overwrite table
        self._table = self._db.create_table(
            table_name, 
            data=data, 
            mode="overwrite"
        )
        self._table_name = table_name
```

---

#### [Dep 7] pipeline.GraphRAGPipeline._save_master_list

**File:** `E:\PYTHON_CODE _DESKTOP\finite-monkey\finite-monkey-engine\src\tree_sitter_parsing\graph_rag\pipeline.py`

**Relationship:** Graph neighbor (CALLS/CALLED_BY)

**Context:**
- Defined In: `pipeline.GraphRAGPipeline`
- Calls: `open`, `json.dump`
- Called By: `pipeline.GraphRAGPipeline.index`

**Code:**
```python
def _save_master_list(self) -> None:
        """Save master list to JSON file."""
        with open(self._master_list_path, 'w', encoding='utf-8') as f:
            json.dump(self._master_list, f, indent=2)
```

---

#### [Dep 8] pipeline.GraphRAGPipeline._save_enriched_chunks

**File:** `E:\PYTHON_CODE _DESKTOP\finite-monkey\finite-monkey-engine\src\tree_sitter_parsing\graph_rag\pipeline.py`

**Relationship:** Graph neighbor (CALLS/CALLED_BY)

**Context:**
- Defined In: `pipeline.GraphRAGPipeline`
- Calls: `chunks_data.append`, `open`, `json.dump`
- Called By: `pipeline.GraphRAGPipeline.index`

**Code:**
```python
def _save_enriched_chunks(self) -> None:
        """Save enriched chunks to JSON file."""
        chunks_data = []
        for chunk in self._enriched_chunks:
            chunks_data.append({
                'file_path': chunk.file_path,
                'function_name': chunk.function_name,
                'node_type': chunk.node_type,
                'node_id': chunk.node_id,
                'defined_in': chunk.defined_in,
                'calls': chunk.calls,
                'called_by': chunk.called_by,
                'code': chunk.code,
            })
        with open(self._enriched_chunks_path, 'w', encoding='utf-8') as f:
            json.dump(chunks_data, f, indent=2)
```

---

#### [Dep 9] pipeline.GraphRAGPipeline._build_indices

**File:** `E:\PYTHON_CODE _DESKTOP\finite-monkey\finite-monkey-engine\src\tree_sitter_parsing\graph_rag\pipeline.py`

**Relationship:** Graph neighbor (CALLS/CALLED_BY)

**Context:**
- Defined In: `pipeline.GraphRAGPipeline`
- Calls: `func.get`, `name.split`
- Called By: `pipeline.GraphRAGPipeline.index`, `pipeline.GraphRAGPipeline._ensure_index_loaded`

**Code:**
```python
def _build_indices(self) -> None:
        """Build lookup indices for master list and enriched chunks."""
        # Build master list index by node_id
        self._master_list_by_node_id = {}
        if self._master_list:
            for func in self._master_list:
                # Generate node_id from function data
                file_path = func.get('file_path', '') or func.get('absolute_file_path', '')
                name = func.get('name', '')
                func_name = name.split('.')[-1] if '.' in name else name
                line_number = func.get('line_number', 0) or func.get('start_line', 0)
                node_id = f"Function:{file_path}:{func_name}:{line_number}"
                self._master_list_by_node_id[node_id] = func
        
        # Build enriched chunks index by node_id
        self._enriched_chunks_by_node_id = {}
        if self._enriched_chunks:
            for chunk in self._enriched_chunks:
                self._enriched_chunks_by_node_id[chunk.node_id] = chunk
```

---

#### [Dep 10] pipeline.GraphRAGPipeline._ensure_index_loaded

**File:** `E:\PYTHON_CODE _DESKTOP\finite-monkey\finite-monkey-engine\src\tree_sitter_parsing\graph_rag\pipeline.py`

**Relationship:** Graph neighbor (CALLS/CALLED_BY)

**Context:**
- Defined In: `pipeline.GraphRAGPipeline`
- Calls: `os.path.exists`, `IndexNotFoundError`, `graph_builder.GraphBuilder.load_json`, `logger.info`, `pipeline.GraphRAGPipeline._load_master_list`, `pipeline.GraphRAGPipeline._load_enriched_chunks`, `pipeline.GraphRAGPipeline._build_indices`
- Called By: `pipeline.GraphRAGPipeline.query`

**Code:**
```python
def _ensure_index_loaded(self) -> None:
        """
        Ensure indexed data is loaded into memory.
        
        Loads graph, master list, and enriched chunks from persisted files
        if not already in memory.
        
        Raises:
            IndexNotFoundError: If no index files exist.
        """
        # Check if we need to load data
        if self._graph is not None and self._master_list is not None:
            return
        
        # Try to load from persisted files
        if not os.path.exists(self._graph_path):
            raise IndexNotFoundError(
                f"Graph file not found: {self._graph_path}. "
                "Please run index() first."
            )
        
        if not os.path.exists(self._master_list_path):
            raise IndexNotFoundError(
                f"Master list file not found: {self._master_list_path}. "
                "Please run index() first."
            )
        
        try:
            # Load graph
            self._graph = self.graph_builder.load_json(self._graph_path)
            logger.info(f"Loaded graph from {self._graph_path}")
            
            # Load master list
            self._load_master_list()
            logger.info(f"Loaded master list from {self._master_list_path}")
            
            # Load enriched chunks if available
            if os.path.exists(self._enriched_chunks_path):
                self._load_enriched_chunks()
                logger.info(f"Loaded enriched chunks from {self._enriched_chunks_path}")
            
            # Build indices
            self._build_indices()
            
        except Exception as e:
            raise IndexNotFoundError(
                f"Failed to load index data: {e}"
            )
```

---

#### [Dep 11] embedding_service.EmbeddingService.embed_query

**File:** `E:\PYTHON_CODE _DESKTOP\finite-monkey\finite-monkey-engine\src\tree_sitter_parsing\graph_rag\embedding_service.py`

**Relationship:** Graph neighbor (CALLS/CALLED_BY)

**Context:**
- Defined In: `embedding_service.EmbeddingService`
- Calls: `embedding_service.EmbeddingService._embed_batch`
- Called By: `pipeline.GraphRAGPipeline.query`

**Code:**
```python
def embed_query(self, query: str) -> List[float]:
        """
        Generate embedding for a search query.
        
        Args:
            query: Search query text
            
        Returns:
            2048-dimensional embedding vector
            
        Raises:
            Exception: If embedding fails after all retries
        """
        embeddings = self._embed_batch([query], input_type="query")
        return embeddings[0]
```

---

#### [Dep 12] vector_store.VectorStore.search

**File:** `E:\PYTHON_CODE _DESKTOP\finite-monkey\finite-monkey-engine\src\tree_sitter_parsing\graph_rag\vector_store.py`

**Relationship:** Graph neighbor (CALLS/CALLED_BY)

**Context:**
- Defined In: `vector_store.VectorStore`
- Calls: `self._db.open_table`, `table`
- Called By: `document_chunker.DocumentChunker._detect_chapter_markers`, `pipeline.GraphRAGPipeline.query`, `vector_store.VectorStore.get_by_node_id`

**Code:**
```python
def search(
        self, 
        query_embedding: List[float], 
        limit: int = 10,
        table_name: str = None
    ) -> List[SearchResult]:
        """
        Search for similar chunks by embedding vector.
        
        Returns results ordered by ascending distance (most similar first).
        
        Args:
            query_embedding: Query embedding vector (2048 dimensions)
            limit: Maximum number of results to return (default: 10)
            table_name: Optional table name to search (defaults to current table)
            
        Returns:
            List of SearchResult with text, node_id, distance, filename
            ordered by ascending distance (most similar first)
        """
        table_name = table_name or self._table_name
        
        try:
            table = self._db.open_table(table_name)
        except Exception:
            # Table doesn't exist
            return []
        
        # Perform vector search
        results = (
            table
            .search(query_embedding)
            .limit(limit)
            .to_list()
        )
        
        search_results = []
        for row in results:
            search_results.append(SearchResult(
                text=row.get("text", ""),
                node_id=row.get("node_id", ""),
                distance=row.get("_distance", 0.0),
                filename=row.get("filename", "")
            ))
        
        return search_results
```

---

#### [Dep 13] embedding_service.EmbeddingService.rerank

**File:** `E:\PYTHON_CODE _DESKTOP\finite-monkey\finite-monkey-engine\src\tree_sitter_parsing\graph_rag\embedding_service.py`

**Relationship:** Graph neighbor (CALLS/CALLED_BY)

**Context:**
- Defined In: `embedding_service.EmbeddingService`
- Calls: `results.append`, `RerankResult`
- Called By: `pipeline.GraphRAGPipeline.query`

**Code:**
```python
def rerank(
        self, 
        query: str, 
        documents: List[str], 
        top_k: int = 3
    ) -> List[RerankResult]:
        """
        Rerank documents using rerank-2.5 model.
        
        Args:
            query: Search query
            documents: List of document texts to rerank
            top_k: Number of top results to return (default: 3)
            
        Returns:
            List of RerankResult with index, relevance_score, document
            ordered by descending relevance score
            
        Raises:
            Exception: If reranking fails after all retries
        """
        if not documents:
            return []
        
        response = self._client.rerank(
            query=query,
            documents=documents,
            model=self.RERANK_MODEL,
            top_k=top_k
        )
        
        results = []
        for r in response.results:
            results.append(RerankResult(
                index=r.index,
                relevance_score=r.relevance_score,
                document=r.document
            ))
        
        return results
```

---

#### [Dep 14] context_assembler.ContextAssembler.assemble

**File:** `E:\PYTHON_CODE _DESKTOP\finite-monkey\finite-monkey-engine\src\tree_sitter_parsing\graph_rag\context_assembler.py`

**Relationship:** Graph neighbor (CALLS/CALLED_BY)

**Context:**
- Defined In: `context_assembler.ContextAssembler`
- Calls: `parts.append`, `enumerate`, `"\n".join`
- Called By: `context_assembler.ContextAssembler.assemble_minimal`, `pipeline.GraphRAGPipeline.query`

**Code:**
```python
def assemble(
        self,
        primary_results: List[str],
        secondary_results: List[str],
        query: str
    ) -> str:
        """
        Assemble the final prompt for LLM generation.
        
        Args:
            primary_results: List of enriched chunk texts from vector search.
            secondary_results: List of code texts from graph expansion.
            query: The user's query.
            
        Returns:
            Assembled prompt string with system instruction, primary matches,
            related code, and the user query.
        """
        parts = []
        
        # Add system instruction
        parts.append(self._system_instruction)
        parts.append("")
        
        # Add PRIMARY MATCHES section
        parts.append(self.PRIMARY_SECTION_HEADER)
        if primary_results:
            for i, result in enumerate(primary_results, 1):
                parts.append(f"[Chunk {i}]")
                parts.append(result)
                parts.append("")
        else:
            parts.append("No primary matches found.")
            parts.append("")
        
        # Add RELATED CODE section
        parts.append(self.SECONDARY_SECTION_HEADER)
        if secondary_results:
            for i, result in enumerate(secondary_results, 1):
                parts.append(f"[Dep {i}]")
                parts.append(result)
                parts.append("")
        else:
            parts.append("No related code found.")
            parts.append("")
        
        # Add user query at the end
        parts.append(f"Question: {query}")
        
        return "\n".join(parts)
```

---

#### [Dep 15] llm_service.LLMService.generate

**File:** `E:\PYTHON_CODE _DESKTOP\finite-monkey\finite-monkey-engine\src\tree_sitter_parsing\graph_rag\llm_service.py`

**Relationship:** Graph neighbor (CALLS/CALLED_BY)

**Context:**
- Defined In: `llm_service.LLMService`
- Calls: `llm_service.LLMService._generate_content_stream`, `chunks.append`, `"".join`, `llm_service.LLMService._generate_content`
- Called By: `pipeline.GraphRAGPipeline.query`

**Code:**
```python
def generate(self, prompt: str, stream: bool = True) -> str:
        """
        Generate response from Gemini.
        
        Args:
            prompt: Assembled context + query
            stream: Whether to stream output (default: True)
            
        Returns:
            Generated response text
            
        Raises:
            Exception: If generation fails after all retries
        """
        if stream:
            # Collect streamed chunks into full response
            chunks = []
            for chunk in self._generate_content_stream(prompt):
                chunks.append(chunk)
            return "".join(chunks)
        else:
            return self._generate_content(prompt)
```

---

#### [Dep 16] graph_expander.GraphExpander.expand

**File:** `E:\PYTHON_CODE _DESKTOP\finite-monkey\finite-monkey-engine\src\tree_sitter_parsing\graph_rag\graph_expander.py`

**Relationship:** Graph neighbor (CALLS/CALLED_BY)

**Context:**
- Defined In: `graph_expander.GraphExpander`
- Calls: `set`, `graph_expander.GraphExpander._get_neighbors`, `graph_expander.GraphExpander._is_external_node`, `logger.debug`, `graph_expander.GraphExpander._get_content_for_node`, `seen_neighbors.add`, `results.append`, `ExpandedResult`, `logger.info`, `len`
- Called By: `graph_expander.GraphExpander.expand_with_depth`, `pipeline.GraphRAGPipeline._expand_results`

**Code:**
```python
def expand(
        self,
        node_ids: List[str],
        exclude_ids: Optional[Set[str]] = None
    ) -> List[ExpandedResult]:
        """
        Get immediate neighbors of the given nodes via CALLS edges.
        
        Args:
            node_ids: Node IDs from primary search results.
            exclude_ids: Node IDs to exclude (already in results).
            
        Returns:
            List of ExpandedResult with node_id, relationship, content.
            - Skips External nodes (unresolved calls)
            - Returns source code from Master List or enriched chunks
        """
        if exclude_ids is None:
            exclude_ids = set()
        
        # Add input node_ids to exclude set to avoid duplicates
        all_exclude_ids = exclude_ids | set(node_ids)
        
        # Track seen neighbors to avoid duplicates in results
        seen_neighbors: Set[str] = set()
        results: List[ExpandedResult] = []
        
        for node_id in node_ids:
            neighbors = self._get_neighbors(node_id)
            
            for neighbor_id, relationship in neighbors:
                # Skip if already processed or in exclude list
                if neighbor_id in seen_neighbors or neighbor_id in all_exclude_ids:
                    continue
                
                # Skip External nodes (Requirement 7.5)
                if self._is_external_node(neighbor_id):
                    logger.debug(f"Skipping External node: {neighbor_id}")
                    continue
                
                # Get content for the neighbor
                content = self._get_content_for_node(neighbor_id)
                if content is None:
                    logger.debug(f"No content available for neighbor: {neighbor_id}")
                    continue
                
                # Mark as seen and add to results
                seen_neighbors.add(neighbor_id)
                results.append(ExpandedResult(
                    node_id=neighbor_id,
                    relationship=relationship,
                    content=content
                ))
        
        logger.info(
            f"Expanded {len(node_ids)} nodes to {len(results)} neighbors "
            f"(excluded {len(all_exclude_ids)} nodes)"
        )
        
        return results
```

---

## Raw Data

### Search Results Node IDs

```
Function:E:\PYTHON_CODE _DESKTOP\finite-monkey\finite-monkey-engine\src\tree_sitter_parsing\graph_rag\pipeline.py:pipeline.GraphRAGPipeline.graph:597
Function:E:\PYTHON_CODE _DESKTOP\finite-monkey\finite-monkey-engine\src\tree_sitter_parsing\graph_rag\pipeline.py:pipeline.GraphRAGPipeline.__init__:63
Function:E:\PYTHON_CODE _DESKTOP\finite-monkey\finite-monkey-engine\src\tree_sitter_parsing\graph_rag\pipeline.py:pipeline.GraphRAGPipeline.graph_builder:121
```

### Graph Statistics

- Total Graph Nodes: 451
- Total Graph Edges: 792
- Indexed Functions: 131
- Enriched Chunks: 131

---

*Report generated by Graph-Enhanced RAG System*
