"""
GraphRAGPipeline - Unified interface for Graph-Enhanced RAG System.

This module provides the main pipeline for indexing codebases and
querying them using a combination of vector search and graph expansion.

Requirements: 9.1, 9.2, 9.3, 9.4
"""

import json
import logging
import os
from typing import Dict, List, Optional, Set

import networkx as nx

# Try package imports first, fall back to local imports
try:
    from graph_rag.models import (
        EnrichedChunk,
        ExpandedResult,
        FunctionDict,
        IndexResult,
        QueryResult,
        SearchResult,
    )
    from graph_rag.content_extractor import ContentExtractor
    from graph_rag.graph_builder import GraphBuilder
    from graph_rag.chunk_enricher import ChunkEnricher
    from graph_rag.embedding_service import EmbeddingService
    from graph_rag.vector_store import VectorStore
    from graph_rag.graph_expander import GraphExpander
    from graph_rag.context_assembler import ContextAssembler
    from graph_rag.llm_service import LLMService
    from graph_rag.exceptions import (
        PipelineError,
        ExtractionError,
        GraphBuildError,
        EnrichmentError,
        EmbeddingError,
        StorageError,
        RetrievalError,
        GenerationError,
        IndexNotFoundError,
    )
except ImportError:
    from models import (
        EnrichedChunk,
        ExpandedResult,
        FunctionDict,
        IndexResult,
        QueryResult,
        SearchResult,
    )
    from content_extractor import ContentExtractor
    from graph_builder import GraphBuilder
    from chunk_enricher import ChunkEnricher
    from embedding_service import EmbeddingService
    from vector_store import VectorStore
    from graph_expander import GraphExpander
    from context_assembler import ContextAssembler
    from llm_service import LLMService
    from exceptions import (
        PipelineError,
        ExtractionError,
        GraphBuildError,
        EnrichmentError,
        EmbeddingError,
        StorageError,
        RetrievalError,
        GenerationError,
        IndexNotFoundError,
    )

# Configure logging
logger = logging.getLogger(__name__)


class GraphRAGPipeline:
    """
    Main pipeline for Graph-Enhanced RAG.
    
    Provides a unified interface for:
    - Indexing: Extract content, build graph, enrich chunks, embed, and store
    - Querying: Search, rerank, expand, assemble context, and generate response
    """
    
    # Default file names for persisted data
    DEFAULT_GRAPH_FILENAME = "knowledge_graph.json"
    DEFAULT_MASTER_LIST_FILENAME = "master_list.json"
    DEFAULT_ENRICHED_CHUNKS_FILENAME = "enriched_chunks.json"

    def __init__(
        self,
        db_path: str = "lancedb_store",
        graph_path: str = None,
        voyage_api_key: str = None,
        gemini_api_key: str = None,
        data_dir: str = None,
    ):
        """
        Initialize the GraphRAGPipeline.
        
        Args:
            db_path: Path to LanceDB database directory.
            graph_path: Path to knowledge graph JSON file. If None, uses
                       data_dir/knowledge_graph.json.
            voyage_api_key: Voyage AI API key. If None, uses VOYAGE_API_KEY env var.
            gemini_api_key: Google API key. If None, uses GOOGLE_API_KEY env var.
            data_dir: Directory for storing pipeline data files (graph, master list).
                     If None, uses current directory.
        """
        self._db_path = db_path
        self._data_dir = data_dir or "."
        self._graph_path = graph_path or os.path.join(
            self._data_dir, self.DEFAULT_GRAPH_FILENAME
        )
        self._master_list_path = os.path.join(
            self._data_dir, self.DEFAULT_MASTER_LIST_FILENAME
        )
        self._enriched_chunks_path = os.path.join(
            self._data_dir, self.DEFAULT_ENRICHED_CHUNKS_FILENAME
        )
        
        self._voyage_api_key = voyage_api_key
        self._gemini_api_key = gemini_api_key
        
        # Lazy-initialized components
        self._content_extractor: Optional[ContentExtractor] = None
        self._graph_builder: Optional[GraphBuilder] = None
        self._embedding_service: Optional[EmbeddingService] = None
        self._vector_store: Optional[VectorStore] = None
        self._llm_service: Optional[LLMService] = None
        self._context_assembler: Optional[ContextAssembler] = None
        
        # Cached data from indexing
        self._graph: Optional[nx.DiGraph] = None
        self._master_list: Optional[List[FunctionDict]] = None
        self._enriched_chunks: Optional[List[EnrichedChunk]] = None
        self._master_list_by_node_id: Optional[Dict[str, FunctionDict]] = None
        self._enriched_chunks_by_node_id: Optional[Dict[str, EnrichedChunk]] = None
    
    @property
    def content_extractor(self) -> ContentExtractor:
        """Get or create ContentExtractor instance."""
        if self._content_extractor is None:
            self._content_extractor = ContentExtractor()
        return self._content_extractor
    
    @property
    def graph_builder(self) -> GraphBuilder:
        """Get or create GraphBuilder instance."""
        if self._graph_builder is None:
            self._graph_builder = GraphBuilder()
        return self._graph_builder
    
    @property
    def embedding_service(self) -> EmbeddingService:
        """Get or create EmbeddingService instance."""
        if self._embedding_service is None:
            self._embedding_service = EmbeddingService(api_key=self._voyage_api_key)
        return self._embedding_service
    
    @property
    def vector_store(self) -> VectorStore:
        """Get or create VectorStore instance."""
        if self._vector_store is None:
            self._vector_store = VectorStore(db_path=self._db_path)
        return self._vector_store
    
    @property
    def llm_service(self) -> LLMService:
        """Get or create LLMService instance."""
        if self._llm_service is None:
            self._llm_service = LLMService(api_key=self._gemini_api_key)
        return self._llm_service
    
    @property
    def context_assembler(self) -> ContextAssembler:
        """Get or create ContextAssembler instance."""
        if self._context_assembler is None:
            self._context_assembler = ContextAssembler()
        return self._context_assembler

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

    def _ensure_index_loaded(self) -> None:
        """
        Ensure indexed data is loaded into memory.
        
        Loads graph, master list, and enriched chunks from persisted files
        if not already in memory.
        
        Raises:
            IndexNotFoundError: If no index files exist or required files are missing.
        """
        # Check if we need to load data
        if self._graph is not None and self._master_list is not None:
            return
        
        # Check all required files exist before attempting to load
        required_files = [
            (self._graph_path, "knowledge graph"),
            (self._master_list_path, "master list"),
        ]
        
        missing_files = []
        for file_path, description in required_files:
            if not os.path.exists(file_path):
                missing_files.append(f"  - {description}: {file_path}")
        
        if missing_files:
            missing_list = "\n".join(missing_files)
            raise IndexNotFoundError(
                f"Required index files not found:\n{missing_list}\n"
                "Please run index() first to create the index."
            )
        
        try:
            # Load graph
            self._graph = self.graph_builder.load_json(self._graph_path)
            logger.info(f"Loaded graph from {self._graph_path}")
            
            # Load master list
            self._load_master_list()
            logger.info(f"Loaded master list from {self._master_list_path}")
            
            # Load enriched chunks if available (optional file)
            if os.path.exists(self._enriched_chunks_path):
                self._load_enriched_chunks()
                logger.info(f"Loaded enriched chunks from {self._enriched_chunks_path}")
            else:
                logger.warning(
                    f"Enriched chunks file not found: {self._enriched_chunks_path}. "
                    "Graph expansion may be limited."
                )
            
            # Build indices
            self._build_indices()
            
        except IndexNotFoundError:
            raise
        except Exception as e:
            raise IndexNotFoundError(
                f"Failed to load index data: {e}. "
                "The index files may be corrupted. Try running index() again."
            )
    
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
    
    def _save_master_list(self) -> None:
        """Save master list to JSON file."""
        with open(self._master_list_path, 'w', encoding='utf-8') as f:
            json.dump(self._master_list, f, indent=2)
    
    def _load_master_list(self) -> None:
        """Load master list from JSON file."""
        with open(self._master_list_path, 'r', encoding='utf-8') as f:
            self._master_list = json.load(f)
    
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
    
    def _load_enriched_chunks(self) -> None:
        """Load enriched chunks from JSON file."""
        with open(self._enriched_chunks_path, 'r', encoding='utf-8') as f:
            chunks_data = json.load(f)
        
        self._enriched_chunks = []
        for data in chunks_data:
            chunk = EnrichedChunk(
                file_path=data.get('file_path', ''),
                function_name=data.get('function_name', ''),
                node_type=data.get('node_type', ''),
                node_id=data.get('node_id', ''),
                defined_in=data.get('defined_in', ''),
                calls=data.get('calls', []),
                called_by=data.get('called_by', []),
                code=data.get('code', ''),
            )
            self._enriched_chunks.append(chunk)

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
    
    def is_indexed(self) -> bool:
        """
        Check if an index exists.
        
        Returns:
            True if index files exist, False otherwise.
        """
        return (
            os.path.exists(self._graph_path) and
            os.path.exists(self._master_list_path)
        )
    
    def clear_index(self) -> None:
        """
        Clear all indexed data.
        
        Removes persisted files and clears in-memory caches.
        """
        # Clear in-memory data
        self._graph = None
        self._master_list = None
        self._enriched_chunks = None
        self._master_list_by_node_id = None
        self._enriched_chunks_by_node_id = None
        
        # Remove persisted files
        for path in [
            self._graph_path,
            self._master_list_path,
            self._enriched_chunks_path
        ]:
            if os.path.exists(path):
                os.remove(path)
                logger.info(f"Removed {path}")
        
        # Note: LanceDB data is not cleared here to avoid accidental data loss
        # Use vector_store directly if needed
    
    @property
    def graph(self) -> Optional[nx.DiGraph]:
        """Get the knowledge graph (if loaded)."""
        return self._graph
    
    @property
    def master_list(self) -> Optional[List[FunctionDict]]:
        """Get the master list of functions (if loaded)."""
        return self._master_list
    
    @property
    def enriched_chunks(self) -> Optional[List[EnrichedChunk]]:
        """Get the enriched chunks (if loaded)."""
        return self._enriched_chunks
