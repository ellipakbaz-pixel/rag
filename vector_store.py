"""
Vector Store for Graph-Enhanced RAG System.

This module provides vector storage and retrieval using LanceDB,
supporting storage of enriched chunks with embeddings and metadata.
"""

import os
from typing import List, Optional
from pathlib import Path

import lancedb
import pyarrow as pa

try:
    from graph_rag.models import EnrichedChunk, SearchResult
except ImportError:
    from models import EnrichedChunk, SearchResult


def _escape_node_id(node_id: str) -> str:
    """
    Escape special characters in node_id for use in SQL-like filters.
    
    Handles:
    - Single quotes: ' -> ''
    - Backslashes: \\ -> \\\\
    
    Args:
        node_id: The node ID to escape
        
    Returns:
        Escaped node ID safe for SQL filter expressions
    """
    # First escape backslashes (must be done first to avoid double-escaping)
    escaped = node_id.replace("\\", "\\\\")
    # Then escape single quotes
    escaped = escaped.replace("'", "''")
    return escaped


class VectorStore:
    """
    Manages vector storage in LanceDB.
    
    Provides methods for:
    - Storing enriched chunks with their embeddings
    - Searching for similar chunks by embedding vector
    - Direct lookup by node ID
    """
    
    DEFAULT_TABLE_NAME = "code_embeddings"
    VECTOR_DIMENSION = 2048
    
    def __init__(self, db_path: str = "lancedb_store"):
        """
        Initialize the vector store with LanceDB connection.
        
        Args:
            db_path: Path to the LanceDB database directory.
                     Defaults to "lancedb_store" in current directory.
        """
        self._db_path = db_path
        self._db = lancedb.connect(db_path)
        self._table_name = self.DEFAULT_TABLE_NAME
        self._table = None

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
        
        Validates that node_id and filename metadata are correctly set for each chunk.
        
        Args:
            chunks: List of EnrichedChunk objects to store
            embeddings: List of embedding vectors corresponding to chunks
            table_name: Optional table name (defaults to "code_embeddings")
            
        Raises:
            ValueError: If chunks and embeddings have different lengths
            ValueError: If any chunk has an empty node_id
        """
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Number of chunks ({len(chunks)}) must match "
                f"number of embeddings ({len(embeddings)})"
            )
        
        if not chunks:
            return
        
        table_name = table_name or self._table_name
        
        # Prepare data for storage with validation
        data = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            # Validate node_id is present
            if not chunk.node_id:
                raise ValueError(
                    f"Chunk at index {i} has empty node_id. "
                    f"Function: {chunk.function_name}, File: {chunk.file_path}"
                )
            
            # Extract filename from file_path
            filename = os.path.basename(chunk.file_path) if chunk.file_path else ""
            
            # Validate filename could be extracted (warn but don't fail)
            if not filename and chunk.file_path:
                import logging
                logging.warning(
                    f"Could not extract filename from file_path: {chunk.file_path}"
                )
            
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
    
    def get_by_node_id(
        self, 
        node_id: str, 
        table_name: str = None
    ) -> Optional[str]:
        """
        Retrieve enriched chunk text by node ID.
        
        Special characters in node_id (quotes, backslashes) are properly escaped
        to prevent SQL injection and query errors.
        
        Args:
            node_id: Graph node ID to look up
            table_name: Optional table name (defaults to current table)
            
        Returns:
            Enriched chunk text if found, None otherwise
        """
        table_name = table_name or self._table_name
        
        try:
            table = self._db.open_table(table_name)
        except Exception:
            # Table doesn't exist
            return None
        
        # Escape special characters in node_id for safe SQL filter
        escaped_node_id = _escape_node_id(node_id)
        
        # Query by node_id
        results = table.search().where(f"node_id = '{escaped_node_id}'").limit(1).to_list()
        
        if results:
            return results[0].get("text")
        
        return None
    
    def get_table(self, table_name: str = None):
        """
        Get the LanceDB table object.
        
        Args:
            table_name: Optional table name (defaults to current table)
            
        Returns:
            LanceDB table object or None if not found
        """
        table_name = table_name or self._table_name
        
        try:
            return self._db.open_table(table_name)
        except Exception:
            return None
    
    @property
    def db_path(self) -> str:
        """Return the database path."""
        return self._db_path
    
    @property
    def table_name(self) -> str:
        """Return the current table name."""
        return self._table_name
