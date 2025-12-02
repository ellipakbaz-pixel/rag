"""
Embedding Service for Graph-Enhanced RAG System.

This module provides embedding generation using Voyage AI voyage-code-3 model
with batch processing, retry logic, and reranking capabilities.
"""

import os
import time
import math
from typing import List, Callable, Any, Optional
from functools import wraps

import voyageai

try:
    from graph_rag.models import RerankResult
except ImportError:
    from models import RerankResult


# Retry configuration
MAX_RETRIES = 5
RETRY_DELAYS = [2, 4, 8, 16, 32]  # Exponential backoff delays in seconds


def retry_with_exponential_backoff(func: Callable) -> Callable:
    """
    Decorator to retry a function with exponential backoff on retryable errors.
    
    Handles rate limit errors (429), resource exhausted, timeout, and connection
    errors with increasing delays: 2s, 4s, 8s, 16s, 32s.
    
    Args:
        func: Function to wrap with retry logic
        
    Returns:
        Wrapped function with retry capability
    """
    @wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        last_exception = None
        
        for attempt in range(MAX_RETRIES):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                error_str = str(e).lower()
                
                # Check for retryable errors (rate limit, resource exhausted, timeout, connection)
                is_retryable = (
                    '429' in str(e) or
                    'rate' in error_str or
                    'limit' in error_str or
                    'resource_exhausted' in error_str or
                    'quota' in error_str or
                    'unavailable' in error_str or
                    'timeout' in error_str or
                    'connection' in error_str or
                    '503' in str(e) or
                    '500' in str(e)
                )
                
                if is_retryable and attempt < MAX_RETRIES - 1:
                    delay = RETRY_DELAYS[attempt]
                    time.sleep(delay)
                else:
                    raise
        
        # If we've exhausted all retries, raise the last exception
        if last_exception:
            raise last_exception
        
        return None
    
    return wrapper


class EmbeddingService:
    """
    Generates embeddings using Voyage AI voyage-code-3 model.
    
    Provides methods for:
    - Document embedding with batch processing
    - Query embedding
    - Reranking using rerank-2.5 model
    """
    
    # Model configuration
    MODEL_NAME = "voyage-code-3"
    RERANK_MODEL = "rerank-2.5"
    OUTPUT_DIMENSION = 2048
    DEFAULT_BATCH_SIZE = 100
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the embedding service with Voyage AI client.
        
        Args:
            api_key: Voyage AI API key. If not provided, uses VOYAGE_API_KEY env var.
        """
        if api_key:
            self._client = voyageai.Client(api_key=api_key)
        else:
            # voyageai.Client will use VOYAGE_API_KEY env var by default
            self._client = voyageai.Client()
    
    @retry_with_exponential_backoff
    def _embed_batch(self, texts: List[str], input_type: str) -> List[List[float]]:
        """
        Embed a single batch of texts with retry logic.
        
        Args:
            texts: List of texts to embed
            input_type: Either "document" or "query"
            
        Returns:
            List of embedding vectors
        """
        response = self._client.embed(
            texts,
            model=self.MODEL_NAME,
            input_type=input_type,
            output_dimension=self.OUTPUT_DIMENSION
        )
        return response.embeddings
    
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
    
    @retry_with_exponential_backoff
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
