"""
Exception hierarchy for the Graph-Enhanced RAG System.

This module defines custom exceptions for pipeline error handling,
with each exception including stage information for debugging.

Requirements: 9.3
"""


class PipelineError(Exception):
    """
    Base exception for pipeline errors.
    
    All pipeline errors include the stage name where the error occurred,
    making it easy to identify which step failed.
    """
    
    def __init__(self, stage: str, message: str, original_error: Exception = None):
        """
        Initialize a PipelineError.
        
        Args:
            stage: The pipeline stage where the error occurred
                   (e.g., "extraction", "graph_building", "embedding")
            message: Human-readable error message
            original_error: The original exception that caused this error
        """
        self.stage = stage
        self.message = message
        self.original_error = original_error
        super().__init__(f"[{stage}] {message}")


class ExtractionError(PipelineError):
    """Error during content extraction (Phase 1, Step 1)."""
    
    def __init__(self, message: str, original_error: Exception = None):
        super().__init__("extraction", message, original_error)


class GraphBuildError(PipelineError):
    """Error during graph building (Phase 1, Step 2)."""
    
    def __init__(self, message: str, original_error: Exception = None):
        super().__init__("graph_building", message, original_error)


class EnrichmentError(PipelineError):
    """Error during chunk enrichment (Phase 1, Step 3)."""
    
    def __init__(self, message: str, original_error: Exception = None):
        super().__init__("enrichment", message, original_error)


class EmbeddingError(PipelineError):
    """Error during embedding generation (Phase 2, Step 4)."""
    
    def __init__(self, message: str, original_error: Exception = None):
        super().__init__("embedding", message, original_error)


class StorageError(PipelineError):
    """Error during storage operations (Phase 2, Step 5)."""
    
    def __init__(self, message: str, original_error: Exception = None):
        super().__init__("storage", message, original_error)


class RetrievalError(PipelineError):
    """Error during retrieval operations (Phase 3, Steps 6-7)."""
    
    def __init__(self, message: str, original_error: Exception = None):
        super().__init__("retrieval", message, original_error)


class GenerationError(PipelineError):
    """Error during LLM generation (Phase 3, Step 8)."""
    
    def __init__(self, message: str, original_error: Exception = None):
        super().__init__("generation", message, original_error)


class IndexNotFoundError(PipelineError):
    """Error when trying to query without an existing index."""
    
    def __init__(self, message: str = "No index found. Please run index() first."):
        super().__init__("retrieval", message, None)
