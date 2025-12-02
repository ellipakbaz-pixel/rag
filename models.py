"""
Core data models for the Graph-Enhanced RAG System.

This module defines the data structures used throughout the pipeline:
- FunctionDict: TypedDict for function extraction results
- EnrichedChunk: Dataclass for enriched code chunks with serialization
- Result types: IndexResult, QueryResult, SearchResult, ExpandedResult
"""

from dataclasses import dataclass, field
from typing import List, Optional, TypedDict
import re


class FunctionDict(TypedDict, total=False):
    """
    TypedDict representing a function extracted from source code.
    
    This matches the output format from project_parser.py.
    """
    name: str                    # Qualified name like "Contract.method" or "file.function"
    content: str                 # Full source code
    file_path: str               # Absolute file path
    start_line: int              # Starting line number
    end_line: int                # Ending line number
    calls: List[str]             # Called function names
    contract_name: str           # Parent class/contract name
    visibility: str              # public/private/internal/external
    modifiers: List[str]         # Function modifiers
    parameters: List[str]        # Function parameters
    return_type: str             # Return type
    line_number: int             # Line number (same as start_line)
    type: str                    # "FunctionDefinition"
    signature: str               # Function signature
    relative_file_path: str      # Relative file path
    absolute_file_path: str      # Absolute file path


@dataclass
class EnrichedChunk:
    """
    Represents an enriched code chunk with metadata and context.
    
    Contains three sections:
    - METADATA: File, Function, Type fields
    - CONTEXT: Defined_In, Calls, Called_By fields
    - CODE: Complete function source code
    
    For large chunks that are split, chunk_id and parent_doc_id track
    the parent-child relationship between sub-chunks.
    """
    # Metadata
    file_path: str
    function_name: str
    node_type: str
    node_id: str
    
    # Context from graph
    defined_in: str = ""
    calls: List[str] = field(default_factory=list)
    called_by: List[str] = field(default_factory=list)
    
    # Code content
    code: str = ""
    
    # Chunk splitting fields (for large function handling)
    chunk_id: str = ""           # Unique ID for this chunk (empty for unsplit chunks)
    parent_doc_id: str = ""      # Parent chunk ID (empty for original/unsplit chunks)
    chunk_order: int = 0         # Order of this sub-chunk (0 for unsplit chunks)

    def to_text(self) -> str:
        """
        Serialize to structured text format for embedding.
        
        Format:
        [METADATA]
        File: /path/to/file.py
        Function: ClassName.method_name
        Type: Function
        Node_ID: Function:/path:name:line
        Chunk_ID: (optional, for split chunks)
        Parent_Doc_ID: (optional, for split chunks)
        Chunk_Order: (optional, for split chunks)
        
        [CONTEXT]
        Defined_In: ClassName
        Calls: helper_func, utils.process
        Called_By: main, test_method
        
        [CODE]
        def method_name(self, arg1, arg2):
            ...
        """
        lines = []
        
        # METADATA section
        lines.append("[METADATA]")
        lines.append(f"File: {self.file_path}")
        lines.append(f"Function: {self.function_name}")
        lines.append(f"Type: {self.node_type}")
        lines.append(f"Node_ID: {self.node_id}")
        # Include chunk fields only if they are set (for split chunks)
        if self.chunk_id:
            lines.append(f"Chunk_ID: {self.chunk_id}")
        if self.parent_doc_id:
            lines.append(f"Parent_Doc_ID: {self.parent_doc_id}")
        if self.chunk_order > 0 or self.parent_doc_id:
            lines.append(f"Chunk_Order: {self.chunk_order}")
        lines.append("")
        
        # CONTEXT section
        lines.append("[CONTEXT]")
        lines.append(f"Defined_In: {self.defined_in}")
        lines.append(f"Calls: {', '.join(self.calls) if self.calls else ''}")
        lines.append(f"Called_By: {', '.join(self.called_by) if self.called_by else ''}")
        lines.append("")
        
        # CODE section
        lines.append("[CODE]")
        lines.append(self.code)
        
        return "\n".join(lines)
    
    @classmethod
    def from_text(cls, text: str) -> 'EnrichedChunk':
        """
        Parse from structured text format.
        
        Reconstructs an EnrichedChunk from the serialized text format.
        Uses position-based parsing to handle special characters in code sections.
        """
        # Initialize default values
        file_path = ""
        function_name = ""
        node_type = ""
        node_id = ""
        defined_in = ""
        calls: List[str] = []
        called_by: List[str] = []
        code = ""
        chunk_id = ""
        parent_doc_id = ""
        chunk_order = 0
        
        # Find section boundaries using index-based search (not regex split)
        # This prevents issues when code contains [METADATA], [CONTEXT], or [CODE]
        metadata_start = text.find("[METADATA]")
        context_start = text.find("[CONTEXT]")
        code_start = text.find("[CODE]")
        
        # Parse METADATA section
        if metadata_start != -1:
            metadata_end = context_start if context_start != -1 else (code_start if code_start != -1 else len(text))
            metadata_section = text[metadata_start + len("[METADATA]"):metadata_end].strip()
            for line in metadata_section.split('\n'):
                line = line.strip()
                if line.startswith("File:"):
                    file_path = line[5:].strip()
                elif line.startswith("Function:"):
                    function_name = line[9:].strip()
                elif line.startswith("Type:"):
                    node_type = line[5:].strip()
                elif line.startswith("Node_ID:"):
                    node_id = line[8:].strip()
                elif line.startswith("Chunk_ID:"):
                    chunk_id = line[9:].strip()
                elif line.startswith("Parent_Doc_ID:"):
                    parent_doc_id = line[14:].strip()
                elif line.startswith("Chunk_Order:"):
                    try:
                        chunk_order = int(line[12:].strip())
                    except ValueError:
                        chunk_order = 0
        
        # Parse CONTEXT section
        if context_start != -1:
            context_end = code_start if code_start != -1 else len(text)
            context_section = text[context_start + len("[CONTEXT]"):context_end].strip()
            for line in context_section.split('\n'):
                line = line.strip()
                if line.startswith("Defined_In:"):
                    defined_in = line[11:].strip()
                elif line.startswith("Calls:"):
                    calls_str = line[6:].strip()
                    calls = [c.strip() for c in calls_str.split(',') if c.strip()]
                elif line.startswith("Called_By:"):
                    called_by_str = line[10:].strip()
                    called_by = [c.strip() for c in called_by_str.split(',') if c.strip()]
        
        # Parse CODE section - everything after [CODE] marker
        # This is the last section, so we take everything to the end
        if code_start != -1:
            code_section = text[code_start + len("[CODE]"):]
            # Remove only the leading newline that follows [CODE]
            if code_section.startswith('\n'):
                code = code_section[1:]
            elif code_section.startswith('\r\n'):
                code = code_section[2:]
            else:
                code = code_section
        
        return cls(
            file_path=file_path,
            function_name=function_name,
            node_type=node_type,
            node_id=node_id,
            defined_in=defined_in,
            calls=calls,
            called_by=called_by,
            code=code,
            chunk_id=chunk_id,
            parent_doc_id=parent_doc_id,
            chunk_order=chunk_order
        )


@dataclass
class SearchResult:
    """Result from vector similarity search."""
    text: str                    # Enriched chunk text
    node_id: str                 # Graph node ID
    distance: float              # Distance/similarity score
    filename: str = ""           # Source filename


@dataclass
class RerankResult:
    """Result from reranking operation."""
    index: int                   # Original index in candidates
    relevance_score: float       # Relevance score from reranker
    document: str                # Document text


@dataclass
class ExpandedResult:
    """Result from graph expansion."""
    node_id: str                 # Graph node ID
    relationship: str            # Relationship type (CALLS/CALLED_BY)
    content: str                 # Source code or enriched text


@dataclass
class IndexResult:
    """Result from indexing a project."""
    function_count: int          # Number of functions extracted
    node_count: int              # Number of graph nodes
    edge_count: int              # Number of graph edges
    chunk_count: int             # Number of enriched chunks
    embedding_count: int         # Number of embeddings generated
    success: bool = True         # Whether indexing succeeded
    message: str = ""            # Status message


@dataclass
class QueryResult:
    """Result from querying the indexed codebase."""
    query: str                   # Original query
    retrieved_count: int         # Number of retrieved chunks
    reranked_docs: List[str]     # Reranked document texts
    expanded_context: List[str]  # Graph-expanded context
    gemini_response: str         # LLM-generated response
    success: bool = True         # Whether query succeeded
    message: str = ""            # Status message
