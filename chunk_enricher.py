"""
ChunkEnricher for the Graph-Enhanced RAG System.

This module merges content from the Master List with graph context
to create enriched chunks for embedding.

Requirements: 3.1, 3.2, 3.3, 3.4, 10.1, 10.2, 10.3
"""

import logging
import uuid
from typing import Dict, List, Optional, Tuple

import networkx as nx

try:
    from graph_rag.models import EnrichedChunk, FunctionDict
    from graph_rag.kg_implementation import normalize_path
except ImportError:
    from models import EnrichedChunk, FunctionDict
    from kg_implementation import normalize_path

# Configure logging
logger = logging.getLogger(__name__)

# Default token limit for chunk splitting (approximate word count)
DEFAULT_TOKEN_LIMIT = 1000


class ChunkEnricher:
    """
    Creates enriched chunks by merging content with graph context.
    
    Takes a knowledge graph and master list of functions, and produces
    EnrichedChunk objects that contain:
    - METADATA: File, Function, Type fields
    - CONTEXT: Defined_In, Calls, Called_By fields
    - CODE: Complete function source code
    
    For large functions exceeding the token limit, the CODE section is
    split into smaller sub-chunks while preserving METADATA and CONTEXT.
    """
    
    def __init__(
        self, 
        graph: nx.DiGraph, 
        master_list: List[FunctionDict],
        token_limit: int = DEFAULT_TOKEN_LIMIT,
        chunk_overlap: int = 100
    ):
        """
        Initialize the ChunkEnricher.
        
        Args:
            graph: Knowledge graph (NetworkX DiGraph) with nodes and edges.
            master_list: List of function dictionaries from content extraction.
            token_limit: Maximum token count (word count) for CODE section before splitting.
            chunk_overlap: Number of overlapping words between sub-chunks.
        """
        self._graph = graph
        self._master_list = master_list
        self._token_limit = token_limit
        self._chunk_overlap = chunk_overlap
        
        # Build index for faster node lookup
        self._node_index: Dict[str, str] = self._build_node_index()
    
    def _build_node_index(self) -> Dict[str, str]:
        """
        Build an index mapping (file_path, name, line_number) to node IDs.
        
        Uses normalized paths (forward slashes) for cross-platform consistency.
        
        Returns:
            Dictionary mapping tuple keys to node IDs for fast lookup.
        """
        index = {}
        for node_id, data in self._graph.nodes(data=True):
            if data.get('type') == 'Function':
                file_path = data.get('file_path', '')
                label = data.get('label', '')  # Short name like "__init__"
                name = data.get('name', '')    # Full name like "module.Class.__init__"
                line_number = data.get('line_number', 0)
                
                # Use consistent path normalization
                normalized_path = normalize_path(file_path)
                
                # Create keys using normalized paths for consistent lookup
                # Key with full name
                if name:
                    index[(normalized_path, name, line_number)] = node_id
                
                # Key with label (short name)
                if label:
                    index[(normalized_path, label, line_number)] = node_id
                
        return index

    def _find_node_id(self, func: FunctionDict) -> Optional[str]:
        """
        Find matching graph node ID for a function.
        
        Matches based on file_path, name, and line_number using multiple
        path normalization strategies for cross-platform consistency.
        
        The graph node ID format is: Function:{file_path}:{func_name}:{line_number}
        
        Args:
            func: Function dictionary from the master list.
            
        Returns:
            Node ID if found, None otherwise.
        """
        file_path = func.get('file_path', '') or func.get('absolute_file_path', '')
        name = func.get('name', '')
        line_number = func.get('line_number', 0) or func.get('start_line', 0)
        
        # Normalize path for consistent lookup
        normalized_path = normalize_path(file_path)
        
        # Extract just the function name for label matching
        func_name_only = name.split('.')[-1] if '.' in name else name
        
        # Strategy 1: Direct node ID construction with normalized path (most efficient)
        normalized_node_id = f"Function:{normalized_path}:{name}:{line_number}"
        if normalized_node_id in self._graph:
            return normalized_node_id
        
        # Strategy 2: Try with just function name in node ID
        normalized_node_id_short = f"Function:{normalized_path}:{func_name_only}:{line_number}"
        if normalized_node_id_short in self._graph:
            return normalized_node_id_short
        
        # Strategy 3: Try index lookup with normalized path and full name
        key = (normalized_path, name, line_number)
        if key in self._node_index:
            return self._node_index[key]
        
        # Strategy 4: Try index lookup with normalized path and just function name
        key = (normalized_path, func_name_only, line_number)
        if key in self._node_index:
            return self._node_index[key]
        
        # Strategy 5: Try relative path variations
        # Extract filename and potential relative paths
        path_parts = normalized_path.replace('\\', '/').split('/')
        for i in range(len(path_parts)):
            partial_path = '/'.join(path_parts[i:])
            if not partial_path:
                continue
            
            # Try with full name
            key = (partial_path, name, line_number)
            if key in self._node_index:
                return self._node_index[key]
            
            # Try with short name
            key = (partial_path, func_name_only, line_number)
            if key in self._node_index:
                return self._node_index[key]
        
        # Strategy 6: Fallback - search through all function nodes with suffix matching
        for node_id, data in self._graph.nodes(data=True):
            if data.get('type') != 'Function':
                continue
            
            node_file = data.get('file_path', '')
            node_label = data.get('label', '')
            node_name = data.get('name', '')
            node_line = data.get('line_number', 0)
            
            # Normalize paths for comparison
            norm_node_file = normalize_path(node_file)
            
            # Match by exact file path and name/label and line number
            if (norm_node_file == normalized_path and 
                (node_name == name or node_label == name or node_label == func_name_only) and 
                node_line == line_number):
                return node_id
            
            # Match by file path suffix (handles different root paths)
            paths_match = (
                norm_node_file.endswith(normalized_path) or 
                normalized_path.endswith(norm_node_file) or
                self._paths_match_by_suffix(norm_node_file, normalized_path)
            )
            
            if paths_match:
                names_match = (
                    node_name == name or 
                    node_label == name or 
                    node_label == func_name_only or
                    node_name == func_name_only
                )
                if names_match and node_line == line_number:
                    return node_id
        
        logger.warning(
            f"Could not find graph node for function: {name} "
            f"at {file_path}:{line_number}"
        )
        return None
    
    def _paths_match_by_suffix(self, path1: str, path2: str) -> bool:
        """
        Check if two paths match by comparing their suffixes.
        
        This handles cases where paths have different root directories
        but refer to the same file.
        
        Args:
            path1: First path (normalized with forward slashes)
            path2: Second path (normalized with forward slashes)
            
        Returns:
            True if paths match by suffix, False otherwise.
        """
        if not path1 or not path2:
            return False
        
        # Split paths into components
        parts1 = path1.split('/')
        parts2 = path2.split('/')
        
        # Compare from the end (filename first)
        min_len = min(len(parts1), len(parts2))
        if min_len == 0:
            return False
        
        # At minimum, filenames must match
        if parts1[-1] != parts2[-1]:
            return False
        
        # Check how many trailing components match
        matching = 0
        for i in range(1, min_len + 1):
            if parts1[-i] == parts2[-i]:
                matching += 1
            else:
                break
        
        # Require at least filename match (1 component)
        # For better accuracy, require at least 2 matching components if both paths have 2+ parts
        if min_len >= 2:
            return matching >= 2
        return matching >= 1
    
    def _get_parent_scope(self, node_id: str) -> str:
        """
        Get parent scope from incoming DEFINES edge.
        
        Args:
            node_id: The node ID to find the parent for.
            
        Returns:
            Name of the parent scope (class/contract name or file name),
            or empty string if no parent found.
        """
        for source, target, data in self._graph.in_edges(node_id, data=True):
            if data.get('relation') == 'DEFINES':
                source_data = self._graph.nodes.get(source, {})
                source_type = source_data.get('type', '')
                
                if source_type == 'Class/Contract':
                    return source_data.get('name', '')
                elif source_type == 'File':
                    return source_data.get('name', '')
        
        return ""
    
    def _get_callees(self, node_id: str) -> List[str]:
        """
        Get functions called by this node (outgoing CALLS edges).
        
        Args:
            node_id: The node ID to find callees for.
            
        Returns:
            List of function names that this node calls.
        """
        callees = []
        for source, target, data in self._graph.out_edges(node_id, data=True):
            if data.get('relation') == 'CALLS':
                target_data = self._graph.nodes.get(target, {})
                target_type = target_data.get('type', '')
                
                if target_type == 'Function':
                    callees.append(target_data.get('label', ''))
                elif target_type == 'External':
                    # External nodes have format "External:{name}"
                    name = target_data.get('name', '')
                    if not name and ':' in target:
                        name = target.split(':', 1)[1]
                    callees.append(name)
        
        return callees
    
    def _get_callers(self, node_id: str) -> List[str]:
        """
        Get functions that call this node (incoming CALLS edges).
        
        Args:
            node_id: The node ID to find callers for.
            
        Returns:
            List of function names that call this node.
        """
        callers = []
        for source, target, data in self._graph.in_edges(node_id, data=True):
            if data.get('relation') == 'CALLS':
                source_data = self._graph.nodes.get(source, {})
                source_type = source_data.get('type', '')
                
                if source_type == 'Function':
                    callers.append(source_data.get('label', ''))
        
        return callers
    
    def enrich(self, func: FunctionDict) -> Optional[EnrichedChunk]:
        """
        Create an EnrichedChunk for a single function.
        
        Uses normalized paths for cross-platform consistency.
        
        Args:
            func: Function dictionary from the master list.
            
        Returns:
            EnrichedChunk if the function can be matched to a graph node,
            None otherwise.
        """
        node_id = self._find_node_id(func)
        
        if node_id is None:
            # Create a chunk without graph context for orphan functions
            return self._create_orphan_chunk(func)
        
        # Get context from graph
        defined_in = self._get_parent_scope(node_id)
        calls = self._get_callees(node_id)
        called_by = self._get_callers(node_id)
        
        # Get function metadata with normalized path
        file_path = func.get('file_path', '') or func.get('absolute_file_path', '')
        normalized_path = normalize_path(file_path)
        function_name = func.get('name', '')
        code = func.get('content', '')
        
        return EnrichedChunk(
            file_path=normalized_path,
            function_name=function_name,
            node_type='Function',
            node_id=node_id,
            defined_in=defined_in,
            calls=calls,
            called_by=called_by,
            code=code
        )
    
    def _create_orphan_chunk(self, func: FunctionDict) -> EnrichedChunk:
        """
        Create an EnrichedChunk for a function without graph context.
        
        This handles functions that couldn't be matched to graph nodes.
        Uses normalized paths for cross-platform consistency.
        Preserves all available metadata from the function dictionary.
        
        Args:
            func: Function dictionary from the master list.
            
        Returns:
            EnrichedChunk with available metadata (no called_by context).
        """
        # Extract file path from multiple possible keys
        file_path = func.get('file_path', '') or func.get('absolute_file_path', '') or func.get('relative_file_path', '')
        function_name = func.get('name', '')
        code = func.get('content', '')
        line_number = func.get('line_number', 0) or func.get('start_line', 0)
        
        # Normalize path for cross-platform consistency
        normalized_path = normalize_path(file_path)
        
        # Generate a synthetic node ID with normalized path
        node_id = f"Function:{normalized_path}:{function_name}:{line_number}"
        
        # Use contract_name as defined_in if available
        defined_in = func.get('contract_name', '')
        
        # Preserve calls from the function dict - ensure it's a list copy
        raw_calls = func.get('calls', None)
        if raw_calls is None:
            calls = []
        elif isinstance(raw_calls, list):
            calls = list(raw_calls)  # Make a copy to avoid mutation
        else:
            calls = []
        
        logger.debug(
            f"Creating orphan chunk for function '{function_name}' at {normalized_path}:{line_number} "
            f"with {len(calls)} calls, defined_in='{defined_in}'"
        )
        
        return EnrichedChunk(
            file_path=normalized_path,
            function_name=function_name,
            node_type='Function',
            node_id=node_id,
            defined_in=defined_in,
            calls=calls,
            called_by=[],  # No graph context available for orphan chunks
            code=code
        )
    
    def _estimate_token_count(self, text: str) -> int:
        """
        Estimate token count for a text (using word count as approximation).
        
        Args:
            text: The text to estimate tokens for.
            
        Returns:
            Estimated token count (word count).
        """
        return len(text.split())
    
    def _split_code_into_chunks(self, code: str) -> List[str]:
        """
        Split code into smaller chunks respecting the token limit.
        
        Uses word-based splitting with overlap to maintain context.
        Guarantees termination by ensuring step_size >= 1.
        
        Args:
            code: The code content to split.
            
        Returns:
            List of code chunks.
        """
        words = code.split()
        total_words = len(words)
        
        if total_words <= self._token_limit:
            logger.debug(f"Code has {total_words} words, no splitting needed (limit: {self._token_limit})")
            return [code]
        
        # Ensure we make forward progress: step size must be at least 1
        # and overlap cannot exceed token_limit - 1 to guarantee progress
        effective_overlap = min(self._chunk_overlap, self._token_limit - 1)
        step_size = max(1, self._token_limit - effective_overlap)
        
        logger.debug(
            f"Splitting code: {total_words} words, token_limit={self._token_limit}, "
            f"overlap={self._chunk_overlap}, effective_overlap={effective_overlap}, step_size={step_size}"
        )
        
        chunks = []
        start = 0
        iteration = 0
        max_iterations = total_words + 1  # Safety limit to prevent infinite loops
        
        while start < total_words and iteration < max_iterations:
            end = min(start + self._token_limit, total_words)
            chunk_words = words[start:end]
            chunk_text = ' '.join(chunk_words)
            chunks.append(chunk_text)
            
            logger.debug(f"Created chunk {iteration}: words[{start}:{end}] ({len(chunk_words)} words)")
            
            # Safety check: if we've reached the end, break before incrementing
            if end >= total_words:
                logger.debug(f"Reached end of words at iteration {iteration}")
                break
            
            # Move start forward by step_size (guaranteed to be at least 1)
            start += step_size
            iteration += 1
        
        if iteration >= max_iterations:
            logger.warning(f"Chunk splitting hit max iterations ({max_iterations}), possible logic error")
        
        logger.debug(f"Split complete: created {len(chunks)} chunks from {total_words} words")
        
        return chunks
    
    def _split_large_chunk(self, chunk: EnrichedChunk) -> List[EnrichedChunk]:
        """
        Split a large EnrichedChunk into smaller sub-chunks.
        
        Preserves METADATA and CONTEXT sections in each sub-chunk.
        Maintains parent-child relationship via chunk_id and parent_doc_id.
        
        Args:
            chunk: The EnrichedChunk to split.
            
        Returns:
            List of EnrichedChunk sub-chunks. Returns [chunk] if no split needed.
        """
        code_token_count = self._estimate_token_count(chunk.code)
        
        if code_token_count <= self._token_limit:
            return [chunk]
        
        # Generate parent document ID
        parent_id = f"{chunk.node_id}:{uuid.uuid4().hex[:8]}"
        
        # Split the code section
        code_chunks = self._split_code_into_chunks(chunk.code)
        
        sub_chunks = []
        for i, code_part in enumerate(code_chunks):
            # Generate unique chunk ID for each sub-chunk
            chunk_id = f"{parent_id}_chunk_{i}"
            
            sub_chunk = EnrichedChunk(
                file_path=chunk.file_path,
                function_name=chunk.function_name,
                node_type=chunk.node_type,
                node_id=chunk.node_id,
                defined_in=chunk.defined_in,
                calls=chunk.calls.copy(),
                called_by=chunk.called_by.copy(),
                code=code_part,
                chunk_id=chunk_id,
                parent_doc_id=parent_id,
                chunk_order=i
            )
            sub_chunks.append(sub_chunk)
        
        logger.info(
            f"Split large chunk '{chunk.function_name}' ({code_token_count} tokens) "
            f"into {len(sub_chunks)} sub-chunks"
        )
        
        return sub_chunks
    
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
