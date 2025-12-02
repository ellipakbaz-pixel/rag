"""
GraphExpander for the Graph-Enhanced RAG System.

This module expands search results using knowledge graph traversal
to discover related code not found by vector search.

Requirements: 7.1, 7.2, 7.3, 7.4, 7.5
"""

import json
import logging
from typing import Dict, List, Optional, Set, Tuple

import networkx as nx

try:
    from graph_rag.models import EnrichedChunk, ExpandedResult, FunctionDict
except ImportError:
    from models import EnrichedChunk, ExpandedResult, FunctionDict

# Configure logging
logger = logging.getLogger(__name__)


class GraphExpander:
    """
    Expands search results using knowledge graph traversal.
    
    Given primary search result node IDs, this component:
    1. Loads the persisted graph
    2. Queries for immediate neighbors via CALLS edges (both directions)
    3. Retrieves source code for neighbors from the Master List
    4. Filters out nodes already in primary results
    5. Skips External nodes during expansion
    """
    
    def __init__(
        self,
        graph: nx.DiGraph,
        master_list: Dict[str, FunctionDict],
        enriched_chunks: Optional[Dict[str, EnrichedChunk]] = None
    ):
        """
        Initialize the GraphExpander.
        
        Args:
            graph: Knowledge graph (loaded from JSON).
            master_list: Dict mapping node_id to function dict (for content retrieval).
            enriched_chunks: Optional dict mapping node_id to EnrichedChunk.
        """
        self._graph = graph
        self._master_list = master_list
        self._enriched_chunks = enriched_chunks or {}
    
    @classmethod
    def from_json(
        cls,
        graph_path: str,
        master_list: Dict[str, FunctionDict],
        enriched_chunks: Optional[Dict[str, EnrichedChunk]] = None
    ) -> 'GraphExpander':
        """
        Create a GraphExpander by loading graph from JSON file.
        
        Args:
            graph_path: Path to the graph JSON file.
            master_list: Dict mapping node_id to function dict.
            enriched_chunks: Optional dict mapping node_id to EnrichedChunk.
            
        Returns:
            GraphExpander instance with loaded graph.
            
        Raises:
            FileNotFoundError: If the graph file does not exist.
        """
        with open(graph_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        graph = nx.node_link_graph(data, edges="links")
        return cls(graph, master_list, enriched_chunks)

    def _get_neighbors(self, node_id: str) -> List[Tuple[str, str]]:
        """
        Get (neighbor_id, relationship) tuples for a node.
        
        Returns both:
        - Outgoing CALLS edges (functions this node calls) -> relationship="CALLS"
        - Incoming CALLS edges (functions that call this node) -> relationship="CALLED_BY"
        
        Self-loops (where a node calls itself) are excluded.
        
        Args:
            node_id: The node ID to find neighbors for.
            
        Returns:
            List of (neighbor_id, relationship) tuples.
        """
        neighbors = []
        
        # Check if node exists in graph
        if node_id not in self._graph:
            logger.warning(f"Node not found in graph: {node_id}")
            return neighbors
        
        # Get outgoing CALLS edges (functions this node calls)
        # Skip self-loops (Requirement 4.2)
        for source, target, data in self._graph.out_edges(node_id, data=True):
            if data.get('relation') == 'CALLS' and target != node_id:
                neighbors.append((target, 'CALLS'))
        
        # Get incoming CALLS edges (functions that call this node)
        # Skip self-loops (Requirement 4.2)
        for source, target, data in self._graph.in_edges(node_id, data=True):
            if data.get('relation') == 'CALLS' and source != node_id:
                neighbors.append((source, 'CALLED_BY'))
        
        return neighbors
    
    def _get_content_for_node(self, node_id: str) -> Optional[str]:
        """
        Retrieve source code for a node from Master List or enriched chunks.
        
        Args:
            node_id: The node ID to get content for.
            
        Returns:
            Source code string if found, None otherwise.
        """
        # First try enriched chunks (preferred - has full context)
        if node_id in self._enriched_chunks:
            chunk = self._enriched_chunks[node_id]
            if isinstance(chunk, EnrichedChunk):
                return chunk.to_text()
            return str(chunk)
        
        # Then try master list
        if node_id in self._master_list:
            func_dict = self._master_list[node_id]
            return func_dict.get('content', '')
        
        # Try to get content from graph node attributes
        if node_id in self._graph:
            node_data = self._graph.nodes[node_id]
            content = node_data.get('content', '')
            if content:
                return content
        
        logger.debug(f"No content found for node: {node_id}")
        return None
    
    def _is_external_node(self, node_id: str) -> bool:
        """
        Check if a node is an External node (unresolved call).
        
        Args:
            node_id: The node ID to check.
            
        Returns:
            True if the node is External, False otherwise.
        """
        if node_id not in self._graph:
            return False
        
        node_data = self._graph.nodes[node_id]
        return node_data.get('type') == 'External'
    
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
    
    def expand_with_depth(
        self,
        node_ids: List[str],
        depth: int = 1,
        exclude_ids: Optional[Set[str]] = None
    ) -> List[ExpandedResult]:
        """
        Expand nodes to a specified depth (multi-hop expansion).
        
        Tracks visited nodes across all expansion levels to prevent cycles
        (Requirement 4.1).
        
        Args:
            node_ids: Starting node IDs.
            depth: Number of hops to expand (default 1 for immediate neighbors).
            exclude_ids: Node IDs to exclude.
            
        Returns:
            List of ExpandedResult from all expansion levels.
        """
        if depth < 1:
            return []
        
        if exclude_ids is None:
            exclude_ids = set()
        
        # Track all visited nodes to prevent cycles (Requirement 4.1)
        # Start with exclude_ids and the starting node_ids
        visited: Set[str] = exclude_ids | set(node_ids)
        all_results: List[ExpandedResult] = []
        current_level_ids = list(node_ids)
        
        for level in range(depth):
            # Pass the visited set as exclude_ids to prevent revisiting nodes
            level_results = self.expand(current_level_ids, visited)
            
            if not level_results:
                logger.debug(f"No more results at depth level {level + 1}")
                break
            
            all_results.extend(level_results)
            
            # Update visited set with newly found nodes BEFORE next iteration
            # This ensures cycle prevention across all levels
            current_level_ids = [r.node_id for r in level_results]
            visited.update(current_level_ids)
            
            logger.debug(
                f"Depth level {level + 1}: found {len(level_results)} nodes, "
                f"total visited: {len(visited)}"
            )
        
        return all_results
