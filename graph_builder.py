"""
GraphBuilder wrapper for kg_implementation.py.

This module wraps the KnowledgeGraphBuilder from kg_implementation.py
to provide a clean interface for building knowledge graphs from codebases.
"""

import json
import os
from typing import Optional

import networkx as nx

# Import KnowledgeGraphBuilder from local kg_implementation module
try:
    from .kg_implementation import KnowledgeGraphBuilder
except ImportError:
    # Fallback for direct script execution or parent directory import
    try:
        from kg_implementation import KnowledgeGraphBuilder
    except ImportError:
        from ..kg_implementation import KnowledgeGraphBuilder


class GraphBuilder:
    """Builds a knowledge graph from a codebase.
    
    Wraps the KnowledgeGraphBuilder from kg_implementation.py to provide
    a clean interface for building, exporting, and loading knowledge graphs.
    """
    
    def __init__(self):
        """Initialize the GraphBuilder."""
        self._builder: Optional[KnowledgeGraphBuilder] = None
    
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
    
    def export_json(self, graph: nx.DiGraph, output_path: str) -> None:
        """
        Export graph to JSON using node-link format.
        
        Args:
            graph: NetworkX DiGraph to export.
            output_path: Path to the output JSON file.
        """
        # Ensure parent directory exists
        parent_dir = os.path.dirname(output_path)
        if parent_dir and not os.path.exists(parent_dir):
            os.makedirs(parent_dir, exist_ok=True)
        
        # Use edges="links" for compatibility with current NetworkX behavior
        data = nx.node_link_data(graph, edges="links")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
    
    def load_json(self, input_path: str) -> nx.DiGraph:
        """
        Load graph from JSON file.
        
        Args:
            input_path: Path to the JSON file to load.
            
        Returns:
            NetworkX DiGraph reconstructed from the JSON file.
            
        Raises:
            FileNotFoundError: If the input file does not exist.
            json.JSONDecodeError: If the file contains invalid JSON.
        """
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Graph file not found: {input_path}")
        
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Use edges="links" for compatibility with current NetworkX behavior
        return nx.node_link_graph(data, edges="links")
