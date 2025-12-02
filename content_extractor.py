"""
Content Extractor wrapper for the Graph-Enhanced RAG System.

This module wraps project_parser.py to extract code content and metadata
from a codebase, creating a Master List of code objects for embedding.

Requirements: 1.1, 1.2, 1.3, 1.4, 1.5
"""

import logging
import os
import sys
from typing import List, Optional

# Import from local module (within graph_rag package)
try:
    from .project_parser import parse_project, TreeSitterProjectFilter
except ImportError:
    # Fallback for direct script execution or parent directory import
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from project_parser import parse_project, TreeSitterProjectFilter

try:
    from graph_rag.models import FunctionDict
except ImportError:
    from models import FunctionDict

# Configure logging
logger = logging.getLogger(__name__)


class ContentExtractor:
    """
    Extracts functions and classes from a codebase using tree-sitter.
    
    Wraps project_parser.py to provide a clean interface for content extraction.
    Supports Solidity, Rust, C++, Move, Go, and Python files.
    """
    
    def __init__(self, project_filter: Optional[TreeSitterProjectFilter] = None):
        """
        Initialize the ContentExtractor with an optional custom filter.
        
        Args:
            project_filter: Optional TreeSitterProjectFilter for customizing
                           which files and functions to include/exclude.
                           If None, uses default filter.
        """
        # Always provide a filter instance to avoid bug in parse_project
        # where it tries to instantiate TreeSitterProjectFilter with wrong args
        self._filter = project_filter if project_filter is not None else TreeSitterProjectFilter()
    
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
    
    def extract_all(self, root_path: str) -> List[FunctionDict]:
        """
        Extract all functions from the project without filtering.
        
        This returns all extracted functions before any filtering is applied.
        Useful for testing or when you need the complete function list.
        
        Args:
            root_path: Path to the project root directory.
            
        Returns:
            List of all function dictionaries (unfiltered).
        """
        if not os.path.exists(root_path):
            raise FileNotFoundError(f"Project path does not exist: {root_path}")
        
        if not os.path.isdir(root_path):
            raise ValueError(f"Project path is not a directory: {root_path}")
        
        try:
            functions, _, _ = parse_project(
                root_path,
                project_filter=self._filter
            )
            
            logger.info(f"Extracted {len(functions)} total functions from {root_path}")
            return functions
            
        except Exception as e:
            logger.error(f"Error extracting content from {root_path}: {e}")
            raise
