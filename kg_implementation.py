#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Knowledge Graph Implementation
Generates a Knowledge Graph from the parsed project data, outputting JSON format.
Leverages project_parser.py for detailed AST extraction to ensure all grammar, parameters,
and aspects of the program language are exactly extracted based on detected language.
"""

import os
import sys
import json
import logging
import networkx as nx
from pathlib import Path
from typing import List, Dict, Any, Optional, Set

# Configure logging for call resolution debugging
logger = logging.getLogger(__name__)

# Import project parser components
try:
    from .project_parser import parse_project, TreeSitterProjectFilter
except ImportError:
    # Try direct import if running as a script
    try:
        from project_parser import parse_project, TreeSitterProjectFilter
    except ImportError:
        print("Error: project_parser module not found.")
        sys.exit(1)

def normalize_path(path: str) -> str:
    """
    Normalize path separators to forward slashes for cross-platform consistency.
    
    This ensures node IDs are stable across Windows and Linux systems.
    
    Args:
        path: A file path that may contain backslashes or forward slashes
        
    Returns:
        Path with all backslashes converted to forward slashes
    """
    return path.replace('\\', '/')


class KnowledgeGraphBuilder:
    def __init__(self):
        self.graph = nx.DiGraph()
        self.graph.graph['created_at'] = str(os.times())
        self.file_nodes: Dict[str, str] = {}
        self.contract_nodes: Dict[str, str] = {}
        self.function_nodes: Dict[str, str] = {}
        
        # Additional indices for improved call resolution
        # Maps short function name -> list of (full_name, func_id, file_path, class_name)
        self._short_name_index: Dict[str, List[tuple]] = {}
        # Maps (file_path, short_name) -> func_id for file-local resolution
        self._file_func_index: Dict[tuple, str] = {}
        # Maps (class_name, method_name) -> func_id for class method resolution
        self._class_method_index: Dict[tuple, str] = {}

    def _sanitize_id(self, text: str) -> str:
        """Sanitize text for use as an ID"""
        import re
        return re.sub(r'[^a-zA-Z0-9_.:-]', '_', text)

    def process_project(self, root_path: str):
        """
        Process the project using project_parser and build the graph.
        """
        print(f"Processing project at {root_path}...")
        
        # Use the existing robust parser which handles multiple languages
        # (Solidity, Rust, C++, Move, Go, Python) and extracts detailed info
        parser_filter = TreeSitterProjectFilter()
        functions, _, _ = parse_project(root_path, parser_filter)
        
        print(f"Found {len(functions)} functions. Building graph...")

        # Pass 1: Create Nodes (Files, Contracts, Functions)
        for func in functions:
            self._add_function_to_graph(func)

        # Pass 2: Create Edges (Calls)
        for func in functions:
            self._add_calls_to_graph(func)

    def _add_function_to_graph(self, func: Dict[str, Any]):
        """Add function and its hierarchy to the graph"""
        # Normalize file path for cross-platform consistency
        file_path = normalize_path(func.get('file_path', 'unknown'))
        contract_name = func.get('contract_name')
        func_name = func.get('name')
        
        # 1. File Node
        if file_path not in self.file_nodes:
            file_id = f"File:{file_path}"
            self.graph.add_node(file_id, type="File", name=os.path.basename(file_path), path=file_path)
            self.file_nodes[file_path] = file_id
        else:
            file_id = self.file_nodes[file_path]

        # 2. Contract/Class Node (if applicable)
        parent_id = file_id
        if contract_name:
            # Create a unique ID for the contract using normalized path
            contract_id = f"Contract:{file_path}:{contract_name}"
            if contract_id not in self.contract_nodes:
                self.graph.add_node(contract_id, type="Class/Contract", name=contract_name, file_path=file_path)
                self.graph.add_edge(file_id, contract_id, relation="DEFINES")
                self.contract_nodes[contract_id] = contract_id
            
            parent_id = contract_id

        # 3. Function Node
        # Use a unique ID for the function with normalized path
        func_id = f"Function:{file_path}:{func_name}:{func.get('line_number')}"
        
        # Store all function attributes in the node to ensure all parameters/grammar aspects are captured
        node_attrs = func.copy()
        node_attrs['type'] = 'Function'
        node_attrs['label'] = func_name
        # Store normalized file_path in node attributes
        node_attrs['file_path'] = file_path
        
        self.graph.add_node(func_id, **node_attrs)
        self.graph.add_edge(parent_id, func_id, relation="DEFINES")
        
        # Map function name to ID for call resolution
        # We map the fully qualified name (as returned by parser) to the ID
        self.function_nodes[func_name] = func_id
        
        # Build additional indices for improved call resolution
        # Extract short name (last part after dots)
        short_name = func_name.split('.')[-1] if '.' in func_name else func_name
        
        # Index by short name (using normalized file_path)
        if short_name not in self._short_name_index:
            self._short_name_index[short_name] = []
        self._short_name_index[short_name].append((func_name, func_id, file_path, contract_name))
        
        # Index by (file_path, short_name) for file-local resolution (using normalized path)
        self._file_func_index[(file_path, short_name)] = func_id
        
        # Index by (class_name, method_name) for class method resolution
        if contract_name:
            self._class_method_index[(contract_name, short_name)] = func_id
            # Also index with module prefix if present (e.g., "module.ClassName")
            if '.' in func_name:
                parts = func_name.split('.')
                if len(parts) >= 2:
                    # Try "module.Class" as class name
                    module_class = '.'.join(parts[:-1])
                    self._class_method_index[(module_class, short_name)] = func_id

    def _resolve_call_target(self, call_target_name: str, caller_file_path: str, caller_class: Optional[str]) -> Optional[str]:
        """
        Resolve a call target name to a function node ID.
        
        Tries multiple resolution strategies:
        1. Exact match with full qualified name
        2. self.method -> method in same class
        3. Class.method -> method in that class
        4. Short name in same file
        5. Short name in same class
        6. Unique short name across project
        
        Args:
            call_target_name: The call target as extracted by parser
            caller_file_path: File path of the calling function
            caller_class: Class/contract name of the calling function (if any)
            
        Returns:
            Function node ID if resolved, None otherwise
        """
        # Strategy 1: Exact match with full qualified name
        if call_target_name in self.function_nodes:
            return self.function_nodes[call_target_name]
        
        # Clean up the call target name
        clean_name = call_target_name.strip()
        
        # Strategy 2: Handle self.method calls (Python style)
        if clean_name.startswith('self.'):
            method_name = clean_name[5:]  # Remove 'self.'
            # Look up in the caller's class
            if caller_class:
                # Direct class name match
                target_id = self._class_method_index.get((caller_class, method_name))
                if target_id:
                    return target_id
                
                # Try with module prefix variations
                # Handle cases like "module.ClassName" vs "ClassName"
                for (class_name, meth_name), func_id in self._class_method_index.items():
                    if meth_name == method_name:
                        # Check if class names match (with or without module prefix)
                        if class_name == caller_class:
                            return func_id
                        # caller_class might be "module.Class" and class_name is "Class"
                        if caller_class.endswith('.' + class_name):
                            return func_id
                        # class_name might be "module.Class" and caller_class is "Class"
                        if class_name.endswith('.' + caller_class):
                            return func_id
                        # Both might have different module prefixes but same class name
                        caller_short = caller_class.split('.')[-1] if '.' in caller_class else caller_class
                        class_short = class_name.split('.')[-1] if '.' in class_name else class_name
                        if caller_short == class_short:
                            return func_id
        
        # Strategy 3: Handle Class.method or module.function calls
        if '.' in clean_name:
            parts = clean_name.split('.')
            if len(parts) == 2:
                class_or_module, method_name = parts
                # Try class method lookup
                target_id = self._class_method_index.get((class_or_module, method_name))
                if target_id:
                    return target_id
                # Try with full module.class prefix
                for (class_name, meth_name), func_id in self._class_method_index.items():
                    if meth_name == method_name and (
                        class_name == class_or_module or 
                        class_name.endswith('.' + class_or_module)
                    ):
                        return func_id
        
        # Extract short name for remaining strategies
        short_name = clean_name.split('.')[-1] if '.' in clean_name else clean_name
        
        # Strategy 4: Short name in same class (check class first when caller has class context)
        # This takes precedence over same-file when both apply
        if caller_class:
            target_id = self._class_method_index.get((caller_class, short_name))
            if target_id:
                return target_id
        
        # Strategy 5: Short name in same file
        target_id = self._file_func_index.get((caller_file_path, short_name))
        if target_id:
            return target_id
        
        # Strategy 6: Unique short name across project
        if short_name in self._short_name_index:
            matches = self._short_name_index[short_name]
            if len(matches) == 1:
                # Unique match - use it
                return matches[0][1]  # func_id
            elif len(matches) > 1:
                # Multiple matches - prefer same file, then same class
                for full_name, func_id, fpath, cname in matches:
                    if fpath == caller_file_path:
                        return func_id
                for full_name, func_id, fpath, cname in matches:
                    if cname and caller_class and cname == caller_class:
                        return func_id
                
                # Ambiguous: multiple matches exist but none in same file/class
                # Log the ambiguity and return None to create External node
                logger.debug(
                    f"Ambiguous call resolution for '{call_target_name}': "
                    f"found {len(matches)} matches but none in same file/class. "
                    f"Matches: {[m[0] for m in matches]}. "
                    f"Caller file: {caller_file_path}, caller class: {caller_class}"
                )
                return None
        
        return None

    def _add_calls_to_graph(self, func: Dict[str, Any]):
        """Add call edges to the graph"""
        func_name = func.get('name')
        # Normalize file path for cross-platform consistency
        file_path = normalize_path(func.get('file_path', ''))
        line_number = func.get('line_number')
        contract_name = func.get('contract_name')
        source_id = f"Function:{file_path}:{func_name}:{line_number}"
        
        calls = func.get('calls', [])
        for call_target_name in calls:
            # Try to resolve the call target using multiple strategies
            target_id = self._resolve_call_target(call_target_name, file_path, contract_name)
            
            if target_id:
                # Internal call to a known function
                # Avoid self-loops
                if target_id != source_id:
                    self.graph.add_edge(source_id, target_id, relation="CALLS")
            else:
                # External or unresolved call
                # Create a proxy node for the external call
                external_id = f"External:{call_target_name}"
                if external_id not in self.graph:
                    self.graph.add_node(external_id, type="External", name=call_target_name)
                
                self.graph.add_edge(source_id, external_id, relation="CALLS")

    def export_json(self, output_file: str):
        """
        Export the graph to a Node-Link JSON format.
        """
        print(f"Exporting JSON to {output_file}...")
        data = nx.node_link_data(self.graph)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate Knowledge Graph from Codebase')
    parser.add_argument('input_dir', help='Input directory to parse')
    parser.add_argument('--json', default='knowledge_graph.json', help='Output JSON file')
    
    args = parser.parse_args()
    
    if not os.path.isdir(args.input_dir):
        print(f"Error: {args.input_dir} is not a directory.")
        sys.exit(1)
        
    builder = KnowledgeGraphBuilder()
    builder.process_project(args.input_dir)
    
    builder.export_json(args.json)
    
    print("Done!")

if __name__ == "__main__":
    main()


