#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Tree-sitter Based Call Tree Builder
Use real tree-sitter core functionality to replace simplified regex implementation

Note: Now using AdvancedCallTreeBuilder as the main implementation
The original simplified implementation is retained as an alternative
"""

import re
import os
import sys
from typing import List, Dict, Set, Tuple, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# Add path for import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Try to import advanced implementation
try:
    from .advanced_call_tree_builder import AdvancedCallTreeBuilder
    ADVANCED_BUILDER_AVAILABLE = True
    print("✅ Using advanced call tree builder")
except ImportError:
    try:
        # Try direct import
        sys.path.insert(0, os.path.dirname(__file__))
        from advanced_call_tree_builder import AdvancedCallTreeBuilder
        ADVANCED_BUILDER_AVAILABLE = True
        print("✅ Using advanced call tree builder")
    except ImportError:
        ADVANCED_BUILDER_AVAILABLE = False
        print("⚠️ Advanced call tree builder unavailable, using simplified implementation")


class SimplifiedCallTreeBuilder:
    """Simplified call tree builder (alternative implementation, using regex)"""
    
    def __init__(self):
        pass
    
    def analyze_function_relationships(self, functions_to_check: List[Dict]) -> Tuple[Dict, Dict]:
        """
        Analyze call relationships between functions
        Use more precise tree-sitter analysis to replace the original regex matching
        """
        # Build mapping from function names to function info and call relationship dictionaries
        func_map = {}
        relationships = {'upstream': {}, 'downstream': {}}
        
        # Build function mapping
        for idx, func in enumerate(functions_to_check):
            func_name = func['name']  # Use full function name (including contract name)
            func_map[func_name] = {
                'index': idx,
                'data': func
            }
        
        print(f"🔍 Analyzing call relationships of {len(functions_to_check)} functions...")
        
        # Analyze call relationships for each function
        for func in tqdm(functions_to_check, desc="Analyzing function call relationships"):
            func_name = func['name']  # Use full function name (including contract name)
            content = func.get('content', '').lower()
            
            if func_name not in relationships['upstream']:
                relationships['upstream'][func_name] = set()
            if func_name not in relationships['downstream']:
                relationships['downstream'][func_name] = set()
            
            # Use existing calls info (from tree-sitter analysis)
            if 'calls' in func and func['calls']:
                for called_func in func['calls']:
                    # Clean function name
                    clean_called_func = called_func.split('.')[-1] if '.' in called_func else called_func
                    
                    # Check if called function is in our function list and not self-reference
                    if clean_called_func in func_map and clean_called_func != func_name:
                        relationships['downstream'][func_name].add(clean_called_func)
                        if clean_called_func not in relationships['upstream']:
                            relationships['upstream'][clean_called_func] = set()
                        relationships['upstream'][clean_called_func].add(func_name)
            
            # Additional heuristic search (as alternative)
            for other_func in functions_to_check:
                if other_func == func:
                    continue
                    
                other_name = other_func['name']  # Use full function name (including contract name)
                other_content = other_func.get('content', '').lower()
                
                # Check if other functions call the current function (avoid self-reference)
                if other_name != func_name and self._is_function_called_in_content(func_name, other_content):
                    relationships['upstream'][func_name].add(other_name)
                    if other_name not in relationships['downstream']:
                        relationships['downstream'][other_name] = set()
                    relationships['downstream'][other_name].add(func_name)
                
                # Check if current function calls other functions (avoid self-reference)
                if other_name != func_name and self._is_function_called_in_content(other_name, content):
                    relationships['downstream'][func_name].add(other_name)
                    if other_name not in relationships['upstream']:
                        relationships['upstream'][other_name] = set()
                    relationships['upstream'][other_name].add(func_name)
        
        print(f"✅ Call relationship analysis completed")
        return relationships, func_map
    
    def _is_function_called_in_content(self, func_name: str, content: str) -> bool:
        """More precise function call detection"""
        # Multiple pattern matching
        patterns = [
            rf'\b{re.escape(func_name.lower())}\s*\(',  # Direct call
            rf'\.{re.escape(func_name.lower())}\s*\(',  # Member call
            rf'{re.escape(func_name.lower())}\s*\(',    # Simple call
        ]
        
        return any(re.search(pattern, content) for pattern in patterns)
    
    def build_call_tree(self, func_name: str, relationships: Dict, direction: str, func_map: Dict, visited: Set = None) -> Dict:
        """Build call tree"""
        if visited is None:
            visited = set()
        
        if func_name in visited:
            return None
        
        visited.add(func_name)
        
        # Get complete function information
        func_info = func_map.get(func_name, {'index': -1, 'data': None})
        
        node = {
            'name': func_name,
            'index': func_info['index'],
            'function_data': func_info['data'],  # Contains complete function information
            'children': []
        }
        
        # Get all direct calls in that direction
        related_funcs = relationships[direction].get(func_name, set())
        
        # Recursively build call tree for each related function
        for related_func in related_funcs:
            child_tree = self.build_call_tree(related_func, relationships, direction, func_map, visited.copy())
            if child_tree:
                node['children'].append(child_tree)
        
        return node
    
    def build_call_trees(self, functions_to_check: List[Dict], max_workers: int = 1) -> List[Dict]:
        """
        Build call trees for all functions
        Return format compatible with original CallTreeBuilder
        """
        if not functions_to_check:
            return []
        
        print(f"🌳 Starting to build call trees for {len(functions_to_check)} functions...")
        
        # Analyze function relationships
        relationships, func_map = self.analyze_function_relationships(functions_to_check)
        
        call_trees = []
        
        # Build upstream and downstream call trees for each function
        for func in tqdm(functions_to_check, desc="Building call trees"):
            func_name = func['name']  # Use full function name (including contract name)
            
            # Build upstream call tree (functions that call this function)
            upstream_tree = self.build_call_tree(func_name, relationships, 'upstream', func_map)
            
            # Build downstream call tree (functions called by this function)
            downstream_tree = self.build_call_tree(func_name, relationships, 'downstream', func_map)
            
            call_tree_info = {
                'function': func,
                'function_name': func_name,
                'upstream': upstream_tree,
                'downstream': downstream_tree,
                'upstream_count': len(relationships['upstream'].get(func_name, [])),
                'downstream_count': len(relationships['downstream'].get(func_name, [])),
                'relationships': relationships  # Save relationship data for later use
            }
            
            call_trees.append(call_tree_info)
        
        print(f"✅ Call tree construction completed, built {len(call_trees)} call trees")
        return call_trees
    
    def print_call_tree(self, node: Dict, level: int = 0, prefix: str = ''):
        """Print call tree"""
        if not node:
            return
            
        # Print basic information of current node
        func_data = node.get('function_data')
        if func_data:
            visibility = func_data.get('visibility', 'unknown')
            contract = func_data.get('contract_name', 'unknown')
            line_num = func_data.get('line_number', 'unknown')
            
            print(f"{prefix}{'└─' if level > 0 else ''}{node['name']} "
                  f"(index: {node['index']}, {visibility}, {contract}:{line_num})")
        else:
            print(f"{prefix}{'└─' if level > 0 else ''}{node['name']} (index: {node['index']})")
        
        # Recursively print child nodes
        children = node.get('children', [])
        for i, child in enumerate(children):
            is_last = i == len(children) - 1
            child_prefix = prefix + ('    ' if level > 0 else '')
            if not is_last:
                child_prefix += '├─'
            else:
                child_prefix += '└─'
            
            self.print_call_tree(child, level + 1, child_prefix)
    
    def get_call_tree_statistics(self, call_trees: List[Dict]) -> Dict:
        """Get call tree statistics"""
        stats = {
            'total_functions': len(call_trees),
            'functions_with_upstream': 0,
            'functions_with_downstream': 0,
            'max_upstream_count': 0,
            'max_downstream_count': 0,
            'isolated_functions': 0
        }
        
        for tree in call_trees:
            upstream_count = tree.get('upstream_count', 0)
            downstream_count = tree.get('downstream_count', 0)
            
            if upstream_count > 0:
                stats['functions_with_upstream'] += 1
                stats['max_upstream_count'] = max(stats['max_upstream_count'], upstream_count)
            
            if downstream_count > 0:
                stats['functions_with_downstream'] += 1
                stats['max_downstream_count'] = max(stats['max_downstream_count'], downstream_count)
            
            if upstream_count == 0 and downstream_count == 0:
                stats['isolated_functions'] += 1
        
        return stats
    
    def find_entry_points(self, call_trees: List[Dict]) -> List[Dict]:
        """Find entry point functions (functions with no upstream calls)"""
        entry_points = []
        for tree in call_trees:
            if tree.get('upstream_count', 0) == 0:
                entry_points.append(tree['function'])
        return entry_points
    
    def find_leaf_functions(self, call_trees: List[Dict]) -> List[Dict]:
        """Find leaf functions (functions with no downstream calls)"""
        leaf_functions = []
        for tree in call_trees:
            if tree.get('downstream_count', 0) == 0:
                leaf_functions.append(tree['function'])
        return leaf_functions


class TreeSitterCallTreeBuilder:
    """
    Smart call tree builder adapter
    Prefer advanced implementation (real tree-sitter), fallback to simplified implementation
    """
    
    def __init__(self):
        if ADVANCED_BUILDER_AVAILABLE:
            self.builder = AdvancedCallTreeBuilder()
            self.builder_type = "advanced"
        else:
            self.builder = SimplifiedCallTreeBuilder()
            self.builder_type = "simplified"
    
    def build_call_trees(self, functions_to_check: List[Dict], max_workers: int = 1) -> List[Dict]:
        """Build call trees (main interface)"""
        return self.builder.build_call_trees(functions_to_check, max_workers)
    
    def analyze_function_relationships(self, functions_to_check: List[Dict]) -> Tuple[Dict, Dict]:
        """Analyze function relationships"""
        return self.builder.analyze_function_relationships(functions_to_check)
    
    def build_call_tree(self, func_name: str, relationships: Dict, direction: str, func_map: Dict, visited: Set = None) -> Dict:
        """Build single call tree"""
        return self.builder.build_call_tree(func_name, relationships, direction, func_map, visited)
    
    def get_call_tree_statistics(self, call_trees: List[Dict]) -> Dict:
        """Get call tree statistics"""
        if hasattr(self.builder, 'get_call_tree_statistics'):
            return self.builder.get_call_tree_statistics(call_trees)
        else:
            # Alternative statistics for simplified implementation
            return self._basic_statistics(call_trees)
    
    def _basic_statistics(self, call_trees: List[Dict]) -> Dict:
        """Basic statistics"""
        stats = {
            'total_functions': len(call_trees),
            'functions_with_upstream': 0,
            'functions_with_downstream': 0,
            'max_upstream_count': 0,
            'max_downstream_count': 0,
            'isolated_functions': 0
        }
        
        for tree in call_trees:
            upstream_count = tree.get('upstream_count', 0)
            downstream_count = tree.get('downstream_count', 0)
            
            if upstream_count > 0:
                stats['functions_with_upstream'] += 1
                stats['max_upstream_count'] = max(stats['max_upstream_count'], upstream_count)
            
            if downstream_count > 0:
                stats['functions_with_downstream'] += 1
                stats['max_downstream_count'] = max(stats['max_downstream_count'], downstream_count)
            
            if upstream_count == 0 and downstream_count == 0:
                stats['isolated_functions'] += 1
        
        return stats
    
    def get_dependency_graph(self, target_function: str, functions_to_check: List[Dict], max_depth: int = 3) -> Dict:
        """Get function dependency graph (advanced feature)"""
        if hasattr(self.builder, 'get_dependency_graph'):
            return self.builder.get_dependency_graph(target_function, functions_to_check, max_depth)
        else:
            print("⚠️ Dependency graph analysis requires advanced implementation")
            return {'upstream_functions': {}, 'downstream_functions': {}}
    
    def get_builder_info(self) -> Dict:
        """Get builder information"""
        return {
            'type': self.builder_type,
            'advanced_available': ADVANCED_BUILDER_AVAILABLE,
            'features': {
                'basic_call_trees': True,
                'dependency_graph': hasattr(self.builder, 'get_dependency_graph'),
                'visualization': hasattr(self.builder, 'visualize_dependency_graph'),
                'mermaid_export': hasattr(self.builder, 'generate_dependency_mermaid')
            }
        }


# Backward compatible alias
CallTreeBuilder = TreeSitterCallTreeBuilder


if __name__ == '__main__':
    # Test code
    test_functions = [
        {
            'name': 'TestContract.transfer',
            'content': 'function transfer(address to, uint256 amount) public { _transfer(msg.sender, to, amount); }',
            'calls': ['_transfer'],
            'contract_name': 'TestContract',
            'visibility': 'public',
            'line_number': 10,
            'file_path': 'test_contract.sol'
        },
        {
            'name': 'TestContract._transfer',
            'content': 'function _transfer(address from, address to, uint256 amount) internal { emit Transfer(from, to, amount); }',
            'calls': ['emit'],
            'contract_name': 'TestContract',
            'visibility': 'internal',
            'line_number': 15,
            'file_path': 'test_contract.sol'
        }
    ]
    
    print("🧪 Testing smart call tree builder...")
    
    builder = TreeSitterCallTreeBuilder()
    builder_info = builder.get_builder_info()
    
    print(f"📊 Builder information:")
    print(f"  Type: {builder_info['type']}")
    print(f"  Advanced features available: {builder_info['advanced_available']}")
    print(f"  Supported features: {builder_info['features']}")
    
    call_trees = builder.build_call_trees(test_functions)
    
    print(f"\n✅ Built {len(call_trees)} call trees")
    for tree in call_trees:
        print(f"\n📊 Function: {tree['function_name']}")
        print(f"  Upstream call count: {tree['upstream_count']}")
        print(f"  Downstream call count: {tree['downstream_count']}")
        
        if 'analyzer_type' in tree:
            print(f"  Analyzer type: {tree['analyzer_type']}")
    
    # Test dependency graph functionality
    if builder_info['features']['dependency_graph']:
        print(f"\n🔍 Testing dependency graph analysis...")
        dep_graph = builder.get_dependency_graph('transfer', test_functions)
        print(f"  Upstream functions: {list(dep_graph['upstream_functions'].keys())}")
        print(f"  Downstream functions: {list(dep_graph['downstream_functions'].keys())}")
    
    print("\n🎉 Test completed")