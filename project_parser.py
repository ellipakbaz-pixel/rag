#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Tree-sitter Based Project Parser
Use tree-sitter instead of ANTLR for project parsing
"""

import os
import re
import warnings
from pathlib import Path
from typing import List, Dict, Any, Optional
import sys

# Use the installed tree-sitter package
# Suppress deprecation warnings from tree-sitter language bindings that still use int API
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message="int argument support is deprecated", category=DeprecationWarning)
    from tree_sitter import Language, Parser, Node
import tree_sitter_solidity
import tree_sitter_rust
import tree_sitter_cpp
import tree_sitter_move
import tree_sitter_go
import tree_sitter_python

# Import document chunker
try:
    from .document_chunker import chunk_project_files
    from .chunk_config import ChunkConfigManager
except ImportError:
    # If relative import fails, try direct import
    from document_chunker import chunk_project_files
    from chunk_config import ChunkConfigManager

# Create language objects
# Suppress deprecation warnings from tree-sitter-solidity and tree-sitter-move
# which still use the old int API
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message="int argument support is deprecated", category=DeprecationWarning)
    LANGUAGES = {
        'solidity': Language(tree_sitter_solidity.language()),
        'rust': Language(tree_sitter_rust.language()),
        'cpp': Language(tree_sitter_cpp.language()),
        'move': Language(tree_sitter_move.language()),
        'go': Language(tree_sitter_go.language()),
        'python': Language(tree_sitter_python.language())
    }

TREE_SITTER_AVAILABLE = True
print("[SUCCESS] Tree-sitter parser loaded, supports six languages")


class LanguageType:
    SOLIDITY = 'solidity'
    RUST = 'rust'
    CPP = 'cpp'
    MOVE = 'move'
    GO = 'go'
    PYTHON = 'python'


class TreeSitterProjectFilter(object):
    """Tree-sitter based project filter"""
    
    def __init__(self):
        pass

    def filter_file(self, path, filename):
        """Filter files"""
        # Check file extensions - supports six languages: Solidity, Rust, C++, Move, Go, Python
        valid_extensions = ('.sol', '.rs', '.move', '.c', '.cpp', '.cxx', '.cc', '.C', '.h', '.hpp', '.hxx', '.go', '.py')
        if not any(filename.endswith(ext) for ext in valid_extensions) or filename.endswith('.t.sol'):
            return True
        
        return False

    def filter_contract(self, function):
        """Filter contract functions"""
        # Supported languages that are not filtered: rust, move, cpp, python
        # Check file extension or function name characteristics to identify language type
        file_path = function.get('file_path', '')
        if file_path:
            if file_path.endswith('.rs'):  # Rust file
                return False
            if file_path.endswith('.move'):  # Move file
                return False
            if file_path.endswith(('.c', '.cpp', '.cxx', '.cc', '.C', '.h', '.hpp', '.hxx')):  # C++ file
                return False
            if file_path.endswith('.py'):  # Python file
                return False
        
        # Compatible with old naming conventions
        if '_rust' in function["name"]:
            return False
        if '_move' in function["name"]:
            return False
        if '_cpp' in function["name"]:
            return False
        if '_python' in function["name"]:
            return False
        
        # Filter constructor and receive functions
        if function.get('visibility') in ['constructor', 'receive', 'fallback']:
            return True
        
        return False

    def should_check_function_code_if_statevar_assign(self, function_code, contract_code):
        """Check if state variable assignment checking should be performed in function code"""
        return True

    def check_function_code_if_statevar_assign(self, function_code, contract_code):
        """Check state variable assignments in function code"""
        return self.should_check_function_code_if_statevar_assign(function_code, contract_code)


def _detect_language_from_path(file_path: Path) -> Optional[str]:
    """Detect language type based on file path"""
    suffix = file_path.suffix.lower()
    
    if suffix == '.sol':
        return 'solidity'
    elif suffix == '.rs':
        return 'rust'
    elif suffix in ['.cpp', '.cc', '.cxx', '.c', '.h', '.hpp', '.hxx']:
        return 'cpp'
    elif suffix == '.move':
        return 'move'
    elif suffix == '.go':
        return 'go'
    elif suffix == '.py':
        return 'python'
    return None


def _extract_functions_from_node(node: Node, source_code: bytes, language: str, file_path: str) -> List[Dict]:
    """Extract function information from AST nodes"""
    functions = []
    
    def traverse_node(node, contract_name=""):
        if node.type == 'function_definition' and language == 'solidity':
            # Solidity function definition
            func_info = _parse_solidity_function(node, source_code, contract_name, file_path)
            if func_info:
                functions.append(func_info)
        
        elif node.type == 'function_item' and language == 'rust':
            # Rust function definition
            func_info = _parse_rust_function(node, source_code, file_path)
            if func_info:
                functions.append(func_info)
        
        elif node.type == 'function_definition' and language == 'cpp':
            # C++ function definition
            func_info = _parse_cpp_function(node, source_code, file_path)
            if func_info:
                functions.append(func_info)
        
        elif node.type == 'function_decl' and language == 'move':
            # Move function definition
            func_info = _parse_move_function(node, source_code, file_path)
            if func_info:
                functions.append(func_info)
        
        elif node.type == 'function_declaration' and language == 'go':
            # Go function definition
            func_info = _parse_go_function(node, source_code, file_path)
            if func_info:
                functions.append(func_info)
        
        elif node.type == 'function_definition' and language == 'python':
            # Python function definition
            func_info = _parse_python_function(node, source_code, file_path)
            if func_info:
                functions.append(func_info)
        
        elif node.type == 'contract_declaration' and language == 'solidity':
            # Solidity contract declaration
            contract_name = _get_node_text(node.child_by_field_name('name'), source_code)
        
        elif node.type == 'class_definition' and language == 'python':
            # Python class definition - update contract_name for methods
            class_name_node = node.child_by_field_name('name')
            if class_name_node:
                contract_name = _get_node_text(class_name_node, source_code)
        
        # Recursively traverse child nodes
        for child in node.children:
            traverse_node(child, contract_name)
    
    traverse_node(node)
    return functions


def _get_node_text(node: Node, source_code: bytes) -> str:
    """Get the source code text corresponding to the node"""
    if node is None:
        return ""
    return source_code[node.start_byte:node.end_byte].decode('utf-8')


def _extract_function_calls(node: Node, source_code: bytes) -> List[str]:
    """Extract function calls from function nodes"""
    calls = []
    
    def traverse_for_calls(node):
        # Function call node types in Move language
        if node.type in ['call_expr', 'receiver_call']:
            called_func = _get_function_call_name(node, source_code)
            if called_func:
                calls.append(called_func)
        # Function call node types in other languages (Go, Rust, Solidity, C++)
        elif node.type == 'call_expression':
            called_func = _get_function_call_name(node, source_code)
            if called_func:
                calls.append(called_func)
        # Python function call
        elif node.type == 'call':
            called_func = _get_function_call_name(node, source_code)
            if called_func:
                calls.append(called_func)
        
        # Recursively traverse child nodes
        for child in node.children:
            traverse_for_calls(child)
    
    traverse_for_calls(node)
    return calls


def _get_function_call_name(call_node: Node, source_code: bytes) -> Optional[str]:
    """Extract the called function name from call_expression nodes"""
    try:
        # Python language: call node
        if call_node.type == 'call':
            # Python function call: has a 'function' field
            func_node = call_node.child_by_field_name('function')
            if func_node:
                if func_node.type == 'identifier':
                    # Simple function call: func()
                    return _get_node_text(func_node, source_code).strip()
                elif func_node.type == 'attribute':
                    # Method call: obj.method() or module.func()
                    attr_text = _get_node_text(func_node, source_code).strip()
                    if '.' in attr_text:
                        # Return the full path for better tracing
                        return attr_text
                    return attr_text
                else:
                    # Fallback: get the full text
                    return _get_node_text(func_node, source_code).strip()
        
        # Move language: call_expr and receiver_call
        elif call_node.type == 'call_expr':
            # Move function call: name_access_chain + call_args
            for child in call_node.children:
                if child.type == 'name_access_chain':
                    chain_text = _get_node_text(child, source_code).strip()
                    # Handle module calls: module::function -> module.function
                    if '::' in chain_text:
                        parts = chain_text.split('::')
                        if len(parts) >= 2:
                            module_name = parts[-2]
                            func_name = parts[-1]
                            return f"{module_name}.{func_name}"
                    # Simple function call
                    return chain_text
        elif call_node.type == 'receiver_call':
            # Move method call: obj.method()
            for child in call_node.children:
                if child.type == 'identifier':
                    # Return method name
                    return _get_node_text(child, source_code).strip()
        
        # Traverse child nodes of call_expression to find function names (Rust/Solidity)
        for child in call_node.children:
            # Rust: scoped_identifier (e.g. instructions::borrow)
            if child.type == 'scoped_identifier':
                scoped_text = _get_node_text(child, source_code).strip()
                # Convert Rust module calls to our naming format
                # instructions::withdraw -> withdraw.withdraw
                if '::' in scoped_text:
                    parts = scoped_text.split('::')
                    if len(parts) >= 2:
                        module_name = parts[-2]  # instructions
                        func_name = parts[-1]    # withdraw
                        # For instructions module, function name is the file name
                        if module_name == 'instructions':
                            return f"{func_name}.{func_name}"
                        else:
                            return f"{module_name}.{func_name}"
                return scoped_text  # Keep original name as alternative
            # Rust: identifier (e.g. simple_function_call)
            elif child.type == 'identifier':
                return _get_node_text(child, source_code).strip()
            # Rust: field_expression (e.g. obj.method)
            elif child.type == 'field_expression':
                field_text = _get_node_text(child, source_code).strip()
                return field_text
            # Solidity: expression
            elif child.type == 'expression':
                # Find the actual function name in expression
                for expr_child in child.children:
                    if expr_child.type == 'identifier':
                        # Simple function call, e.g.: functionName()
                        return _get_node_text(expr_child, source_code).strip()
                    elif expr_child.type == 'member_expression':
                        # Member function call, e.g.: obj.method()
                        member_text = _get_node_text(expr_child, source_code).strip()
                        if '.' in member_text:
                            return member_text.split('.')[-1]  # Return method name
                        return member_text
            # Solidity: member_expression (as alternative)
            elif child.type == 'member_expression':
                member_text = _get_node_text(child, source_code).strip()
                if '.' in member_text:
                    return member_text.split('.')[-1]
                return member_text
        return None
    except Exception:
        return None


def _parse_solidity_function(node: Node, source_code: bytes, contract_name: str, file_path: str) -> Optional[Dict]:
    """Parse Solidity function"""
    try:
        name_node = node.child_by_field_name('name')
        if not name_node:
            return None
        
        func_name = _get_node_text(name_node, source_code)
        func_content = _get_node_text(node, source_code)
        
        # Extract visibility
        visibility = 'public'  # Default
        for child in node.children:
            if child.type == 'visibility':
                # Find specific visibility keywords in children of visibility node
                for vis_child in child.children:
                    if vis_child.type in ['public', 'private', 'internal', 'external']:
                        visibility = vis_child.type
                        break
                break
        
        # Extract modifiers and parameters
        modifiers = []
        parameters = []
        return_type = ''
        
        for child in node.children:
            if child.type == 'modifier_invocation':
                # Parse modifiers
                modifier_name = _get_node_text(child, source_code).strip()
                if modifier_name:
                    modifiers.append(modifier_name)
            elif child.type == 'parameter':
                # Parse parameters
                param_text = _get_node_text(child, source_code).strip()
                if param_text:
                    parameters.append(param_text)
            elif child.type == 'return_type_definition':
                # Parse return type
                return_type = _get_node_text(child, source_code).strip().replace('returns', '').strip().strip('(').strip(')')
        
        # Extract function calls
        function_calls = _extract_function_calls(node, source_code)
        
        return {
            'name': f"{contract_name}.{func_name}" if contract_name else func_name,
            'contract_name': contract_name,
            'content': func_content,
            'signature': func_content.split('{')[0].strip() if '{' in func_content else func_content,
            'visibility': visibility,
            'modifiers': modifiers,
            'parameters': parameters,
            'return_type': return_type,
            'calls': function_calls,
            'line_number': node.start_point[0] + 1,
            'start_line': node.start_point[0] + 1,
            'end_line': node.end_point[0] + 1,
            'file_path': file_path,
            'relative_file_path': os.path.relpath(file_path) if file_path else '',
            'absolute_file_path': os.path.abspath(file_path) if file_path else '',
            'type': 'FunctionDefinition'
        }
    except Exception as e:
        print(f"Failed to parse Solidity function: {e}")
        return None


def _parse_rust_function(node: Node, source_code: bytes, file_path: str) -> Optional[Dict]:
    """Parse Rust function"""
    try:
        name_node = node.child_by_field_name('name')
        if not name_node:
            return None
        
        func_name = _get_node_text(name_node, source_code)
        func_content = _get_node_text(node, source_code)
        
        # Extract file name from file path (without extension)
        import os
        file_name = os.path.splitext(os.path.basename(file_path))[0] if file_path else 'unknown'
        
        # Extract visibility modifiers
        visibility = 'private'  # Rust defaults to private
        modifiers = []
        parameters = []
        return_type = ''
        
        for child in node.children:
            if child.type == 'visibility_modifier':
                # Rust visibility: pub, pub(crate), pub(super), pub(in path)
                vis_text = _get_node_text(child, source_code).strip()
                if vis_text.startswith('pub'):
                    visibility = 'public'
                    if '(' in vis_text:  # pub(crate), pub(super) etc.
                        modifiers.append(vis_text)
            elif child.type == 'parameters':
                # Parse parameter list
                param_text = _get_node_text(child, source_code).strip().strip('(').strip(')')
                if param_text:
                    # Simple parameter splitting (can be further optimized)
                    params = [p.strip() for p in param_text.split(',') if p.strip()]
                    parameters.extend(params)
            elif child.type in ['type', 'primitive_type', 'generic_type']:
                # May be return type
                return_type = _get_node_text(child, source_code).strip()
        
        # Check if there is a return type arrow
        if '->' in func_content:
            return_part = func_content.split('->')[1].split('{')[0].strip() if '{' in func_content else func_content.split('->')[1].strip()
            if return_part:
                return_type = return_part
        
        # Extract function calls
        function_calls = _extract_function_calls(node, source_code)
        
        return {
            'name': f"{file_name}.{func_name}",  # Changed to file_name.function_name format
            'contract_name': file_name,  # Changed to file name
            'content': func_content,
            'signature': func_content.split('{')[0].strip() if '{' in func_content else func_content,
            'visibility': visibility,
            'modifiers': modifiers,
            'parameters': parameters,
            'return_type': return_type,
            'calls': function_calls,
            'line_number': node.start_point[0] + 1,
            'start_line': node.start_point[0] + 1,
            'end_line': node.end_point[0] + 1,
            'file_path': file_path,
            'relative_file_path': os.path.relpath(file_path) if file_path else '',
            'absolute_file_path': os.path.abspath(file_path) if file_path else '',
            'type': 'FunctionDefinition'
        }
    except Exception as e:
        print(f"Failed to parse Rust function: {e}")
        return None


def _parse_cpp_function(node: Node, source_code: bytes, file_path: str) -> Optional[Dict]:
    """Parse C++ function"""
    try:
        declarator = node.child_by_field_name('declarator')
        if not declarator:
            return None
        
        # Extract function name (from declarator)
        func_name = ''
        if declarator.type == 'function_declarator':
            name_node = declarator.child_by_field_name('declarator')
            if name_node:
                func_name = _get_node_text(name_node, source_code).strip()
        else:
            func_name = _get_node_text(declarator, source_code).strip()
        
        # If still no name, try other methods
        if not func_name or '(' in func_name:
            func_name = func_name.split('(')[0].strip() if '(' in func_name else func_name
        
        func_content = _get_node_text(node, source_code)
        
        # Extract return type, visibility and modifiers
        visibility = 'public'  # C++ defaults to public (may differ in classes)
        modifiers = []
        parameters = []
        return_type = ''
        
        # Extract return type
        type_node = node.child_by_field_name('type')
        if type_node:
            return_type = _get_node_text(type_node, source_code).strip()
        
        # Extract parameters
        if declarator.type == 'function_declarator':
            params_node = declarator.child_by_field_name('parameters')
            if params_node:
                param_text = _get_node_text(params_node, source_code).strip().strip('(').strip(')')
                if param_text and param_text != 'void':
                    # Simple parameter splitting
                    params = [p.strip() for p in param_text.split(',') if p.strip() and p.strip() != 'void']
                    parameters.extend(params)
        
        # Check modifiers (static, const, virtual, override, etc.)
        for child in node.children:
            if child.type in ['storage_class_specifier', 'type_qualifier']:
                modifier_text = _get_node_text(child, source_code).strip()
                if modifier_text in ['static', 'const', 'virtual', 'override', 'final', 'inline']:
                    modifiers.append(modifier_text)
        
        # Check const modifier in declaration
        if 'const' in func_content and func_content.count('const') > len([m for m in modifiers if m == 'const']):
            if 'const' not in modifiers:
                modifiers.append('const')
        
        # Extract function calls
        function_calls = _extract_function_calls(node, source_code)
        
        return {
            'name': f"_cpp.{func_name}",
            'contract_name': 'CppModule',
            'content': func_content,
            'signature': func_content.split('{')[0].strip() if '{' in func_content else func_content,
            'visibility': visibility,
            'modifiers': modifiers,
            'parameters': parameters,
            'return_type': return_type,
            'calls': function_calls,
            'line_number': node.start_point[0] + 1,
            'start_line': node.start_point[0] + 1,
            'end_line': node.end_point[0] + 1,
            'file_path': file_path,
            'relative_file_path': os.path.relpath(file_path) if file_path else '',
            'absolute_file_path': os.path.abspath(file_path) if file_path else '',
            'type': 'FunctionDefinition'
        }
    except Exception as e:
        print(f"Failed to parse C++ function: {e}")
        return None


def _parse_move_function(node: Node, source_code: bytes, file_path: str) -> Optional[Dict]:
    """Parse Move function"""
    try:
        name_node = node.child_by_field_name('name')
        if not name_node:
            return None
        
        func_name = _get_node_text(name_node, source_code)
        func_content = _get_node_text(node, source_code)
        
        # Extract visibility, modifiers, parameters and return type
        visibility = 'public'  # Move defaults to public (different from other languages)
        modifiers = []
        parameters = []
        return_type = ''
        is_test_function = False
        
        # Check if it's a test function (test functions default to private)
        func_content_str = func_content
        if '#[test' in func_content_str or 'test_only' in func_content_str:
            visibility = 'private'
            is_test_function = True
        
        # Check visibility modifiers in parent node (Move-specific structure)
        if node.parent and node.parent.type == 'declaration':
            for sibling in node.parent.children:
                if sibling.type == 'module_member_modifier':
                    modifier_text = _get_node_text(sibling, source_code).strip()
                    if modifier_text.startswith('public'):
                        visibility = 'public'
                        if '(' in modifier_text:  # public(script), public(friend), public(package)
                            modifiers.append(modifier_text)
                elif sibling.type == 'attributes' or '#[test' in _get_node_text(sibling, source_code):
                    # Check test markers in attribute nodes
                    attr_text = _get_node_text(sibling, source_code)
                    if '#[test' in attr_text or 'test_only' in attr_text:
                        visibility = 'private'
                        is_test_function = True
        
        for child in node.children:
            if child.type == 'visibility':
                # Move visibility: public, public(script), public(friend)
                vis_text = _get_node_text(child, source_code).strip()
                if vis_text.startswith('public'):
                    visibility = 'public'
                    if '(' in vis_text:  # public(script), public(friend)
                        modifiers.append(vis_text)
            elif child.type == 'public':
                # public node in Move AST
                visibility = 'public'
                # Check for modifiers like public(friend)
                next_sibling = child.next_sibling
                if next_sibling and next_sibling.type == '(':
                    # Collect public(...) form modifiers
                    pub_modifier = 'public'
                    current = next_sibling
                    while current and current.type != ')':
                        pub_modifier += _get_node_text(current, source_code)
                        current = current.next_sibling
                    if current and current.type == ')':
                        pub_modifier += ')'
                        modifiers.append(pub_modifier)
            elif child.type == 'ability':
                # Move-specific ability
                ability_text = _get_node_text(child, source_code).strip()
                modifiers.append(ability_text)
            elif child.type == 'parameters':
                # Parse parameter list
                param_text = _get_node_text(child, source_code).strip().strip('(').strip(')')
                if param_text:
                    # Simple parameter splitting
                    params = [p.strip() for p in param_text.split(',') if p.strip()]
                    parameters.extend(params)
            elif child.type in ['type', 'primitive_type', 'struct_type']:
                # May be return type
                return_type = _get_node_text(child, source_code).strip()
        
        # Check if there is a return type colon
        if ':' in func_content and '{' in func_content:
            # Try to extract return type between : and {
            try:
                colon_part = func_content.split(':')[1].split('{')[0].strip()
                if colon_part and not return_type:
                    return_type = colon_part
            except:
                pass
        
        # Check native modifier
        if 'native' in func_content:
            modifiers.append('native')
        
        # Extract file name from file path (without extension)
        import os
        file_name = os.path.splitext(os.path.basename(file_path))[0] if file_path else 'unknown'
        
        # Extract function calls
        function_calls = _extract_function_calls(node, source_code)
        
        return {
            'name': f"{file_name}.{func_name}",  # Changed to file_name.function_name format
            'contract_name': file_name,  # Use file name as module name
            'content': func_content,
            'signature': func_content.split('{')[0].strip() if '{' in func_content else func_content,
            'visibility': visibility,
            'modifiers': modifiers,
            'parameters': parameters,
            'return_type': return_type,
            'calls': function_calls,
            'line_number': node.start_point[0] + 1,
            'start_line': node.start_point[0] + 1,
            'end_line': node.end_point[0] + 1,
            'file_path': file_path,
            'relative_file_path': os.path.relpath(file_path) if file_path else '',
            'absolute_file_path': os.path.abspath(file_path) if file_path else '',
            'type': 'FunctionDefinition'
        }
    except Exception as e:
        print(f"Failed to parse Move function: {e}")
        return None


def _parse_go_function(node: Node, source_code: bytes, file_path: str) -> Optional[Dict]:
    """Parse Go function"""
    try:
        name_node = node.child_by_field_name('name')
        if not name_node:
            return None
        
        func_name = _get_node_text(name_node, source_code)
        func_content = _get_node_text(node, source_code)
        
        # Extract visibility, modifiers, parameters and return type
        visibility = 'private'  # Go defaults to private
        modifiers = []
        parameters = []
        return_type = ''
        
        # Go visibility is based on first letter case
        if func_name and func_name[0].isupper():
            visibility = 'public'
        
        # Traverse child nodes to extract parameters and return type
        for child in node.children:
            if child.type == 'parameter_list':
                # Parse parameter list
                param_text = _get_node_text(child, source_code).strip().strip('(').strip(')')
                if param_text:
                    # Simple parameter splitting
                    params = [p.strip() for p in param_text.split(',') if p.strip()]
                    parameters.extend(params)
            elif child.type in ['type_identifier', 'pointer_type', 'slice_type', 'array_type']:
                # May be return type
                return_type = _get_node_text(child, source_code).strip()
        
        # Check if it's a method (receiver)
        receiver_node = node.child_by_field_name('receiver')
        if receiver_node:
            receiver_text = _get_node_text(receiver_node, source_code).strip()
            modifiers.append(f"method:{receiver_text}")
        
        # Extract function calls
        function_calls = _extract_function_calls(node, source_code)
        
        return {
            'name': f"_go.{func_name}",
            'contract_name': 'GoPackage',
            'content': func_content,
            'signature': func_content.split('{')[0].strip() if '{' in func_content else func_content,
            'visibility': visibility,
            'modifiers': modifiers,
            'parameters': parameters,
            'return_type': return_type,
            'calls': function_calls,
            'line_number': node.start_point[0] + 1,
            'start_line': node.start_point[0] + 1,
            'end_line': node.end_point[0] + 1,
            'file_path': file_path,
            'relative_file_path': os.path.relpath(file_path) if file_path else '',
            'absolute_file_path': os.path.abspath(file_path) if file_path else '',
            'type': 'FunctionDefinition'
        }
    except Exception as e:
        print(f"Failed to parse Go function: {e}")
        return None


def _parse_python_function(node: Node, source_code: bytes, file_path: str) -> Optional[Dict]:
    """Parse Python function"""
    try:
        name_node = node.child_by_field_name('name')
        if not name_node:
            return None
        
        func_name = _get_node_text(name_node, source_code)
        func_content = _get_node_text(node, source_code)
        
        # Extract file name and check if it's a class method
        file_name = os.path.splitext(os.path.basename(file_path))[0] if file_path else 'unknown'
        
        # Check if function is inside a class (by looking at parent nodes)
        parent = node.parent
        class_name = None
        while parent:
            if parent.type == 'class_definition':
                class_name_node = parent.child_by_field_name('name')
                if class_name_node:
                    class_name = _get_node_text(class_name_node, source_code)
                break
            parent = parent.parent
        
        # Determine visibility
        visibility = 'public'
        if func_name.startswith('__') and func_name.endswith('__'):
            visibility = 'magic'
        elif func_name.startswith('_'):
            visibility = 'private'
        
        # Extract decorators as modifiers
        modifiers = []
        parameters = []
        return_type = ''
        
        # Check for async functions
        for child in node.children:
            if child.type == 'async':
                modifiers.append('async')
        
        # Extract decorators (check previous siblings)
        sibling = node.prev_sibling
        while sibling and sibling.type == 'decorator':
            decorator_text = _get_node_text(sibling, source_code).strip()
            modifiers.append(decorator_text)
            sibling = sibling.prev_sibling
        
        # Extract parameters
        params_node = node.child_by_field_name('parameters')
        if params_node:
            for param_child in params_node.children:
                if param_child.type == 'identifier':
                    param_name = _get_node_text(param_child, source_code).strip()
                    if param_name not in ['(', ')', ',', 'self', 'cls']:
                        parameters.append(param_name)
                elif param_child.type == 'typed_parameter':
                    param_text = _get_node_text(param_child, source_code).strip()
                    parameters.append(param_text)
                elif param_child.type == 'default_parameter':
                    param_text = _get_node_text(param_child, source_code).strip()
                    parameters.append(param_text)
        
        # Extract return type annotation
        return_type_node = node.child_by_field_name('return_type')
        if return_type_node:
            return_type = _get_node_text(return_type_node, source_code).strip()
            if return_type.startswith('->'):
                return_type = return_type[2:].strip()
        
        # Extract function calls
        function_calls = _extract_function_calls(node, source_code)
        
        # Extract docstring
        docstring = ""
        body_node = node.child_by_field_name('body')
        if body_node and body_node.child_count > 0:
            first_child = body_node.children[0]
            if first_child.type == 'expression_statement':
                # Check children of expression_statement for string
                for child in first_child.children:
                    if child.type == 'string':
                        docstring = _get_node_text(child, source_code).strip().strip('"""').strip("'''")
                        break
        
        # Extract signature (everything before the body)
        signature = ""
        if body_node:
            # Use byte offsets to get the signature text from source code
            sig_bytes = source_code[node.start_byte:body_node.start_byte]
            signature = sig_bytes.decode('utf-8').strip()
            if signature.endswith(':'):
                signature = signature[:-1].strip()
        else:
            signature = func_content.split('\n')[0].strip()

        # Build function name with module/class prefix
        if class_name:
            full_name = f"{file_name}.{class_name}.{func_name}"
            contract_name = f"{file_name}.{class_name}"
        else:
            full_name = f"{file_name}.{func_name}"
            contract_name = file_name
        
        return {
            'name': full_name,
            'contract_name': contract_name,
            'content': func_content,
            'signature': signature,
            'docstring': docstring,
            'visibility': visibility,
            'modifiers': modifiers,
            'parameters': parameters,
            'return_type': return_type,
            'calls': function_calls,
            'line_number': node.start_point[0] + 1,
            'start_line': node.start_point[0] + 1,
            'end_line': node.end_point[0] + 1,
            'file_path': file_path,
            'relative_file_path': os.path.relpath(file_path) if file_path else '',
            'absolute_file_path': os.path.abspath(file_path) if file_path else '',
            'type': 'FunctionDefinition'
        }
    except Exception as e:
        print(f"Failed to parse Python function: {e}")
        return None


def parse_project(project_path, project_filter=None):
    """
    Parse project using tree-sitter
    Maintain the same interface as the original parse_project function, and add document chunking functionality
    """
    if project_filter is None:
        project_filter = TreeSitterProjectFilter()

    ignore_folders = set()
    if os.environ.get('IGNORE_FOLDERS'):
        ignore_folders = set(os.environ.get('IGNORE_FOLDERS').split(','))
    ignore_folders.add('.git')

    all_results = []
    all_file_paths = []  # Collect all file paths for chunking

    # Traverse project directory
    for dirpath, dirs, files in os.walk(project_path):
        dirs[:] = [d for d in dirs if d not in ignore_folders]
        for file in files:
            file_path = os.path.join(dirpath, file)
            
            # Collect all file paths (regardless of extension) for chunking
            all_file_paths.append(file_path)
            
            # Apply file filtering (only for function parsing)
            to_scan = not project_filter.filter_file(dirpath, file)
            print("parsing file: ", file_path, " " if to_scan else "[skipped]")

            if to_scan:
                # Detect language type
                language = _detect_language_from_path(Path(file))
                if language:
                    try:
                        # Analyze file using tree-sitter
                        with open(file_path, 'rb') as f:
                            source_code = f.read()
                        
                        parser = Parser()
                        parser.language = LANGUAGES[language]  # Correct API call
                        
                        tree = parser.parse(source_code)
                        functions = _extract_functions_from_node(tree.root_node, source_code, language, file_path)
                        
                        all_results.extend(functions)
                        
                        if functions:
                            print(f"  -> Parsed {len(functions)} functions")
                                
                    except Exception as e:
                        print(f"⚠️  Failed to parse file {file_path}: {e}")
                        continue

    # Filter functions
    functions = [result for result in all_results if result['type'] == 'FunctionDefinition']
    
    # Apply function filtering
    functions_to_check = []
    for function in functions:
        if not project_filter.filter_contract(function):
            functions_to_check.append(function)

    print(f"📊 Parsing completed: total functions {len(functions)}, to check {len(functions_to_check)}")
    
    # Chunk all files in the project (regardless of extension)
    print("🧩 Starting to chunk project files...")
    
    # Get chunking configuration - project parsing defaults to code project configuration
    config = ChunkConfigManager.get_config('code_project')
    print(f"📋 Using configuration: code_project")
    
    # Process file chunking
    chunks = chunk_project_files(all_file_paths, config=config)
    
    print(f"✅ Chunking completed: generated {len(chunks)} document chunks")
    
    # Output chunking statistics
    if chunks:
        chunk_stats = {}
        for chunk in chunks:
            ext = chunk.metadata.get('file_extension', 'unknown') if hasattr(chunk, 'metadata') else 'unknown'
            chunk_stats[ext] = chunk_stats.get(ext, 0) + 1
        
        print("📊 Chunking statistics:")
        for ext, count in sorted(chunk_stats.items()):
            ext_display = ext if ext else '[no extension]'
            print(f"  - {ext_display}: {count} blocks")
    
    return functions, functions_to_check, chunks


if __name__ == "__main__":
    # Simple test
    print("🧪 Testing Tree-sitter project parser...")
    
    import tempfile
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test file
        test_file = os.path.join(temp_dir, 'test.sol')
        with open(test_file, 'w') as f:
            f.write("""
pragma solidity ^0.8.0;

contract TestContract {
    uint256 public balance;
    
    function deposit() public payable {
        balance += msg.value;
    }
    
    function withdraw(uint256 amount) public {
        require(balance >= amount, "Insufficient balance");
        balance -= amount;
        payable(msg.sender).transfer(amount);
    }
}
""")
        
        # Test parsing
        functions, functions_to_check, chunks = parse_project(temp_dir)
        print(f"✅ Found {len(functions)} functions, {len(functions_to_check)} need checking, {len(chunks)} chunks")
        
        if functions_to_check:
            for func in functions_to_check:
                print(f"  - {func['name']} ({func['visibility']})")
        
    print("✅ Test completed")