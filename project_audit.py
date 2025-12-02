#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Tree-sitter Based Project Audit
Use tree-sitter instead of ANTLR for project auditing
"""

import csv
import re
import os
import sys
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add path for import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    from .project_parser import parse_project, TreeSitterProjectFilter
except ImportError:
    # If relative import fails, try direct import
    from project_parser import parse_project, TreeSitterProjectFilter

# Import call_tree_builder from local module
try:
    from .call_tree_builder import TreeSitterCallTreeBuilder
except ImportError:
    # If relative import fails, try direct import
    from call_tree_builder import TreeSitterCallTreeBuilder

# Import call_graph related modules
from ts_parser_core import MultiLanguageAnalyzer, LanguageType
from ts_parser_core.ts_parser.data_structures import CallGraphEdge

# Import logging system
try:
    from logging_config import get_logger, log_step, log_success, log_warning, log_data_info
    LOGGING_AVAILABLE = True
except ImportError:
    LOGGING_AVAILABLE = False


class TreeSitterProjectAudit(object):
    """Tree-sitter based project auditor"""
    
    def __init__(self, project_id, project_path, db_engine=None):
        self.project_id = project_id
        self.project_path = project_path
        self.db_engine = db_engine  # Optional database engine
        self.functions = []
        self.functions_to_check = []
        self.chunks = []  # Store document chunking results
        self.tasks = []
        self.taskkeys = set()
        self.call_tree_builder = TreeSitterCallTreeBuilder()
        self.call_trees = []
        
        # Initialize call_graph related attributes
        self.call_graphs = []  # Store call_graph for all languages
        self.analyzer = MultiLanguageAnalyzer()
        
        # Initialize logging
        if LOGGING_AVAILABLE:
            self.logger = get_logger(f"ProjectAudit[{project_id}]")
            self.logger.info(f"Initialize project auditor: {project_id}")
            self.logger.info(f"Project path: {project_path}")
        else:
            self.logger = None

    def print_call_tree(self, node, level=0, prefix=''):
        """Print call tree (proxy to CallTreeBuilder)"""
        self.call_tree_builder.print_call_tree(node, level, prefix)

    def parse(self):
        """
        Parse project files and build call tree
        """
        if self.logger:
            log_step(self.logger, "Create project filter")
        
        parser_filter = TreeSitterProjectFilter()
        
        if self.logger:
            log_step(self.logger, "Start parsing project files")
        
        functions, functions_to_check, chunks = parse_project(self.project_path, parser_filter)
        self.functions = functions
        self.functions_to_check = functions_to_check
        self.chunks = chunks
        
        if self.logger:
            log_success(self.logger, "Project file parsing completed")
            log_data_info(self.logger, "Total number of functions", len(self.functions))
            log_data_info(self.logger, "Number of functions to check", len(self.functions_to_check))
            log_data_info(self.logger, "Number of document chunks", len(self.chunks))
        
        # 使用TreeSitterCallTreeBuilder构建调用树
        if self.logger:
            log_step(self.logger, "Start building call tree")
        else:
            print("🌳 Start building call tree...")
            
        self.call_trees = self.call_tree_builder.build_call_trees(functions_to_check, max_workers=1)
        
        if self.logger:
            log_success(self.logger, "Call tree building completed")
            log_data_info(self.logger, "Built call trees", len(self.call_trees))
        else:
            print(f"✅ Call tree building completed, built {len(self.call_trees)} call trees")
        
        # Build call graph
        self._build_call_graphs()

    def get_function_names(self):
        """Get all function names"""
        return set([function['name'] for function in self.functions])
    
    def get_functions_by_contract(self, contract_name):
        """Get function list by contract name"""
        return [func for func in self.functions if func.get('contract_name') == contract_name]
    
    def get_function_by_name(self, function_name):
        """Get function information by function name"""
        for func in self.functions:
            if func['name'] == function_name:
                return func
        return None
    
    def export_to_csv(self, output_path):
        """Export analysis results to CSV"""
        with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['name', 'contract', 'visibility', 'line_number', 'file_path', 'modifiers', 'calls_count']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for func in self.functions_to_check:
                writer.writerow({
                    'name': func.get('name', ''),
                    'contract': func.get('contract_name', ''),
                    'visibility': func.get('visibility', ''),
                    'line_number': func.get('line_number', ''),
                    'file_path': func.get('file_path', ''),
                    'modifiers': ', '.join(func.get('modifiers', [])),
                    'calls_count': len(func.get('calls', []))
                })
    
    def _build_call_graphs(self):
        """Build call graphs (internal method)"""
        if not self.analyzer:
            if self.logger:
                log_warning(self.logger, "MultiLanguageAnalyzer unavailable, skip call graph building")
            else:
                print("⚠️ MultiLanguageAnalyzer unavailable, skip call graph building")
            self.call_graphs = []
            return
        
        if self.logger:
            log_step(self.logger, "Start building call graph")
        else:
            print("🔗 Start building call graph...")
        
        try:
            # Build call graph based on project path and function information
            language_paths = self._detect_project_languages()
            
            total_call_graphs = []
            
            for language, paths in language_paths.items():
                for project_path in paths:
                    try:
                        if self.logger:
                            self.logger.info(f"Analyze {language.value} project directory: {project_path}")
                        else:
                            print(f"  📁 Analyze {language.value} project directory: {project_path}")
                        
                        # Use MultiLanguageAnalyzer to analyze the entire directory
                        self.analyzer.analyze_directory(project_path, language)
                        
                        # Get call graph
                        call_graph = self.analyzer.get_call_graph(language)
                        
                        if call_graph:
                            total_call_graphs.extend(call_graph)
                            
                        if self.logger:
                            self.logger.info(f"Found {len(call_graph)} call relationships")
                        else:
                            print(f"  ✅ Found {len(call_graph)} call relationships")
                            
                    except Exception as e:
                        if self.logger:
                            self.logger.warning(f"Failed to analyze directory {project_path}: {e}")
                        else:
                            print(f"  ⚠️ Failed to analyze directory {project_path}: {e}")
                        continue
            
            self.call_graphs = total_call_graphs
            
            if self.logger:
                log_success(self.logger, "Call graph building completed")
                log_data_info(self.logger, "Built call relationships", len(self.call_graphs))
            else:
                print(f"✅ Call graph building completed, found {len(self.call_graphs)} call relationships")
                
        except Exception as e:
            if self.logger:
                self.logger.error(f"Call graph building failed: {e}")
            else:
                print(f"❌ Call graph building failed: {e}")
            self.call_graphs = []
    
    def _detect_project_languages(self):
        """Detect language types in the project"""
        from pathlib import Path
        language_paths = {}
        
        project_path = Path(self.project_path)
        
        # Detect Solidity files
        sol_files = list(project_path.rglob('*.sol'))
        if sol_files:
            language_paths[LanguageType.SOLIDITY] = [str(project_path)]
        
        # Detect Rust files
        rs_files = list(project_path.rglob('*.rs'))
        if rs_files:
            language_paths[LanguageType.RUST] = [str(project_path)]
        
        # Detect C++ files
        cpp_files = list(project_path.rglob('*.cpp')) + list(project_path.rglob('*.cc')) + list(project_path.rglob('*.cxx'))
        if cpp_files:
            language_paths[LanguageType.CPP] = [str(project_path)]
        
        # Detect Move files
        move_files = list(project_path.rglob('*.move'))
        if move_files:
            language_paths[LanguageType.MOVE] = [str(project_path)]
        
        # Detect Go files
        go_files = list(project_path.rglob('*.go'))
        if go_files:
            language_paths[LanguageType.GO] = [str(project_path)]
        
        # Detect Python files
        py_files = list(project_path.rglob('*.py'))
        if py_files:
            language_paths[LanguageType.PYTHON] = [str(project_path)]
        
        return language_paths
    
    def get_call_graphs(self):
        """Get call graphs"""
        return self.call_graphs.copy() if self.call_graphs else []
    
    def print_call_graph(self, limit=50):
        """Print call graph information"""
        if not self.call_graphs:
            print("📊 No call graph data")
            return
        
        print(f"📊 Call Graph Overview (total {len(self.call_graphs)} call relationships):")
        print("=" * 80)
        
        displayed = 0
        for edge in self.call_graphs:
            if displayed >= limit:
                print(f"... and {len(self.call_graphs) - limit} more call relationships")
                break
                
            caller_short = edge.caller.split('.')[-1] if '.' in edge.caller else edge.caller
            callee_short = edge.callee.split('.')[-1] if '.' in edge.callee else edge.callee
            
            print(f"➡️  {caller_short} -> {callee_short} [{edge.call_type.value}] ({edge.language.value})")
            displayed += 1
        
        print("=" * 80)
    
    def get_call_graph_statistics(self):
        """Get call graph statistics"""
        if not self.call_graphs:
            return {"total_edges": 0, "languages": {}, "call_types": {}}
        
        stats = {
            "total_edges": len(self.call_graphs),
            "languages": {},
            "call_types": {},
            "unique_functions": set()
        }
        
        for edge in self.call_graphs:
            # 统计语言
            lang = edge.language.value
            stats["languages"][lang] = stats["languages"].get(lang, 0) + 1
            
            # 统计调用类型
            call_type = edge.call_type.value
            stats["call_types"][call_type] = stats["call_types"].get(call_type, 0) + 1
            
            # 统计独特函数
            stats["unique_functions"].add(edge.caller)
            stats["unique_functions"].add(edge.callee)
        
        stats["unique_functions_count"] = len(stats["unique_functions"])
        del stats["unique_functions"]  # 移除set，不需要返回
        
        return stats
    
    def get_chunks(self):
        """Get document chunking results"""
        return self.chunks.copy() if self.chunks else []
    
    def get_chunks_by_file(self, file_path):
        """Get chunking results by file path"""
        if not self.chunks:
            return []
        return [chunk for chunk in self.chunks if chunk.original_file == file_path]
    
    def get_chunk_statistics(self):
        """Get chunking statistics"""
        if not self.chunks:
            return {"total_chunks": 0, "files": {}, "avg_chunk_size": 0}
        
        stats = {
            "total_chunks": len(self.chunks),
            "files": {},
            "file_extensions": {},
            "total_size": 0
        }
        
        for chunk in self.chunks:
            # 按文件统计
            file_path = chunk.original_file
            stats["files"][file_path] = stats["files"].get(file_path, 0) + 1
            
            # 按文件扩展名统计
            if hasattr(chunk, 'metadata') and 'file_extension' in chunk.metadata:
                ext = chunk.metadata['file_extension']
                stats["file_extensions"][ext] = stats["file_extensions"].get(ext, 0) + 1
            
            # 累计大小
            stats["total_size"] += chunk.chunk_size
        
        # 计算平均大小
        if stats["total_chunks"] > 0:
            stats["avg_chunk_size"] = stats["total_size"] / stats["total_chunks"]
        else:
            stats["avg_chunk_size"] = 0
        
        return stats
    
    def print_chunk_statistics(self):
        """Print chunking statistics"""
        stats = self.get_chunk_statistics()
        
        if stats["total_chunks"] == 0:
            print("📄 No chunking data")
            return
        
        print(f"📄 Document chunking statistics (total {stats['total_chunks']} chunks):")
        print("=" * 60)
        print(f"🔢 Total chunks: {stats['total_chunks']}")
        print(f"📏 Average chunk size: {stats['avg_chunk_size']:.1f} units")
        print(f"📁 Number of files involved: {len(stats['files'])}")
        
        if stats['file_extensions']:
            print("\n📂 File type distribution:")
            for ext, count in sorted(stats['file_extensions'].items()):
                print(f"  {ext if ext else '[No extension]'}: {count} chunks")
        
        print("\n📄 File chunking details (top 10):")
        file_list = sorted(stats['files'].items(), key=lambda x: x[1], reverse=True)[:10]
        for file_path, count in file_list:
            file_name = os.path.basename(file_path)
            print(f"  {file_name}: {count} chunks")
        
        if len(stats['files']) > 10:
            print(f"  ... and {len(stats['files']) - 10} more files")
        
        print("=" * 60)
    
    def print_chunk_samples(self, limit=3):
        """Print chunking samples"""
        if not self.chunks:
            print("📄 No chunking data")
            return
        
        print(f"📄 Chunking samples (first {min(limit, len(self.chunks))}):")
        print("=" * 80)
        
        for i, chunk in enumerate(self.chunks[:limit]):
            print(f"\n🧩 Chunk {i+1}:")
            print(f"  📁 File: {os.path.basename(chunk.original_file)}")
            print(f"  🔢 Order: {chunk.chunk_order}")
            print(f"  📏 Size: {chunk.chunk_size} units")
            print(f"  📝 Content preview:")
            preview = chunk.chunk_text[:200]
            if len(chunk.chunk_text) > 200:
                preview += "..."
            print(f"     {preview}")
        
        print("=" * 80)


if __name__ == '__main__':
    # Simple test
    print("🧪 Testing TreeSitterProjectAudit...")
    
    # Create temporary test directory
    import tempfile
    with tempfile.TemporaryDirectory() as temp_dir:
        test_file = os.path.join(temp_dir, 'test.sol')
        with open(test_file, 'w') as f:
            f.write("""
pragma solidity ^0.8.0;

contract TestContract {
    function testFunction() public pure returns (uint256) {
        return 42;
    }
}
""")
        
        audit = TreeSitterProjectAudit("test", temp_dir)
        audit.parse()
        
        print(f"✅ Parsing completed, found {len(audit.functions)} functions")
        print(f"✅ Need to check {len(audit.functions_to_check)} functions")
        print(f"✅ Built {len(audit.call_trees)} call trees")
        print(f"✅ Built {len(audit.call_graphs)} call relationships")
        print(f"✅ Generated {len(audit.chunks)} document chunks")
        
        # 测试 call graph 相关功能
        call_graph_stats = audit.get_call_graph_statistics()
        print(f"📊 Call Graph Statistics: {call_graph_stats}")
        
        if audit.call_graphs:
            print("🔗 Call Graph Examples:")
            audit.print_call_graph(limit=5)
        
        # 测试分块相关功能
        if audit.chunks:
            print("\n📄 Testing chunking functionality:")
            audit.print_chunk_statistics()
            audit.print_chunk_samples(limit=2)
        
    print("✅ TreeSitterProjectAudit test completed")