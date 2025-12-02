#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Document chunker - Intelligent chunking tool based on adalflow TextSplitter

Features:
- Traverse all files in specified folders
- Use adalflow TextSplitter for document chunking
- Support multiple chunking strategies and parameter configurations
- Paragraph segmentation optimized for long texts
- Support configuration file management
- Output structured chunking results

Author: Implemented based on adalflow components, integrated long text processing capabilities
"""

import os
import re
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Literal
from dataclasses import dataclass, asdict

# 尝试导入 adalflow，如果不可用则使用简单实现
try:
    from adalflow.components.data_process.text_splitter import TextSplitter
    from adalflow.core.types import Document
    ADALFLOW_AVAILABLE = True
except ImportError:
    ADALFLOW_AVAILABLE = False
    print("[WARNING] adalflow unavailable, using simple text chunking implementation")

# 导入配置管理器
try:
    from .chunk_config import ChunkConfig, ChunkConfigManager
except ImportError:
    try:
        from chunk_config import ChunkConfig, ChunkConfigManager
    except ImportError:
        print("[WARNING] Unable to import chunk_config, will use basic configuration")


@dataclass
class ChunkResult:
    """Chunk result data structure"""
    chunk_id: str
    original_file: str
    chunk_text: str
    chunk_order: int
    parent_doc_id: str
    chunk_size: int
    metadata: Dict[str, Any]


@dataclass
class ProcessingStats:
    """Processing statistics information"""
    total_files: int
    processed_files: int
    total_chunks: int
    skipped_files: List[str]
    error_files: List[str]


class SimpleTextSplitter:
    """Simple text splitter implementation (used when adalflow is unavailable)"""
    
    def __init__(self, split_by="word", chunk_size=800, chunk_overlap=200, **kwargs):
        self.split_by = split_by
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def call(self, documents):
        """Chunk process document list"""
        result = []
        for doc in documents:
            chunks = self._split_document(doc)
            result.extend(chunks)
        return result
    
    def _split_document(self, doc):
        """Chunk single document"""
        text = doc.text
        chunks = []
        
        if self.split_by == "word":
            words = text.split()
            for i in range(0, len(words), self.chunk_size - self.chunk_overlap):
                chunk_words = words[i:i + self.chunk_size]
                chunk_text = ' '.join(chunk_words)
                
                if chunk_text.strip():
                    chunk_doc = type('Document', (), {
                        'id': f"{doc.id}_chunk_{len(chunks)}",
                        'text': chunk_text,
                        'order': len(chunks),
                        'parent_doc_id': doc.id,
                        'meta_data': doc.meta_data.copy() if doc.meta_data else {}
                    })()
                    chunks.append(chunk_doc)
        
        elif self.split_by == "sentence":
            # 简单按句子分割
            sentences = text.split('.')
            for i in range(0, len(sentences), self.chunk_size - self.chunk_overlap):
                chunk_sentences = sentences[i:i + self.chunk_size]
                chunk_text = '.'.join(chunk_sentences)
                
                if chunk_text.strip():
                    chunk_doc = type('Document', (), {
                        'id': f"{doc.id}_chunk_{len(chunks)}",
                        'text': chunk_text,
                        'order': len(chunks),
                        'parent_doc_id': doc.id,
                        'meta_data': doc.meta_data.copy() if doc.meta_data else {}
                    })()
                    chunks.append(chunk_doc)
        
        else:
            # 按字符分割
            for i in range(0, len(text), self.chunk_size - self.chunk_overlap):
                chunk_text = text[i:i + self.chunk_size]
                
                if chunk_text.strip():
                    chunk_doc = type('Document', (), {
                        'id': f"{doc.id}_chunk_{len(chunks)}",
                        'text': chunk_text,
                        'order': len(chunks),
                        'parent_doc_id': doc.id,
                        'meta_data': doc.meta_data.copy() if doc.meta_data else {}
                    })()
                    chunks.append(chunk_doc)
        
        return chunks


class SimpleDocument:
    """Simple document class (used when adalflow is unavailable)"""
    
    def __init__(self, text, id, meta_data=None):
        self.text = text
        self.id = id
        self.meta_data = meta_data or {}


class DocumentChunker:
    """Document chunker class"""
    
    # Supported file extensions
    SUPPORTED_EXTENSIONS = {
        '.txt', '.md', '.rst', '.py', '.js', '.ts', '.html', '.xml',
        '.json', '.yaml', '.yml', '.css', '.sql', '.sh', '.bat',
        '.c', '.cpp', '.h', '.hpp', '.java', '.php', '.rb', '.go',
        '.rs', '.scala', '.kt', '.swift', '.dart', '.r', '.m', '.sol', '.move','.cc'
    }
    
    def __init__(
        self,
        split_by: Literal["word", "sentence", "page", "passage", "token"] = "word",
        chunk_size: int = 800,
        chunk_overlap: int = 200,
        batch_size: int = 1000,
        encoding: str = 'utf-8',
        max_file_size_mb: float = 50.0,
        include_extensions: Optional[List[str]] = None,
        exclude_patterns: Optional[List[str]] = None,
        long_text_mode: bool = False
    ):
        """
        Initialize document chunker

        Args:
            split_by: Split strategy ("word", "sentence", "page", "passage", "token")
            chunk_size: Maximum units per chunk
            chunk_overlap: Overlap units between chunks
            batch_size: Batch processing size
            encoding: File encoding
            max_file_size_mb: Maximum file size limit (MB)
            include_extensions: Specified file extensions to include
            exclude_patterns: File name patterns to exclude
            long_text_mode: Long text mode, automatically optimize parameters for processing long documents
        """
        
        # Long text mode automatic parameter optimization
        self.long_text_mode = long_text_mode
        if long_text_mode:
            # Optimize parameters for long text passage splitting
            if split_by == "passage":
                chunk_size = max(chunk_size, 5)  # 至少5个段落
                chunk_overlap = max(chunk_overlap, 2)  # 至少2个段落重叠
                max_file_size_mb = max(max_file_size_mb, 100.0)  # 增加文件大小限制
            elif split_by == "word":
                chunk_size = max(chunk_size, 1500)  # 长文本适用更大的块
                chunk_overlap = max(chunk_overlap, 300)
        
        # Initialize TextSplitter
        if ADALFLOW_AVAILABLE:
            self.text_splitter = TextSplitter(
                split_by=split_by,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                batch_size=batch_size
            )
        else:
            self.text_splitter = SimpleTextSplitter(
                split_by=split_by,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
        
        self.encoding = encoding
        self.max_file_size_bytes = max_file_size_mb * 1024 * 1024
        self.include_extensions = set(include_extensions) if include_extensions else None  # None表示包含所有文件
        self.exclude_patterns = exclude_patterns or []
        
        # Configure logging
        self.logger = logging.getLogger(__name__)
        
        # Output initialization information
        print(f"[INFO] DocumentChunker initialized:")
        print(f"  - Split strategy: {split_by}")
        print(f"  - Chunk size: {chunk_size}")
        print(f"  - Overlap size: {chunk_overlap}")
        print(f"  - Long text mode: {'Yes' if long_text_mode else 'No'}")
        print(f"  - Adalflow available: {'Yes' if ADALFLOW_AVAILABLE else 'No'}")
        
        if long_text_mode:
            print(f"  [INFO] Long text mode enabled, parameters optimized")
    
    @classmethod
    def from_config(cls, config: 'ChunkConfig'):
        """
        Create chunker from configuration object

        Args:
            config: ChunkConfig configuration object

        Returns:
            DocumentChunker: Configured chunker instance
        """
        return cls(
            split_by=config.split_by,
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            batch_size=config.batch_size,
            encoding=config.encoding,
            max_file_size_mb=config.max_file_size_mb,
            include_extensions=config.include_extensions,
            exclude_patterns=config.exclude_patterns,
            long_text_mode=config.long_text_mode
        )
    
    @classmethod
    def for_long_text_passage(
        cls,
        chunk_size: int = 8,
        chunk_overlap: int = 3,
        max_file_size_mb: float = 200.0,
        include_extensions: Optional[List[str]] = None
    ):
        """
        Convenient constructor specifically created for long text passage splitting

        Args:
            chunk_size: Number of paragraphs, recommended 5-15 paragraphs
            chunk_overlap: Number of overlapping paragraphs, recommended 2-5 paragraphs
            max_file_size_mb: Maximum file size, long text recommends 100MB or more
            include_extensions: Supported file types, defaults to text types

        Returns:
            DocumentChunker: Configured long text processor
        """
        
        # Common file types for long text
        if include_extensions is None:
            include_extensions = [
                '.txt', '.md', '.rst', '.doc', '.docx',  # 文档类型
                '.pdf', '.rtf', '.odt',                   # 富文本类型  
                '.epub', '.mobi',                         # 电子书类型
                '.html', '.xml'                           # 标记语言
            ]
        
        return cls(
            split_by="passage",
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            batch_size=500,  # 长文本减少批处理大小
            max_file_size_mb=max_file_size_mb,
            include_extensions=include_extensions,
            long_text_mode=True
        )
    
    def process_files(self, file_paths: List[str]) -> List[ChunkResult]:
        """
        Process file list and perform chunking

        Args:
            file_paths: File path list

        Returns:
            Chunk result list
        """
        all_chunks = []
        
        for file_path in file_paths:
            try:
                chunks = self._process_single_file(Path(file_path))
                if chunks:
                    all_chunks.extend(chunks)
                    print(f"[SUCCESS] Chunked file: {Path(file_path).name} -> {len(chunks)} chunks")
                    
            except Exception as e:
                print(f"[ERROR] Failed to chunk file: {Path(file_path).name} - {str(e)}")
                continue
        
        return all_chunks
    
    def _should_process_file(self, file_path: Path) -> bool:
        """Determine if the file should be processed"""
        # 检查文件大小
        try:
            file_size = file_path.stat().st_size
            if file_size > self.max_file_size_bytes:
                print(f"[WARNING] File too large, skipping chunking: {file_path.name} ({file_size / 1024 / 1024:.2f}MB)")
                return False
        except OSError:
            print(f"[WARNING] Unable to access file: {file_path.name}")
            return False
        
        # Check exclude patterns
        file_path_str = str(file_path)
        for pattern in self.exclude_patterns:
            # If pattern starts with dot, match extension exactly
            if pattern.startswith('.'):
                if file_path.suffix == pattern or file_path.name.endswith(pattern):
                    return False
            # Otherwise match by path contains
            elif pattern in file_path_str:
                return False
        
        # Check include extensions (only when include_extensions is not None)
        if self.include_extensions is not None:
            file_ext = file_path.suffix.lower()
            # 将include_extensions也转换为小写进行比较
            include_exts_lower = {ext.lower() for ext in self.include_extensions}
            if file_ext not in include_exts_lower and file_path.name not in self.include_extensions:
                return False
        
        return True
    
    def _process_single_file(self, file_path: Path) -> List[ChunkResult]:
        """Process single file"""
        if not self._should_process_file(file_path):
            return []
            
        try:
            # Read file content
            content = self._read_file_with_encoding(file_path)
            
            if not content or len(content.strip()) < 50:  # 跳过过短的文件
                return []
            
            # Special preprocessing for long text mode
            if self.long_text_mode and self.text_splitter.split_by == "passage":
                content = self._preprocess_long_text(content)
            
            # Create Document object
            if ADALFLOW_AVAILABLE:
                doc = Document(
                    text=content,
                    id=str(file_path),
                    meta_data={
                        'file_name': file_path.name,
                        'file_path': str(file_path),
                        'file_size': file_path.stat().st_size,
                        'file_extension': file_path.suffix
                    }
                )
            else:
                doc = SimpleDocument(
                    text=content,
                    id=str(file_path),
                    meta_data={
                        'file_name': file_path.name,
                        'file_path': str(file_path),
                        'file_size': file_path.stat().st_size,
                        'file_extension': file_path.suffix
                    }
                )
            
            # Use TextSplitter for chunking
            split_docs = self.text_splitter.call([doc])
            
            # Convert to ChunkResult objects
            chunks = []
            for split_doc in split_docs:
                chunk = ChunkResult(
                    chunk_id=split_doc.id,
                    original_file=str(file_path),
                    chunk_text=split_doc.text,
                    chunk_order=split_doc.order,
                    parent_doc_id=split_doc.parent_doc_id,
                    chunk_size=len(split_doc.text.split()) if self.text_splitter.split_by == "word" else len(split_doc.text),
                    metadata=split_doc.meta_data or {}
                )
                chunks.append(chunk)
            
            return chunks
            
        except Exception as e:
            print(f"⚠️  Error processing file {file_path}: {e}")
            return []
    
    def _read_file_with_encoding(self, file_path: Path) -> str:
        """Try reading file with different encodings"""
        encodings = [self.encoding, 'utf-8', 'gbk', 'latin1', 'cp1252']
        
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding, errors='ignore') as f:
                    content = f.read().strip()
                return content
            except UnicodeDecodeError:
                continue
            except Exception:
                continue
        
        # 如果所有编码都失败，返回空字符串
        return ""
    
    def _preprocess_long_text(self, content: str) -> str:
        """
        Long text preprocessing, optimize passage splitting effect

        Args:
            content: Original text content

        Returns:
            str: Preprocessed text
        """
        # 1. Normalize line breaks
        content = content.replace('\r\n', '\n').replace('\r', '\n')
        
        # 2. Clean up extra blank lines, but preserve paragraph separation
        lines = content.split('\n')
        cleaned_lines = []
        prev_empty = False
        
        for line in lines:
            line = line.strip()
            if line:  # 非空行
                cleaned_lines.append(line)
                prev_empty = False
            else:  # 空行
                if not prev_empty:  # 只保留一个空行作为段落分隔
                    cleaned_lines.append('')
                    prev_empty = True
        
        # 3. 重新组合，确保段落之间有双换行
        processed_content = '\n'.join(cleaned_lines)
        
        # 4. 确保段落分隔符正确
        # 将单换行后的空行转换为双换行
        processed_content = processed_content.replace('\n\n', '\n\n')  # 保持现有的双换行
        
        # 5. Process chapter titles (if there are obvious title markers)
        if self._detect_chapter_markers(processed_content):
            processed_content = self._enhance_chapter_separation(processed_content)
        
        # 6. Statistical processing results
        original_paragraphs = content.count('\n\n') + 1
        processed_paragraphs = processed_content.count('\n\n') + 1
        
        if hasattr(self, 'logger') and self.logger:
            self.logger.info(f"📝 Long text preprocessing completed: {original_paragraphs} -> {processed_paragraphs} paragraphs")
        
        return processed_content
    
    def _detect_chapter_markers(self, content: str) -> bool:
        """Detect if there are chapter markers"""
        chapter_patterns = [
            r'第[一二三四五六七八九十\d]+章',  # 中文章节
            r'Chapter\s+\d+',                   # 英文章节
            r'^#{1,3}\s+',                      # Markdown标题
            r'^\d+\.\s+[A-Z]',                 # 数字标题
        ]
        
        for pattern in chapter_patterns:
            if re.search(pattern, content, re.MULTILINE):
                return True
        return False
    
    def _enhance_chapter_separation(self, content: str) -> str:
        """Enhance chapter separation"""
        # 在章节标题前添加额外的分隔
        patterns = [
            (r'(第[一二三四五六七八九十\d]+章)', r'\n\n\1'),
            (r'(Chapter\s+\d+)', r'\n\n\1'),
            (r'^(#{1,3}\s+)', r'\n\n\1', re.MULTILINE),
        ]
        
        for pattern, replacement, *flags in patterns:
            flag = flags[0] if flags else 0
            content = re.sub(pattern, replacement, content, flags=flag)
        
        return content


def chunk_project_files(file_paths: List[str], config: Optional['ChunkConfig'] = None, **kwargs) -> List[ChunkResult]:
    """
    Convenient function to chunk project files

    Args:
        file_paths: File path list
        config: ChunkConfig configuration object, prioritized if provided
        **kwargs: DocumentChunker parameters (used when config is None)

    Returns:
        Chunk result list
    """
    if config:
        chunker = DocumentChunker.from_config(config)
    else:
        chunker = DocumentChunker(**kwargs)
    return chunker.process_files(file_paths)


def chunk_project_files_with_preset(file_paths: List[str], preset: str = "code_project") -> List[ChunkResult]:
    """
    Chunk project files using preset configuration

    Args:
        file_paths: File path list
        preset: Preset configuration name, defaults to 'code_project'

    Returns:
        Chunk result list
    """
    config = ChunkConfigManager.get_config(preset)
    return chunk_project_files(file_paths, config=config)


if __name__ == "__main__":
    # Simple test
    import tempfile
    
    print("🧪 Testing DocumentChunker...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # 创建测试文件
        test_file = Path(temp_dir) / 'test.txt'
        with open(test_file, 'w', encoding='utf-8') as f:
            f.write("This is a test document." * 100)  # Create longer content
        
        # 测试分块
        chunker = DocumentChunker(chunk_size=50, chunk_overlap=10)
        chunks = chunker.process_files([str(test_file)])
        
        print(f"✅ Test completed, generated {len(chunks)} chunks")
        for i, chunk in enumerate(chunks[:3]):  # Show first 3
            print(f"  Chunk {i+1}: {len(chunk.chunk_text)} characters")