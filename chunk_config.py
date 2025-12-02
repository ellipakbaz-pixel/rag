#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Document chunker configuration file

Provides preset configurations and custom configuration options for various chunking strategies
"""

import os
from typing import List, Dict, Any, Optional, Literal
from dataclasses import dataclass


@dataclass
class ChunkConfig:
    """Chunk configuration dataclass"""
    split_by: Literal["word", "sentence", "page", "passage", "token"] = "word"
    chunk_size: int = 800
    chunk_overlap: int = 200
    batch_size: int = 1000
    encoding: str = 'utf-8'
    max_file_size_mb: float = 50.0
    include_extensions: Optional[List[str]] = None
    exclude_patterns: Optional[List[str]] = None
    long_text_mode: bool = False


class ChunkConfigManager:
    """Chunk configuration manager"""
    
    # Preset configurations
    PRESET_CONFIGS = {
        # Default configuration - suitable for general projects
        "default": ChunkConfig(
            split_by="word",
            chunk_size=800,
            chunk_overlap=200,
            max_file_size_mb=10.0,
            exclude_patterns=['.git', '__pycache__', '.pyc', '.log', '.tmp', '.cache']
        ),

        # Code project configuration - suitable for code auditing, includes all files (except excluded ones)
        "code_project": ChunkConfig(
            split_by="word",
            chunk_size=1000,
            chunk_overlap=250,
            max_file_size_mb=20.0,
            exclude_patterns=[
                '.git', '__pycache__', '.pyc', '.log', '.tmp', '.cache',
                'node_modules', '.next', 'dist', 'build', '.vscode',
                '.idea', '.DS_Store', 'coverage', '.nyc_output', '.bin',
                '.dll', '.so', '.dylib', '.exe', '.zip', '.tar', '.gz',
                '.rar', '.7z', '.jar', '.war', '.ear', '.deb', '.rpm',
                '.dmg', '.iso', '.img', '.vdi', '.vmdk', '.qcow2'
            ]
        ),

        # Long text configuration - suitable for documents, novels, etc.
        "long_text": ChunkConfig(
            split_by="passage",
            chunk_size=8,
            chunk_overlap=3,
            batch_size=500,
            max_file_size_mb=200.0,
            include_extensions=[
                '.txt', '.md', '.rst', '.doc', '.docx',
                '.pdf', '.rtf', '.odt', '.epub', '.mobi',
                '.html', '.xml'
            ],
            exclude_patterns=['.git', '.cache', '.tmp'],
            long_text_mode=True
        ),

        # Academic paper configuration
        "academic": ChunkConfig(
            split_by="passage", 
            chunk_size=6,
            chunk_overlap=2,
            max_file_size_mb=100.0,
            include_extensions=['.txt', '.md', '.tex', '.pdf', '.doc', '.docx'],
            exclude_patterns=['.git', '.cache', '.tmp', '.aux', '.log', '.bbl'],
            long_text_mode=True
        ),

        # Technical documentation configuration
        "tech_docs": ChunkConfig(
            split_by="passage",
            chunk_size=5,
            chunk_overlap=2,
            max_file_size_mb=50.0,
            include_extensions=['.md', '.rst', '.txt', '.html', '.xml'],
            exclude_patterns=['.git', '.cache', '.tmp'],
            long_text_mode=True
        ),

        # Small file precise segmentation configuration
        "precise": ChunkConfig(
            split_by="sentence",
            chunk_size=3,
            chunk_overlap=1,
            max_file_size_mb=10.0,
            exclude_patterns=['.git', '__pycache__', '.pyc', '.log', '.tmp']
        ),

        # Large context configuration - maintains more context information
        "large_context": ChunkConfig(
            split_by="passage",
            chunk_size=12,
            chunk_overlap=4,
            max_file_size_mb=500.0,
            long_text_mode=True,
            exclude_patterns=['.git', '.cache', '.tmp']
        ),

        # Token segmentation configuration - suitable for LLM processing
        "token_based": ChunkConfig(
            split_by="token",
            chunk_size=512,
            chunk_overlap=50,
            max_file_size_mb=100.0,
            exclude_patterns=['.git', '__pycache__', '.pyc', '.log', '.tmp']
        )
    }
    
    @classmethod
    def get_config(cls, preset_name: str = "default") -> ChunkConfig:
        """
        Get preset configuration

        Args:
            preset_name: Preset configuration name

        Returns:
            ChunkConfig: Configuration object
        """
        if preset_name not in cls.PRESET_CONFIGS:
            print(f"⚠️  Unknown preset configuration: {preset_name}, using default configuration")
            preset_name = "default"
        
        config = cls.PRESET_CONFIGS[preset_name]
        print(f"📋 Using preset configuration: {preset_name}")
        print(f"  - Split strategy: {config.split_by}")
        print(f"  - Chunk size: {config.chunk_size}")
        print(f"  - Overlap size: {config.chunk_overlap}")
        print(f"  - Long text mode: {'Yes' if config.long_text_mode else 'No'}")
        
        return config
    
    @classmethod
    def get_config_for_project_type(cls, project_type: str = "code") -> ChunkConfig:
        """
        Get configuration based on project type

        Args:
            project_type: Project type ('code', 'docs', 'long_text', 'academic', etc.)

        Returns:
            ChunkConfig: Configuration corresponding to project type
        """
        type_mapping = {
            'code': 'code_project',
            'project': 'code_project', 
            'docs': 'tech_docs',
            'documentation': 'tech_docs',
            'long_text': 'long_text',
            'novel': 'long_text',
            'book': 'long_text',
            'academic': 'academic',
            'paper': 'academic',
            'research': 'academic',
            'precise': 'precise',
            'context': 'large_context',
            'token': 'token_based',
            'llm': 'token_based'
        }
        
        preset = type_mapping.get(project_type.lower(), 'code_project')
        return cls.get_config(preset)
    
    @classmethod
    def create_custom_config(
        cls,
        base_preset: str = "default",
        **overrides
    ) -> ChunkConfig:
        """
        Create custom configuration based on preset configuration

        Args:
            base_preset: Base preset configuration name
            **overrides: Parameters to override

        Returns:
            ChunkConfig: Custom configuration object
        """
        config = cls.get_config(base_preset)
        
        # Apply override parameters
        for key, value in overrides.items():
            if hasattr(config, key):
                setattr(config, key, value)
                print(f"  ✏️  Override parameter {key}: {value}")
            else:
                print(f"  ⚠️  Unknown parameter {key}, ignored")
        
        return config
    
    @classmethod
    def list_presets(cls) -> None:
        """List all available preset configurations"""
        print("📋 Available preset configurations:")
        print("=" * 60)
        
        for name, config in cls.PRESET_CONFIGS.items():
            print(f"\n🔧 {name}:")
            print(f"  - Split strategy: {config.split_by}")
            print(f"  - Chunk size: {config.chunk_size}")
            print(f"  - Overlap: {config.chunk_overlap}")
            print(f"  - Max file: {config.max_file_size_mb}MB")
            print(f"  - Long text mode: {'Yes' if config.long_text_mode else 'No'}")
            
            if config.include_extensions:
                ext_preview = config.include_extensions[:5]
                ext_str = ', '.join(ext_preview)
                if len(config.include_extensions) > 5:
                    ext_str += f" ... (+{len(config.include_extensions) - 5} more)"
                print(f"  - Supported formats: {ext_str}")
        
        print(f"\n💡 Usage:")
        print(f"  - Python: ChunkConfigManager.get_config('preset_name')")
        print(f"  - Environment variable: export CHUNK_PRESET=preset_name")


def get_project_chunk_config(project_type: str = "code") -> ChunkConfig:
    """
    Convenient function to get project chunk configuration

    Args:
        project_type: Project type, default 'code'

    Returns:
        ChunkConfig: Project chunk configuration
    """
    return ChunkConfigManager.get_config_for_project_type(project_type)


def get_chunk_config_for_type(doc_type: str) -> ChunkConfig:
    """
    Get recommended configuration based on document type

    Args:
        doc_type: Document type ('code', 'long_text', 'academic', 'tech_docs', etc.)

    Returns:
        ChunkConfig: Recommended configuration
    """
    return ChunkConfigManager.get_config_for_project_type(doc_type)


if __name__ == "__main__":
    # Demonstrate configuration functionality
    print("🎯 Document chunker configuration demonstration\n")
    
    # List all presets
    ChunkConfigManager.list_presets()
    
    print(f"\n" + "=" * 60)
    print("🧪 Configuration test:")
    
    # 测试不同配置
    configs_to_test = ['default', 'code_project', 'long_text', 'academic']
    
    for config_name in configs_to_test:
        print(f"\n📋 Test configuration: {config_name}")
        config = ChunkConfigManager.get_config(config_name)
        print(f"  Configuration details: {config}")
    
    print(f"\n🔧 Custom configuration example:")
    custom_config = ChunkConfigManager.create_custom_config(
        'long_text',
        chunk_size=10,
        chunk_overlap=4,
        max_file_size_mb=300.0
    )
    print(f"  Custom result: {custom_config}")
    
    print(f"\n✅ Configuration demonstration completed!")