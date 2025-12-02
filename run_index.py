"""
Wrapper script to run index_codebase.py from within the graph_rag package.
Sets up the Python path correctly for local imports.
"""
import sys
import os

# Add parent directory to path so 'graph_rag' package can be found
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Also ensure current directory is in path for direct imports
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Now we can import and run
if __name__ == "__main__":
    # Import after path setup
    from pipeline import GraphRAGPipeline
    from exceptions import (
        ExtractionError, GraphBuildError, EnrichmentError,
        EmbeddingError, StorageError
    )
    from datetime import datetime
    import argparse
    
    # Set API keys
    os.environ["VOYAGE_API_KEY"] = "pa-72D-aiLWPIFDM0jj3cbUP_de843HzPXVKGPnP5sYbnQ"
    os.environ["GOOGLE_API_KEY"] = "AIzaSyDZGk1eTArJrWpMx6FiKUZpXUrGIL9zs9w"
    
    parser = argparse.ArgumentParser(description="Index a codebase for Graph-Enhanced RAG")
    parser.add_argument(
        "project_path",
        nargs="?",
        default=".",
        help="Path to the project directory to index"
    )
    parser.add_argument(
        "--db-path",
        default="./lancedb_store",
        help="Path to LanceDB database directory"
    )
    parser.add_argument(
        "--data-dir",
        default="./rag_data",
        help="Path to store graph and metadata files"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("GRAPH-ENHANCED RAG - INDEXING PHASE")
    print("=" * 60)
    print(f"Project: {args.project_path}")
    print(f"Database: {args.db_path}")
    print(f"Data Dir: {args.data_dir}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    # Initialize pipeline
    pipeline = GraphRAGPipeline(
        db_path=args.db_path,
        data_dir=args.data_dir
    )
    
    try:
        print("\n[1/6] Extracting content (AST parsing)...")
        print("[2/6] Building knowledge graph...")
        print("[3/6] Enriching chunks with graph context...")
        print("[4/6] Generating embeddings (voyage-code-3)...")
        print("[5/6] Storing in LanceDB...")
        print("[6/6] Exporting graph and metadata...")
        
        result = pipeline.index(args.project_path)
        
        if result.success:
            print("\n" + "=" * 60)
            print("✓ INDEXING COMPLETED SUCCESSFULLY")
            print("=" * 60)
            print(f"  Functions extracted: {result.function_count}")
            print(f"  Graph nodes:         {result.node_count}")
            print(f"  Graph edges:         {result.edge_count}")
            print(f"  Enriched chunks:     {result.chunk_count}")
            print(f"  Embeddings:          {result.embedding_count}")
            print(f"\nFiles created:")
            print(f"  - {args.data_dir}/knowledge_graph.json")
            print(f"  - {args.data_dir}/master_list.json")
            print(f"  - {args.data_dir}/enriched_chunks.json")
            print(f"  - {args.db_path}/code_embeddings.lance/")
            print(f"\nFinished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print("\nYou can now run: python run_query.py")
        else:
            print(f"\n✗ Indexing failed: {result.message}")
            sys.exit(1)
            
    except ExtractionError as e:
        print(f"\n✗ Content extraction failed: {e}")
        sys.exit(1)
    except GraphBuildError as e:
        print(f"\n✗ Graph building failed: {e}")
        sys.exit(1)
    except EnrichmentError as e:
        print(f"\n✗ Chunk enrichment failed: {e}")
        sys.exit(1)
    except EmbeddingError as e:
        print(f"\n✗ Embedding generation failed: {e}")
        sys.exit(1)
    except StorageError as e:
        print(f"\n✗ Storage operation failed: {e}")
        sys.exit(1)
    except Exception as e:
        import traceback
        print(f"\n✗ Unexpected error: {e}")
        traceback.print_exc()
        sys.exit(1)
