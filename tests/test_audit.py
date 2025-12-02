
import unittest
import os
import shutil
import tempfile
import json
import networkx as nx
from unittest.mock import MagicMock, patch
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import FunctionDict, EnrichedChunk
from content_extractor import ContentExtractor
from graph_builder import GraphBuilder
from chunk_enricher import ChunkEnricher
from vector_store import VectorStore
from graph_expander import GraphExpander
from exceptions import ExtractionError, GraphBuildError

class TestGraphRAG(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory for test data
        self.test_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.test_dir, "lancedb_test")

        # Mock data
        self.mock_function = {
            "name": "MyClass.my_method",
            "content": "def my_method():\n    pass",
            "file_path": "/path/to/file.py",
            "start_line": 10,
            "end_line": 15,
            "calls": ["other_func"],
            "contract_name": "MyClass",
            "visibility": "public"
        }

        self.mock_chunk = EnrichedChunk(
            file_path="/path/to/file.py",
            function_name="MyClass.my_method",
            node_type="Function",
            node_id="Function:/path/to/file.py:my_method:10",
            defined_in="MyClass",
            calls=["other_func"],
            called_by=["main"],
            code="def my_method():\n    pass"
        )

    def tearDown(self):
        # Clean up temporary directory
        shutil.rmtree(self.test_dir)

    def test_content_extractor_init(self):
        """Test ContentExtractor initialization."""
        extractor = ContentExtractor()
        self.assertIsNotNone(extractor)

    @patch('content_extractor.parse_project')
    def test_content_extractor_extract(self, mock_parse):
        """Test content extraction logic."""
        mock_parse.return_value = ([self.mock_function], [self.mock_function], [])

        extractor = ContentExtractor()
        # Create a dummy directory to pass validation
        dummy_dir = os.path.join(self.test_dir, "dummy_project")
        os.makedirs(dummy_dir)

        functions = extractor.extract(dummy_dir)
        self.assertEqual(len(functions), 1)
        self.assertEqual(functions[0]["name"], "MyClass.my_method")

    def test_graph_builder_init(self):
        """Test GraphBuilder initialization."""
        builder = GraphBuilder()
        self.assertIsNotNone(builder)

    @patch('graph_builder.KnowledgeGraphBuilder')
    def test_graph_builder_build(self, MockKGBuilder):
        """Test graph building logic."""
        mock_kg = MockKGBuilder.return_value
        mock_kg.graph = nx.DiGraph()
        mock_kg.graph.add_node("node1")

        builder = GraphBuilder()
        dummy_dir = os.path.join(self.test_dir, "dummy_project")
        os.makedirs(dummy_dir)

        graph = builder.build(dummy_dir)
        self.assertIsInstance(graph, nx.DiGraph)
        self.assertTrue(graph.has_node("node1"))

    def test_chunk_enricher(self):
        """Test chunk enrichment."""
        graph = nx.DiGraph()
        # Add nodes
        func_node_id = "Function:/path/to/file.py:my_method:10"
        class_node_id = "Class:/path/to/file.py:MyClass"
        called_node_id = "Function:/path/to/other.py:other_func:5"
        caller_node_id = "Function:/path/to/main.py:main:20"

        graph.add_node(func_node_id, type="Function", name="my_method", file_path="/path/to/file.py")
        graph.add_node(class_node_id, type="Class/Contract", name="MyClass")
        graph.add_node(called_node_id, type="Function", name="other_func")
        graph.add_node(caller_node_id, type="Function", name="main")

        # Add edges
        graph.add_edge(class_node_id, func_node_id, type="DEFINES")
        graph.add_edge(func_node_id, called_node_id, type="CALLS")
        graph.add_edge(caller_node_id, func_node_id, type="CALLS")

        master_list = [self.mock_function]

        enricher = ChunkEnricher(graph, master_list)
        chunks = enricher.enrich_all()

        self.assertEqual(len(chunks), 1)
        chunk = chunks[0]
        self.assertEqual(chunk.function_name, "MyClass.my_method")
        self.assertEqual(chunk.defined_in, "MyClass")
        self.assertIn("other_func", chunk.calls)
        self.assertIn("main", chunk.called_by)

    def test_vector_store(self):
        """Test vector store operations."""
        store = VectorStore(db_path=self.db_path)

        chunks = [self.mock_chunk]
        embeddings = [[0.1] * 2048]

        # Test storage
        store.store(chunks, embeddings, table_name="test_table")

        # Test retrieval
        retrieved_text = store.get_by_node_id(self.mock_chunk.node_id, table_name="test_table")
        self.assertIsNotNone(retrieved_text)
        self.assertIn("def my_method", retrieved_text)

        # Test search
        results = store.search([0.1] * 2048, limit=1, table_name="test_table")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].node_id, self.mock_chunk.node_id)

    def test_graph_expander(self):
        """Test graph expansion."""
        graph = nx.DiGraph()
        node_a = "Function:A"
        node_b = "Function:B"
        node_c = "Function:C"

        graph.add_node(node_a, type="Function")
        graph.add_node(node_b, type="Function")
        graph.add_node(node_c, type="Function")

        graph.add_edge(node_a, node_b, type="CALLS")
        graph.add_edge(node_c, node_a, type="CALLS") # C calls A (incoming to A)

        master_list_map = {
            node_b: {"content": "code_b"},
            node_c: {"content": "code_c"}
        }

        expander = GraphExpander(graph, master_list_map, {})

        # Expand around A
        results = expander.expand([node_a], exclude_ids={node_a})

        # Should find B (outgoing) and C (incoming)
        found_ids = [r.node_id for r in results]
        self.assertIn(node_b, found_ids)
        self.assertIn(node_c, found_ids)

        # Check relationships
        for res in results:
            if res.node_id == node_b:
                self.assertEqual(res.relationship, "calls")
            elif res.node_id == node_c:
                self.assertEqual(res.relationship, "called by")

if __name__ == '__main__':
    unittest.main()
