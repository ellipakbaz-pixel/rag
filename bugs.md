# Bug Report: Graph-Enhanced RAG System

This document outlines the logical bugs, code issues, and potential vulnerabilities identified in the codebase.

## 1. Critical Logical Bugs

### 1.1 Code Destruction During Chunking
**File:** `chunk_enricher.py`
**Function:** `_split_code_into_chunks`

**Issue:** The function splits code using `code.split()` (which splits by whitespace) and then rejoins it with `' '.join()`.
```python
words = code.split()
# ...
chunk_text = ' '.join(chunk_words)
```
**Impact:** This destroys all indentation and newlines. For Python (and most other languages), this renders the code syntactically incorrect and unreadable. When the LLM tries to reason about this code, it will fail to understand the structure.

### 1.2 Broken Dependency: `tree-sitter-move`
**File:** `project_parser.py`

**Issue:** The module imports `tree_sitter_move` at the top level.
```python
import tree_sitter_move
```
**Impact:** The `tree-sitter-move` package in the current environment appears to have a binary incompatibility (invalid ELF header), causing the entire application to crash on startup. Since it's a top-level import, it prevents any part of the system from running, even if the user is not analyzing Move code.

### 1.3 Fragile Import System
**File:** `project_parser.py`

**Issue:** The module imports all supported language parsers at the top level.
```python
import tree_sitter_solidity
import tree_sitter_rust
# ...
```
**Impact:** If *any* of the language bindings are missing or broken (as seen with `tree_sitter_move`), the entire module fails to load. This makes the system extremely fragile and hard to deploy.

## 2. Security Issues

### 2.1 Hardcoded API Keys
**Files:** `run_index.py`, `run_query.py`

**Issue:** API keys are hardcoded in the scripts and explicitly override environment variables.
```python
os.environ["VOYAGE_API_KEY"] = "pa-72D-..."
os.environ["GOOGLE_API_KEY"] = "AIzaSyDZ..."
```
**Impact:** Committing active credentials to version control is a major security risk. Users running these scripts might inadvertently use the author's quota or expose the keys further.

### 2.2 SQL Injection / Escape Logic
**File:** `vector_store.py`

**Issue:** The `get_by_node_id` method constructs a SQL-like filter string manually.
```python
results = table.search().where(f"node_id = '{escaped_node_id}'").limit(1).to_list()
```
While `_escape_node_id` handles single quotes and backslashes, manual string interpolation for queries is generally a bad practice and could be vulnerable if the escaping logic is flawed or if the underlying engine behaves unexpectedly.

## 3. Code Quality & Reliability Issues

### 3.1 Relative Import Errors
**File:** `content_extractor.py`, `graph_builder.py`

**Issue:** The modules use relative imports inside `try-except` blocks to handle being run as a package vs script.
```python
try:
    from .project_parser import parse_project
except ImportError:
    # Fallback
```
**Impact:** This pattern masks genuine `ImportError`s within the imported modules (like the `tree_sitter_move` failure), making debugging extremely difficult. The error message simply says "attempted relative import..." instead of the actual root cause.

### 3.2 Inconsistent Edge Attributes
**Files:** `chunk_enricher.py` vs `tests`

**Issue:** The code in `chunk_enricher.py` and `graph_expander.py` expects edge attributes to have a `relation` key (e.g., `relation='DEFINES'`), while standard NetworkX usage or test setups might use `type` or other keys.
**Impact:** If the graph builder doesn't strictly adhere to this convention (it seems it does in `kg_implementation.py`), graph traversal will fail silently (return empty lists), as seen in the test failures.

### 3.3 Path Normalization Risks
**File:** `chunk_enricher.py`, `kg_implementation.py`

**Issue:** The system relies heavily on `normalize_path` to generate stable Node IDs.
**Impact:** If `normalize_path` (which likely uses `os.path.abspath`) resolves paths differently in different contexts (e.g., inside/outside Docker, with symlinks), the Node IDs will mismatch, leading to broken links between functions and chunks.

### 3.4 Missing Error Handling for File Parsing
**File:** `project_parser.py`

**Issue:** While there is a `try-except` block inside the loop, the top-level language detection or parser initialization could fail.
**Impact:** A single malformed file or unexpected file permission issue could potentially disrupt the indexing process if not handled at the right granularity.

### 3.5 Aggressive Retry Logic
**File:** `embedding_service.py`

**Issue:** The `@retry_with_exponential_backoff` decorator retries on *all* exceptions.
**Impact:** This includes non-transient errors (like `AuthenticationError` or `InvalidRequestError`). Retrying these will not solve the problem and will just delay the failure, potentially hitting rate limits harder.

## 4. Test Failures
During the audit, the following tests failed:

1.  **`test_chunk_enricher`**: `chunk.defined_in` was empty.
    *   **Cause**: Test setup used `type="DEFINES"` but code expects `relation="DEFINES"`.
2.  **`test_graph_expander`**: Expansion returned no results.
    *   **Cause**: Test setup used `type="CALLS"` but code expects `relation="CALLS"`.

## 5. Recommendations

1.  **Refactor Imports**: Use lazy imports or `importlib` for optional dependencies (like language parsers). Handle `ImportError` gracefully for individual languages.
2.  **Fix Chunking**: Use a proper text splitter (like `RecursiveCharacterTextSplitter` from LangChain or similar) that respects code structure, instead of `split()` and `join()`.
3.  **Remove Credentials**: Delete hardcoded API keys and rely solely on environment variables or configuration files.
4.  **Standardize Graph Attributes**: Define constants for edge attributes (e.g., `EDGE_RELATION = 'relation'`) to ensure consistency between builder and consumer components.
5.  **Robust Error Handling**: Only retry on specific transient exceptions (network, 5xx, 429). Fail fast on auth or 400 errors.
