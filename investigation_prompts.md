# Optimized Investigation Prompts for Graph RAG Codebase

This document contains a suite of optimized, high-detail prompts designed to investigate the `graph_rag` codebase. These prompts are intended to be used by an AI assistant to verify implementation correctness, detect bugs, and ensure robust multi-language support.

## 1. Meta-Prompt (System Instruction)

Use this system instruction to set the context for the AI investigator:

> **Role**: You are a Senior Software Architect and Code Security Expert specializing in RAG systems, Graph Theory, and Compiler Design (Tree-sitter).
>
> **Objective**: Analyze the provided `graph_rag` codebase to verify logical correctness, identify potential bugs (logic, race conditions, edge cases), and validate support for all target languages (Solidity, Rust, C++, Move, Go, Python).
>
> **Constraints**:
> - Be extremely specific. Quote code line numbers.
> - Focus on *logic* and *implementation details*, not just syntax.
> - Verify that the implementation matches the requirements implied by the code structure.
> - Look for "silent failures" where code might run but produce incorrect results (e.g., incorrect graph edges, missing function calls).

---

## 2. Component-Specific Investigation Prompts

### 2.1. Project Parser & Language Support (`project_parser.py`)

**Prompt:**
> Analyze `project_parser.py` with a focus on **Multi-Language AST Extraction Correctness**.
>
> 1.  **Language-Specific Parsing**: Review the `_extract_functions_from_node` and `_parse_*_function` methods for ALL supported languages:
>     *   **Solidity**: Does it correctly handle modifiers, visibility keywords, and contract inheritance? Check `_parse_solidity_function`.
>     *   **Rust**: Verify handling of `impl` blocks, visibility (`pub`, `pub(crate)`), and complex return types (`Result<T, E>`). Check `_parse_rust_function`.
>     *   **C++**: Check function declaration vs. definition logic, namespace handling, and `const`/`static` modifiers. Check `_parse_cpp_function`.
>     *   **Move**: Verify `public(script)`, `public(friend)`, and `native` function handling. Check `_parse_move_function`.
>     *   **Go**: Does it correctly infer visibility from capitalization? Does it handle receiver methods (`(s *Struct) Method`) correctly? Check `_parse_go_function`.
>     *   **Python**: Verify decorator handling, async functions, and class method extraction. Check `_parse_python_function`.
>
> 2.  **Function Call Extraction**: Examine `_extract_function_calls` and `_get_function_call_name`.
>     *   Are chained calls (`obj.method().submethod()`) handled correctly for each language?
>     *   Are module-qualified calls (`Module::func` in Rust/Move, `pkg.Func` in Go) parsed accurately?
>     *   Identify any potential for infinite recursion in `traverse_for_calls`.
>
> 3.  **Edge Cases**:
>     *   What happens if a file has syntax errors? Does the parser crash or skip gracefully?
>     *   Are nested functions or lambdas handled?

### 2.2. Graph Construction Logic (`kg_implementation.py`)

**Prompt:**
> Analyze `kg_implementation.py` to verify the **Knowledge Graph Construction and Call Resolution Logic**.
>
> 1.  **Node Identity**: Check how `func_id` is constructed (`Function:{file_path}:{func_name}:{line_number}`).
>     *   Is this ID stable across different OS environments (Windows vs. Linux path separators)?
>     *   Could duplicate IDs be generated for overloaded functions in C++/Solidity?
>
> 2.  **Call Resolution Strategy** (`_resolve_call_target`):
>     *   Review the 6-step resolution strategy. Is the precedence order (Exact -> Self -> Class -> File -> Unique Short Name) logically sound?
>     *   **Potential Bug**: In Strategy 6 (Unique short name), what if `_short_name_index` contains stale data?
>     *   **Ambiguity**: How does it handle cases where multiple functions have the same short name in different files? Does it default to `None` or make a potentially wrong guess?
>
> 3.  **Graph Integrity**:
>     *   Are `DEFINES` edges correctly established between File -> Class -> Function?
>     *   Are `External` nodes created for *every* unresolved call? Is there a risk of bloating the graph with noise (e.g., standard library calls like `print`, `len`)?

### 2.3. Chunking & Enrichment (`chunk_enricher.py`, `document_chunker.py`)

**Prompt:**
> Investigate the **Chunking and Context Enrichment Mechanism**.
>
> 1.  **Enrichment Logic** (`chunk_enricher.py`):
>     *   Review `_find_node_id`. It attempts to match Master List functions to Graph nodes.
>     *   **Critical Check**: Does the path normalization (`replace('\\', '/')`) fully resolve cross-platform path issues?
>     *   Verify `_split_large_chunk`. Does it correctly maintain `chunk_id`, `parent_doc_id`, and `chunk_order`?
>     *   Check `_split_code_into_chunks`. Is the overlap logic (`step_size`) mathematically correct to prevent infinite loops or skipped content?
>
> 2.  **Document Chunking** (`document_chunker.py`):
>     *   Analyze `_preprocess_long_text`. Does it inadvertently destroy code indentation or structure when cleaning blank lines?
>     *   Check encoding handling in `_read_file_with_encoding`. Is the fallback list sufficient?

### 2.4. Pipeline Orchestration & Services (`pipeline.py`, `llm_service.py`, `embedding_service.py`)

**Prompt:**
> Analyze the **Pipeline Orchestration and External Service Integration**.
>
> 1.  **Pipeline Flow** (`pipeline.py`):
>     *   **State Management**: Check `_ensure_index_loaded`. Is there a race condition if `query()` is called immediately after `index()` without reloading?
>     *   **Error Handling**: Verify that `RetrievalError` or `GenerationError` are raised with sufficient context.
>     *   **Graceful Degradation**: In `query()`, if `rerank` fails, does it correctly fall back to raw results? If `expand` fails, does it continue?
>
> 2.  **Retry Logic** (`llm_service.py`, `embedding_service.py`):
>     *   Review `retry_with_exponential_backoff`.
>     *   **Bug Check**: Does it correctly catch *all* relevant exceptions? (e.g., `google.api_core.exceptions.ResourceExhausted`).
>     *   Is the `time.sleep(delay)` blocking the main thread acceptable?
>
> 3.  **Graph Expansion** (`graph_expander.py`):
>     *   Check `expand_with_depth`. Does it correctly handle cycles in the graph (A calls B calls A)?
>     *   Verify `exclude_ids` logic. Are primary results effectively excluded from expansion to avoid redundancy?

### 2.5. Vector Storage (`vector_store.py`)

**Prompt:**
> Analyze `vector_store.py` for **Data Persistence and Retrieval Correctness**.
>
> 1.  **Storage**:
>     *   Check the `store` method. It uses `mode="overwrite"`. Is this safe? Does it wipe previous embeddings for the entire project every time?
>     *   Are `node_id` and `filename` correctly stored as metadata?
>
> 2.  **Search**:
>     *   Verify `search`. Does it handle empty query embeddings or connection failures?
>     *   Check `get_by_node_id`. Is the SQL-like filter `where(f"node_id = '{node_id}'")` vulnerable to injection or formatting errors if `node_id` contains special characters?

---

## 3. Comprehensive "Deep Dive" Prompt

Use this prompt for a holistic review of the entire system:

**Prompt:**
> Perform a **Comprehensive Codebase Audit** of the `graph_rag` system.
>
> **Scope**: All provided files in `src/tree_sitter_parsing/graph_rag`.
>
> **Tasks**:
> 1.  **Trace the Data Flow**: Follow a piece of code from `ContentExtractor` -> `GraphBuilder` -> `ChunkEnricher` -> `VectorStore`. Identify any step where data might be lost or corrupted (e.g., encoding issues, path mismatches).
> 2.  **Verify Logic Consistency**:
>     *   Does the `FunctionDict` structure in `models.py` match what `project_parser.py` produces?
>     *   Does `ChunkEnricher` correctly interpret the graph structure built by `kg_implementation.py`?
> 3.  **Identify "Code Smells"**:
>     *   Hardcoded values (e.g., `DEFAULT_TOKEN_LIMIT`).
>     *   Duplicate logic (e.g., path normalization appearing in multiple places).
>     *   Swallowed exceptions (empty `except:` blocks).
> 4.  **Language Support Audit**:
>     *   Pick one complex language (e.g., C++ or Rust) and trace how its specific features (templates, macros, ownership) are handled throughout the pipeline.
>
> **Output Format**:
> *   **Critical Issues**: Bugs that cause failure or incorrect data.
> *   **Logical Flaws**: Design choices that might lead to poor RAG performance.
> *   **Language Support Gaps**: Missing features for specific languages.
> *   **Optimization Suggestions**: Concrete code improvements.
