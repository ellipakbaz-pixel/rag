# Graph-Enhanced RAG Query Report

**Generated:** 2025-12-02 21:31:10

---

## Query

```
How does the call resolution work?
```

---

## Summary

| Metric | Value |
|--------|-------|
| Vector Search Results | 3 |
| Reranked Results | 3 |
| Graph-Expanded Context | 7 |
| Total Context Chunks | 10 |

---

## LLM Response

The call resolution mechanism is implemented within the `KnowledgeGraphBuilder` class, primarily in the private method `_resolve_call_target`. This function is crucial for mapping raw call strings extracted by the parser to specific, known function node IDs within the project's knowledge graph.

If resolution fails, the calling function, `_add_calls_to_graph`, handles the creation of an `External` node to represent the unresolved dependency.

---

## 1. The Call Resolution Process (`_resolve_call_target`)

The `_resolve_call_target` function attempts to resolve a `call_target_name` using six distinct strategies, applied sequentially based on decreasing specificity and increasing reliance on contextual information.

### Inputs

| Parameter | Description | Purpose |
| :--- | :--- | :--- |
| `call_target_name` | The raw string of the function being called (e.g., `self.method`, `module.Class.func`, or `short_name`). | The target to be resolved. |
| `caller_file_path` | The normalized file path of the function making the call. | Context for file-local resolution. |
| `caller_class` | The class or contract name of the function making the call (if applicable). | Context for class/method resolution. |

### Resolution Strategies (In Order of Precedence)

#### Strategy 1: Exact Match (Full Qualified Name)
The system first checks if the raw `call_target_name` already matches a known function ID exactly. This handles cases where the parser provides a fully qualified name (FQN).
```python
if call_target_name in self.function_nodes:
    return self.function_nodes[call_target_name]
```

#### Strategy 2: `self.method` Calls (Intra-Class Resolution)
If the call starts with `'self.'` (common in Python and similar languages), the system assumes it is a method call within the current class context.

1.  It extracts the `method_name` by removing `'self.'`.
2.  It performs a direct lookup using `(caller_class, method_name)` in `self._class_method_index`.
3.  If the direct lookup fails, it iterates through all methods in the index, attempting to match the class names while accounting for potential module prefixes (e.g., matching `ClassA` against `module.ClassA` or vice versa). (Validated by Dep 4 and Dep 5).

#### Strategy 3: `Class.method` or `module.function` Calls (Two-Part Qualified Name)
If the name contains a dot (`.`) and splits into exactly two parts (`class_or_module`, `method_name`):

1.  It attempts a direct lookup in `self._class_method_index` using `(class_or_module, method_name)`.
2.  If that fails, it iterates through the index, checking if the stored class name either matches `class_or_module` exactly or ends with it (handling cases where the stored name is fully qualified, like `module.Class`, but the call target only provided `Class`).

#### Strategy 4: Short Name in Same Class
After handling qualified names, the system extracts the `short_name` (the part after the last dot). If a `caller_class` is available, it checks for a method with that `short_name` within the caller's class context using `self._class_method_index`. This strategy takes precedence over same-file resolution when both apply. (Validated by Dep 2).

#### Strategy 5: Short Name in Same File
The system checks for a function with the `short_name` defined directly in the `caller_file_path` using `self._file_func_index`.

#### Strategy 6: Unique Short Name Across Project (Global Lookup)
If the call remains unresolved, the system performs a global lookup using `self._short_name_index`.

1.  **Unique Match:** If only one function matches the `short_name` globally, that function is selected.
2.  **Ambiguous Match (Tie-breaking):** If multiple functions share the `short_name`, the system applies preference rules:
    *   **Preference 1:** Prefer the function located in the `caller_file_path`. (Validated by Dep 1).
    *   **Preference 2:** If still ambiguous, prefer the function located in the `caller_class`. (Validated by Dep 2).
3.  **Failure:** If multiple matches remain after applying file and class preferences, the call is considered **ambiguous**. The function logs a debug message detailing the ambiguity and returns `None`. (Validated by Dep 3).

If all six strategies fail, `_resolve_call_target` returns `None`.

---

## 2. Integration and Handling Unresolved Calls

The `_resolve_call_target` function is called by `kg_implementation.KnowledgeGraphBuilder._add_calls_to_graph` (Chunk 3) during the second pass of graph construction (Pass 2: Create Edges), as orchestrated by `process_project` (Dep 7).

### Internal Call Resolution

If `target_id` is successfully resolved (i.e., not `None`):
1.  The call is identified as an **Internal call**.
2.  An edge with the relation `"CALLS"` is added between the `source_id` (the calling function) and the `target_id` (the resolved function node).
3.  Self-loops (`target_id == source_id`) are explicitly avoided.

### External/Unresolved Call Handling

If `target_id` is `None` (meaning the call was ambiguous or points outside the analyzed project):
1.  The call is treated as an **External or unresolved call**.
2.  A unique `external_id` is generated: `f"External:{call_target_name}"`.
3.  If this `External` node does not already exist in the graph, it is created with `type="External"` and its `name` set to the original `call_target_name`.
4.  An edge with the relation `"CALLS"` is added from the `source_id` to this newly created or existing `external_id`.

---

## Detailed Results

### Primary Matches (Vector Search + Reranking)

These are the most relevant code chunks found via semantic search and reranked for relevance.

#### [1] kg_implementation.KnowledgeGraphBuilder._resolve_call_target

**File:** `./kg_implementation.py`

**Node ID:** `Function:./kg_implementation.py:kg_implementation.KnowledgeGraphBuilder._resolve_call_target:158`

**Distance:** 0.7356

**Context:**
- Defined In: `kg_implementation.KnowledgeGraphBuilder`
- Calls: `call_target_name.strip`, `clean_name.startswith`, `self._class_method_index.get`, `self._class_method_index.items`, `caller_class.endswith`, `class_name.endswith`, `caller_class.split`, `class_name.split`, `clean_name.split`, `len`, `self._file_func_index.get`, `logger.debug`
- Called By: `kg_implementation.KnowledgeGraphBuilder._add_calls_to_graph`

**Code:**
```python
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
```

---

#### [2] test_call_resolution.MockKnowledgeGraphBuilder._resolve_call_target

**File:** `./tests/test_call_resolution.py`

**Node ID:** `Function:./tests/test_call_resolution.py:test_call_resolution.MockKnowledgeGraphBuilder._resolve_call_target:74`

**Distance:** 0.8399

**Context:**
- Defined In: `test_call_resolution.MockKnowledgeGraphBuilder`
- Calls: `call_target_name.strip`, `clean_name.startswith`, `self._class_method_index.get`, `self._class_method_index.items`, `caller_class.endswith`, `class_name.endswith`, `caller_class.split`, `class_name.split`, `clean_name.split`, `len`, `self._file_func_index.get`
- Called By: `test_call_resolution.TestCallResolutionSameFilePreference.test_property_3_same_file_preference`, `test_call_resolution.TestCallResolutionSameClassPreference.test_property_4_same_class_preference`, `test_call_resolution.TestAmbiguousCallCreatesExternalNode.test_property_5_ambiguous_call_returns_none`, `test_call_resolution.TestSelfMethodResolution.test_property_6_self_method_resolution`, `test_call_resolution.TestSelfMethodResolution.test_property_6_self_method_with_module_prefix`

**Code:**
```python
def _resolve_call_target(self, call_target_name: str, caller_file_path: str, 
                             caller_class: Optional[str]) -> Optional[str]:
        """
        Resolve a call target name to a function node ID.
        
        This is a copy of the resolution logic from kg_implementation.py.
        """
        # Strategy 1: Exact match with full qualified name
        if call_target_name in self.function_nodes:
            return self.function_nodes[call_target_name]
        
        clean_name = call_target_name.strip()
        
        # Strategy 2: Handle self.method calls (Python style)
        if clean_name.startswith('self.'):
            method_name = clean_name[5:]
            if caller_class:
                target_id = self._class_method_index.get((caller_class, method_name))
                if target_id:
                    return target_id
                
                # Try with module prefix variations
                for (class_name, meth_name), func_id in self._class_method_index.items():
                    if meth_name == method_name:
                        if class_name == caller_class:
                            return func_id
                        if caller_class.endswith('.' + class_name):
                            return func_id
                        if class_name.endswith('.' + caller_class):
                            return func_id
                        caller_short = caller_class.split('.')[-1] if '.' in caller_class else caller_class
                        class_short = class_name.split('.')[-1] if '.' in class_name else class_name
                        if caller_short == class_short:
                            return func_id
        
        # Strategy 3: Handle Class.method or module.function calls
        if '.' in clean_name:
            parts = clean_name.split('.')
            if len(parts) == 2:
                class_or_module, method_name = parts
                target_id = self._class_method_index.get((class_or_module, method_name))
                if target_id:
                    return target_id
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
                return matches[0][1]
            elif len(matches) > 1:
                # Multiple matches - prefer same file, then same class
                for full_name, func_id, fpath, cname in matches:
                    if fpath == caller_file_path:
                        return func_id
                for full_name, func_id, fpath, cname in matches:
                    if cname and caller_class and cname == caller_class:
                        return func_id
                # Ambiguous: return None to create External node
                return None
        
        return None
```

---

#### [3] kg_implementation.KnowledgeGraphBuilder._add_calls_to_graph

**File:** `./kg_implementation.py`

**Node ID:** `Function:./kg_implementation.py:kg_implementation.KnowledgeGraphBuilder._add_calls_to_graph:273`

**Distance:** 0.8853

**Context:**
- Defined In: `kg_implementation.KnowledgeGraphBuilder`
- Calls: `func.get`, `kg_implementation.normalize_path`, `kg_implementation.KnowledgeGraphBuilder._resolve_call_target`, `self.graph.add_edge`, `self.graph.add_node`
- Called By: `kg_implementation.KnowledgeGraphBuilder.process_project`

**Code:**
```python
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
```

---

### Graph-Expanded Context (Related Code)

These are related code chunks discovered by traversing the knowledge graph (CALLS/CALLED_BY relationships).

#### [Dep 1] test_call_resolution.TestCallResolutionSameFilePreference.test_property_3_same_file_preference

**File:** `./tests/test_call_resolution.py`

**Relationship:** Graph neighbor (CALLS/CALLED_BY)

**Context:**
- Defined In: `test_call_resolution.TestCallResolutionSameFilePreference`
- Calls: `assume`, `test_call_resolution.normalize_path`, `len`, `MockKnowledgeGraphBuilder`, `test_call_resolution.MockKnowledgeGraphBuilder.add_function`, `test_call_resolution.MockKnowledgeGraphBuilder._resolve_call_target`
- Called By: None

**Code:**
```python
def test_property_3_same_file_preference(self, file1: str, file2: str, 
                                              func_name: str, line1: int, line2: int):
        """
        Property 3: Call Resolution Same-File Preference
        
        *For any* call resolution scenario where multiple functions share the same
        short name, if one function is in the same file as the caller, that function
        SHALL be selected.
        
        **Feature: graph-rag-bug-investigation, Property 3: Call Resolution Same-File Preference**
        **Validates: Requirements 2.1**
        """
        # Ensure files are different
        assume(normalize_path(file1) != normalize_path(file2))
        assume(len(func_name) > 0)
        
        builder = MockKnowledgeGraphBuilder()
        
        # Add same-named function in two different files
        func_id_file1 = builder.add_function(func_name, file1, line1)
        func_id_file2 = builder.add_function(func_name, file2, line2)
        
        # Resolve from file1 - should prefer function in file1
        resolved = builder._resolve_call_target(func_name, normalize_path(file1), None)
        
        assert resolved == func_id_file1, (
            f"Expected same-file function {func_id_file1}, got {resolved}"
        )
        
        # Resolve from file2 - should prefer function in file2
        resolved = builder._resolve_call_target(func_name, normalize_path(file2), None)
        
        assert resolved == func_id_file2, (
            f"Expected same-file function {func_id_file2}, got {resolved}"
        )
```

---

#### [Dep 2] test_call_resolution.TestCallResolutionSameClassPreference.test_property_4_same_class_preference

**File:** `./tests/test_call_resolution.py`

**Relationship:** Graph neighbor (CALLS/CALLED_BY)

**Context:**
- Defined In: `test_call_resolution.TestCallResolutionSameClassPreference`
- Calls: `assume`, `len`, `MockKnowledgeGraphBuilder`, `test_call_resolution.normalize_path`, `test_call_resolution.MockKnowledgeGraphBuilder.add_function`, `test_call_resolution.MockKnowledgeGraphBuilder._resolve_call_target`
- Called By: None

**Code:**
```python
def test_property_4_same_class_preference(self, file_path: str, class1: str, 
                                               class2: str, method_name: str,
                                               line1: int, line2: int):
        """
        Property 4: Call Resolution Same-Class Preference
        
        *For any* call resolution scenario where multiple functions share the same
        short name in different classes, if one function is in the same class as
        the caller, that function SHALL be selected.
        
        **Feature: graph-rag-bug-investigation, Property 4: Call Resolution Same-Class Preference**
        **Validates: Requirements 2.2**
        """
        # Ensure classes are different
        assume(class1 != class2)
        assume(len(method_name) > 0)
        
        builder = MockKnowledgeGraphBuilder()
        file_path = normalize_path(file_path)
        
        # Add same-named method in two different classes (same file)
        func_id_class1 = builder.add_function(
            f"{class1}.{method_name}", file_path, line1, class1
        )
        func_id_class2 = builder.add_function(
            f"{class2}.{method_name}", file_path, line2, class2
        )
        
        # Resolve from class1 - should prefer method in class1
        resolved = builder._resolve_call_target(method_name, file_path, class1)
        
        assert resolved == func_id_class1, (
            f"Expected same-class method {func_id_class1}, got {resolved}"
        )
        
        # Resolve from class2 - should prefer method in class2
        resolved = builder._resolve_call_target(method_name, file_path, class2)
        
        assert resolved == func_id_class2, (
            f"Expected same-class method {func_id_class2}, got {resolved}"
        )
```

---

#### [Dep 3] test_call_resolution.TestAmbiguousCallCreatesExternalNode.test_property_5_ambiguous_call_returns_none

**File:** `./tests/test_call_resolution.py`

**Relationship:** Graph neighbor (CALLS/CALLED_BY)

**Context:**
- Defined In: `test_call_resolution.TestAmbiguousCallCreatesExternalNode`
- Calls: `test_call_resolution.normalize_path`, `assume`, `len`, `MockKnowledgeGraphBuilder`, `test_call_resolution.MockKnowledgeGraphBuilder.add_function`, `test_call_resolution.MockKnowledgeGraphBuilder._resolve_call_target`
- Called By: None

**Code:**
```python
def test_property_5_ambiguous_call_returns_none(self, file1: str, file2: str,
                                                     caller_file: str, func_name: str,
                                                     line1: int, line2: int):
        """
        Property 5: Ambiguous Call Creates External Node
        
        *For any* call target that cannot be uniquely resolved (multiple matches,
        none in same file/class), the system SHALL create an External node rather
        than selecting arbitrarily.
        
        **Feature: graph-rag-bug-investigation, Property 5: Ambiguous Call Creates External Node**
        **Validates: Requirements 2.3**
        """
        # Ensure all files are different
        file1 = normalize_path(file1)
        file2 = normalize_path(file2)
        caller_file = normalize_path(caller_file)
        
        assume(file1 != file2)
        assume(caller_file != file1)
        assume(caller_file != file2)
        assume(len(func_name) > 0)
        
        builder = MockKnowledgeGraphBuilder()
        
        # Add same-named function in two different files (neither is caller's file)
        builder.add_function(func_name, file1, line1)
        builder.add_function(func_name, file2, line2)
        
        # Resolve from a third file - should return None (ambiguous)
        resolved = builder._resolve_call_target(func_name, caller_file, None)
        
        assert resolved is None, (
            f"Expected None for ambiguous call, got {resolved}"
        )
```

---

#### [Dep 4] test_call_resolution.TestSelfMethodResolution.test_property_6_self_method_resolution

**File:** `./tests/test_call_resolution.py`

**Relationship:** Graph neighbor (CALLS/CALLED_BY)

**Context:**
- Defined In: `test_call_resolution.TestSelfMethodResolution`
- Calls: `assume`, `len`, `MockKnowledgeGraphBuilder`, `test_call_resolution.normalize_path`, `test_call_resolution.MockKnowledgeGraphBuilder.add_function`, `test_call_resolution.MockKnowledgeGraphBuilder._resolve_call_target`
- Called By: None

**Code:**
```python
def test_property_6_self_method_resolution(self, file_path: str, class_name: str,
                                                method_name: str, line_number: int):
        """
        Property 6: Self-Method Resolution
        
        *For any* `self.method` call within a class, the resolution SHALL return
        the method defined in the caller's class.
        
        **Feature: graph-rag-bug-investigation, Property 6: Self-Method Resolution**
        **Validates: Requirements 2.4**
        """
        assume(len(method_name) > 0)
        
        builder = MockKnowledgeGraphBuilder()
        file_path = normalize_path(file_path)
        
        # Add method to the class
        func_id = builder.add_function(
            f"{class_name}.{method_name}", file_path, line_number, class_name
        )
        
        # Resolve self.method from within the class
        call_target = f"self.{method_name}"
        resolved = builder._resolve_call_target(call_target, file_path, class_name)
        
        assert resolved == func_id, (
            f"Expected {func_id} for self.{method_name}, got {resolved}"
        )
```

---

#### [Dep 5] test_call_resolution.TestSelfMethodResolution.test_property_6_self_method_with_module_prefix

**File:** `./tests/test_call_resolution.py`

**Relationship:** Graph neighbor (CALLS/CALLED_BY)

**Context:**
- Defined In: `test_call_resolution.TestSelfMethodResolution`
- Calls: `assume`, `len`, `MockKnowledgeGraphBuilder`, `test_call_resolution.normalize_path`, `test_call_resolution.MockKnowledgeGraphBuilder.add_function`, `test_call_resolution.MockKnowledgeGraphBuilder._resolve_call_target`
- Called By: None

**Code:**
```python
def test_property_6_self_method_with_module_prefix(self, file_path: str, 
                                                        module_name: str,
                                                        class_name: str,
                                                        method_name: str, 
                                                        line_number: int):
        """
        Property 6 (extended): Self-Method Resolution with Module Prefix
        
        *For any* `self.method` call where the class name has a module prefix,
        the resolution SHALL still return the correct method.
        
        **Feature: graph-rag-bug-investigation, Property 6: Self-Method Resolution**
        **Validates: Requirements 2.4**
        """
        assume(len(method_name) > 0)
        assume(len(module_name) > 0)
        
        builder = MockKnowledgeGraphBuilder()
        file_path = normalize_path(file_path)
        
        # Full class name with module prefix
        full_class_name = f"{module_name}.{class_name}"
        
        # Add method with module-prefixed class name
        func_id = builder.add_function(
            f"{full_class_name}.{method_name}", file_path, line_number, full_class_name
        )
        
        # Resolve self.method from within the class (using full class name)
        call_target = f"self.{method_name}"
        resolved = builder._resolve_call_target(call_target, file_path, full_class_name)
        
        assert resolved == func_id, (
            f"Expected {func_id} for self.{method_name} with module prefix, got {resolved}"
        )
```

---

#### [Dep 6] kg_implementation.normalize_path

**File:** `./kg_implementation.py`

**Relationship:** Graph neighbor (CALLS/CALLED_BY)

**Context:**
- Defined In: `kg_implementation`
- Calls: `path.replace`
- Called By: `kg_implementation.KnowledgeGraphBuilder._add_function_to_graph`, `kg_implementation.KnowledgeGraphBuilder._add_calls_to_graph`

**Code:**
```python
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
```

---

#### [Dep 7] kg_implementation.KnowledgeGraphBuilder.process_project

**File:** `./kg_implementation.py`

**Relationship:** Graph neighbor (CALLS/CALLED_BY)

**Context:**
- Defined In: `kg_implementation.KnowledgeGraphBuilder`
- Calls: `print`, `TreeSitterProjectFilter`, `project_parser.parse_project`, `len`, `kg_implementation.KnowledgeGraphBuilder._add_function_to_graph`, `kg_implementation.KnowledgeGraphBuilder._add_calls_to_graph`
- Called By: `graph_builder.GraphBuilder.build`, `kg_implementation.main`

**Code:**
```python
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
```

---

## Raw Data

### Search Results Node IDs

```
Function:./tests/test_call_resolution.py:test_call_resolution.MockKnowledgeGraphBuilder._resolve_call_target:74
Function:./kg_implementation.py:kg_implementation.KnowledgeGraphBuilder._resolve_call_target:158
Function:./tests/test_call_resolution.py:test_call_resolution.TestCallResolutionSameClassPreference.test_property_4_same_class_preference:262
```

### Graph Statistics

- Total Graph Nodes: 781
- Total Graph Edges: 1621
- Indexed Functions: 286
- Enriched Chunks: 286

---

*Report generated by Graph-Enhanced RAG System*
