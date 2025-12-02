"""
ContextAssembler for the Graph-Enhanced RAG System.

This module assembles the final prompt for LLM generation by combining
primary search results with graph-expanded secondary results.

Requirements: 8.1, 8.2
"""

from typing import List


class ContextAssembler:
    """
    Assembles context for LLM prompt.
    
    Combines primary vector search results with graph-expanded secondary
    results into a structured prompt format for the LLM.
    """
    
    # Section headers as specified in requirements
    PRIMARY_SECTION_HEADER = "--- PRIMARY MATCHES ---"
    SECONDARY_SECTION_HEADER = "--- RELATED CODE (Dependencies) ---"
    
    def __init__(self, system_instruction: str = None):
        """
        Initialize the ContextAssembler.
        
        Args:
            system_instruction: Optional custom system instruction for the prompt.
                              Defaults to a code analysis instruction.
        """
        self._system_instruction = system_instruction or (
            "You are a coding assistant. Answer the question based on the following context."
        )
    
    def assemble(
        self,
        primary_results: List[str],
        secondary_results: List[str],
        query: str
    ) -> str:
        """
        Assemble the final prompt for LLM generation.
        
        Args:
            primary_results: List of enriched chunk texts from vector search.
            secondary_results: List of code texts from graph expansion.
            query: The user's query.
            
        Returns:
            Assembled prompt string with system instruction, primary matches,
            related code, and the user query.
        """
        parts = []
        
        # Add system instruction
        parts.append(self._system_instruction)
        parts.append("")
        
        # Add PRIMARY MATCHES section
        parts.append(self.PRIMARY_SECTION_HEADER)
        if primary_results:
            for i, result in enumerate(primary_results, 1):
                parts.append(f"[Chunk {i}]")
                parts.append(result)
                parts.append("")
        else:
            parts.append("No primary matches found.")
            parts.append("")
        
        # Add RELATED CODE section
        parts.append(self.SECONDARY_SECTION_HEADER)
        if secondary_results:
            for i, result in enumerate(secondary_results, 1):
                parts.append(f"[Dep {i}]")
                parts.append(result)
                parts.append("")
        else:
            parts.append("No related code found.")
            parts.append("")
        
        # Add user query at the end
        parts.append(f"Question: {query}")
        
        return "\n".join(parts)
    
    def assemble_minimal(
        self,
        primary_results: List[str],
        query: str
    ) -> str:
        """
        Assemble a minimal prompt with only primary results (no secondary).
        
        Useful when graph expansion is disabled or fails.
        
        Args:
            primary_results: List of enriched chunk texts from vector search.
            query: The user's query.
            
        Returns:
            Assembled prompt string with primary matches and query.
        """
        return self.assemble(primary_results, [], query)
