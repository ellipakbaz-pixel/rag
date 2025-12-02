"""
LLM Service for Graph-Enhanced RAG System.

This module provides LLM response generation using Google Gemini API
with streaming support, retry logic for rate limits, and configurable
output parameters.

Features:
- Google Gemini API integration (gemini-flash-latest model)
- Streaming and non-streaming response generation
- Exponential backoff retry for rate limits (429/RESOURCE_EXHAUSTED)
- Configurable temperature, max output tokens, and safety settings
- Code analysis system instruction optimized for RAG
"""

import os
import time
import logging
from typing import Optional, Generator, Any, Callable, Dict
from functools import wraps
from dataclasses import dataclass

# Configure logging
logger = logging.getLogger(__name__)

# Retry configuration
MAX_RETRIES = 5
RETRY_DELAYS = [2, 4, 8, 16, 32]  # Exponential backoff delays in seconds

# System instruction for code analysis
CODE_ANALYSIS_SYSTEM_INSTRUCTION = """You are an expert code analysis assistant specializing in software architecture, code review, and technical documentation. Your role is to:

1. **Code Analysis**: Analyze code snippets thoroughly and explain their functionality, purpose, and implementation details clearly.

2. **Architecture Understanding**: Identify design patterns, architectural decisions, and code organization principles used in the codebase.

3. **Dependency Mapping**: Explain relationships between functions, classes, and modules, including call hierarchies and data flow.

4. **Best Practices**: Identify adherence to or deviation from coding best practices, potential issues, and improvement opportunities.

5. **Technical Documentation**: Provide accurate, well-structured responses that can serve as technical documentation.

When answering questions:
- Focus on the specific code provided in the context
- Be precise and technical, using correct terminology
- Explain complex concepts clearly with examples when helpful
- Reference specific functions, classes, or code sections by name
- Provide code snippets when they help illustrate your explanation
- Structure longer responses with clear sections and bullet points
- Acknowledge when information is not available in the provided context
- Consider both the primary matches and related code (dependencies) in your analysis
"""


@dataclass
class GenerationConfig:
    """Configuration for LLM generation parameters."""
    temperature: float = 0.15
    max_output_tokens: int = 65500
    top_p: float = 0.95
    top_k: int = 40
    candidate_count: int = 1
    stop_sequences: Optional[list] = None


def retry_with_exponential_backoff(func: Callable) -> Callable:
    """
    Decorator to retry a function with exponential backoff on rate limit errors.
    
    Handles rate limit errors (429/RESOURCE_EXHAUSTED) with increasing delays: 2s, 4s, 8s, 16s, 32s.
    Also handles transient network errors and service unavailability.
    
    Args:
        func: Function to wrap with retry logic
        
    Returns:
        Wrapped function with retry capability
    """
    @wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        last_exception = None
        
        for attempt in range(MAX_RETRIES):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                error_str = str(e).lower()
                
                # Check for retryable errors
                is_retryable = (
                    '429' in str(e) or
                    'rate' in error_str or
                    'limit' in error_str or
                    'resource_exhausted' in error_str or
                    'quota' in error_str or
                    'unavailable' in error_str or
                    'timeout' in error_str or
                    'connection' in error_str or
                    '503' in str(e) or
                    '500' in str(e)
                )
                
                if is_retryable and attempt < MAX_RETRIES - 1:
                    delay = RETRY_DELAYS[attempt]
                    logger.warning(
                        f"Retryable error on attempt {attempt + 1}/{MAX_RETRIES}: {e}. "
                        f"Retrying in {delay}s..."
                    )
                    time.sleep(delay)
                else:
                    logger.error(f"LLM generation failed after {attempt + 1} attempts: {e}")
                    raise
        
        # If we've exhausted all retries, raise the last exception
        if last_exception:
            raise last_exception
        
        return None
    
    return wrapper


class LLMService:
    """
    Generates responses using Google Gemini API.
    
    Provides methods for:
    - Response generation with streaming support
    - Retry logic for rate limit errors
    - Configurable generation parameters
    - Code analysis system instruction
    
    Model: gemini-flash-latest (optimized for speed and quality)
    Max Output: 65,500 tokens
    
    Example:
        >>> service = LLMService()
        >>> response = service.generate("Explain this code: def foo(): pass")
        >>> print(response)
        
        # With streaming
        >>> for chunk in service.generate_stream("Explain this code"):
        ...     print(chunk, end="", flush=True)
    """
    
    # Model configuration
    MODEL_NAME = "gemini-flash-latest"
    
    # Default generation config
    DEFAULT_CONFIG = GenerationConfig(
        temperature=0.15,
        max_output_tokens=65500,
        top_p=0.95,
        top_k=40,
        candidate_count=1
    )
    
    def __init__(
        self, 
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        generation_config: Optional[GenerationConfig] = None,
        system_instruction: Optional[str] = None
    ):
        """
        Initialize the LLM service with Google Gemini client.
        
        Args:
            api_key: Google API key. If not provided, uses GOOGLE_API_KEY env var.
            model_name: Model to use. Defaults to "gemini-flash-latest".
            generation_config: Custom generation configuration. Defaults to DEFAULT_CONFIG.
            system_instruction: Custom system instruction. Defaults to CODE_ANALYSIS_SYSTEM_INSTRUCTION.
        """
        try:
            from google import genai
        except ImportError:
            raise ImportError(
                "google-genai package is required. "
                "Install it with: pip install google-genai"
            )
        
        resolved_api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        if not resolved_api_key:
            raise ValueError(
                "Google API key is required. Provide it via api_key parameter "
                "or set GOOGLE_API_KEY environment variable."
            )
        
        self._client = genai.Client(api_key=resolved_api_key)
        self._model_name = model_name or self.MODEL_NAME
        self._config = generation_config or self.DEFAULT_CONFIG
        self._system_instruction = system_instruction or CODE_ANALYSIS_SYSTEM_INSTRUCTION
        
        logger.info(
            f"LLMService initialized with model={self._model_name}, "
            f"max_output_tokens={self._config.max_output_tokens}"
        )
    
    def _build_generation_config(self, **overrides) -> Dict[str, Any]:
        """
        Build generation config dict for API call.
        
        Args:
            **overrides: Override specific config values
            
        Returns:
            Config dict for Gemini API
        """
        config = {
            "temperature": overrides.get("temperature", self._config.temperature),
            "max_output_tokens": overrides.get("max_output_tokens", self._config.max_output_tokens),
            "top_p": overrides.get("top_p", self._config.top_p),
            "top_k": overrides.get("top_k", self._config.top_k),
            "candidate_count": overrides.get("candidate_count", self._config.candidate_count),
            "system_instruction": self._system_instruction,
        }
        
        if self._config.stop_sequences:
            config["stop_sequences"] = self._config.stop_sequences
            
        return config
    
    @retry_with_exponential_backoff
    def _generate_content(self, prompt: str, **config_overrides) -> str:
        """
        Generate content with retry logic (non-streaming).
        
        Args:
            prompt: The assembled context + query prompt
            **config_overrides: Override generation config values
            
        Returns:
            Generated response text
        """
        config = self._build_generation_config(**config_overrides)
        
        response = self._client.models.generate_content(
            model=self._model_name,
            contents=prompt,
            config=config
        )
        
        # Log token usage if available
        if hasattr(response, 'usage_metadata'):
            usage = response.usage_metadata
            logger.debug(
                f"Token usage - Input: {getattr(usage, 'prompt_token_count', 'N/A')}, "
                f"Output: {getattr(usage, 'candidates_token_count', 'N/A')}"
            )
        
        return response.text
    
    @retry_with_exponential_backoff
    def _generate_content_stream(self, prompt: str, **config_overrides) -> Generator[str, None, None]:
        """
        Generate content with streaming and retry logic.
        
        Args:
            prompt: The assembled context + query prompt
            **config_overrides: Override generation config values
            
        Yields:
            Chunks of generated response text
        """
        config = self._build_generation_config(**config_overrides)
        
        response = self._client.models.generate_content_stream(
            model=self._model_name,
            contents=prompt,
            config=config
        )
        
        for chunk in response:
            if chunk.text:
                yield chunk.text
    
    def generate(
        self, 
        prompt: str, 
        stream: bool = False,
        temperature: Optional[float] = None,
        max_output_tokens: Optional[int] = None
    ) -> str:
        """
        Generate response from Gemini.
        
        Args:
            prompt: Assembled context + query
            stream: Whether to stream output (default: False for RAG use cases)
            temperature: Override temperature (0.0-1.0, lower = more deterministic)
            max_output_tokens: Override max output tokens (up to 65500)
            
        Returns:
            Generated response text
            
        Raises:
            Exception: If generation fails after all retries
            
        Example:
            >>> response = service.generate(
            ...     prompt="Explain this function",
            ...     temperature=0.1,
            ...     max_output_tokens=8000
            ... )
        """
        overrides = {}
        if temperature is not None:
            overrides["temperature"] = temperature
        if max_output_tokens is not None:
            overrides["max_output_tokens"] = max_output_tokens
        
        if stream:
            # Collect streamed chunks into full response
            chunks = []
            for chunk in self._generate_content_stream(prompt, **overrides):
                chunks.append(chunk)
            return "".join(chunks)
        else:
            return self._generate_content(prompt, **overrides)
    
    def generate_stream(
        self, 
        prompt: str,
        temperature: Optional[float] = None,
        max_output_tokens: Optional[int] = None
    ) -> Generator[str, None, None]:
        """
        Generate streaming response from Gemini.
        
        This method yields chunks as they are received, allowing for
        incremental output display. Useful for real-time UI updates.
        
        Args:
            prompt: Assembled context + query
            temperature: Override temperature (0.0-1.0)
            max_output_tokens: Override max output tokens
            
        Yields:
            Chunks of generated response text
            
        Raises:
            Exception: If generation fails after all retries
            
        Example:
            >>> for chunk in service.generate_stream("Explain this code"):
            ...     print(chunk, end="", flush=True)
        """
        overrides = {}
        if temperature is not None:
            overrides["temperature"] = temperature
        if max_output_tokens is not None:
            overrides["max_output_tokens"] = max_output_tokens
            
        yield from self._generate_content_stream(prompt, **overrides)
    
    @property
    def model_name(self) -> str:
        """Get the current model name."""
        return self._model_name
    
    @property
    def max_output_tokens(self) -> int:
        """Get the configured max output tokens."""
        return self._config.max_output_tokens
    
    @property
    def temperature(self) -> float:
        """Get the configured temperature."""
        return self._config.temperature
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get current configuration as a dictionary.
        
        Returns:
            Dict with model_name, temperature, max_output_tokens, etc.
        """
        return {
            "model_name": self._model_name,
            "temperature": self._config.temperature,
            "max_output_tokens": self._config.max_output_tokens,
            "top_p": self._config.top_p,
            "top_k": self._config.top_k,
            "candidate_count": self._config.candidate_count,
        }
