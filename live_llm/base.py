"""
Live LLM Base - Abstract base class for live streaming LLMs with hot KV cache
"""
import asyncio
from abc import ABC, abstractmethod
from typing import AsyncGenerator, Optional
import threading
import queue


class LiveLLMBase(ABC):
    """Abstract base class for live streaming LLMs with persistent KV cache."""
    
    def __init__(self):
        self._input_queue = asyncio.Queue()
        self._output_queue = asyncio.Queue()
        self._processing_task: Optional[asyncio.Task] = None
        self._is_running = False
        self._kv_cache = None  # Model-specific cache storage
        
    @abstractmethod
    async def initialize_model(self) -> None:
        """Load and prepare the model for inference."""
        pass
    
    @abstractmethod
    async def reset(self) -> None:
        """Clear/reset the internal KV cache completely."""
        pass
    
    @abstractmethod
    async def _process_tokens(self, input_text: str) -> AsyncGenerator[str, None]:
        """Process input tokens and yield output tokens using hot KV cache."""
        pass
    
    async def get_output_stream(self) -> AsyncGenerator[str, None]:
        """Get the live output stream. Doesn't block - returns immediately."""
        if not self._is_running:
            self._is_running = True
            self._processing_task = asyncio.create_task(self._processing_loop())
        
        while True:
            try:
                # Non-blocking get with timeout to allow for graceful shutdown
                token = await asyncio.wait_for(self._output_queue.get(), timeout=0.1)
                yield token
            except asyncio.TimeoutError:
                if not self._is_running:
                    break
                continue
            except Exception as e:
                print(f"Error in output stream: {e}")
                break
    
    async def push_input(self, text: str) -> None:
        """Push new input tokens to process. Non-blocking."""
        await self._input_queue.put(text)
    
    async def stop(self) -> None:
        """Stop the processing loop and cleanup."""
        self._is_running = False
        if self._processing_task:
            self._processing_task.cancel()
            try:
                await self._processing_task
            except asyncio.CancelledError:
                pass
    
    async def _processing_loop(self) -> None:
        """Internal processing loop that consumes input and produces output."""
        while self._is_running:
            try:
                # Get input with timeout to allow for graceful shutdown
                input_text = await asyncio.wait_for(self._input_queue.get(), timeout=0.1)
                
                # Process tokens and stream output
                async for token in self._process_tokens(input_text):
                    await self._output_queue.put(token)
                    
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                print(f"Error in processing loop: {e}")
                import traceback
                traceback.print_exc()
                await self._output_queue.put(f"[ERROR: {e}]")
