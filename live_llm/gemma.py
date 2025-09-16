"""
Gemma Live LLM Implementation for CPU inference with hot KV cache (DynamicCache)
- Inherits LiveLLMBase so server can push_input and stream via the base loop
- Forces dynamic cache + eager attention to avoid 4D vs 5D mismatches
- First turn: full prompt
- Subsequent turns: only delta tokens
- Streams token-by-token with a persistent KV across invocations
"""
import asyncio
from typing import AsyncGenerator, Optional

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    GenerationConfig,
    DynamicCache,
)
from .base import LiveLLMBase


class GemmaLiveLLM(LiveLLMBase):
    """Live streaming Gemma implementation with DynamicCache-based KV reuse."""

    def __init__(self, model_name: str = "google/gemma-3-270m-it"):
        super().__init__()  # initialize LiveLLMBase queues/loop
        self.model_name = model_name
        self.model: Optional[AutoModelForCausalLM] = None
        self.tokenizer: Optional[AutoTokenizer] = None
        self.generation_config: Optional[GenerationConfig] = None

        # State: conversation + KV cache
        self._conversation_history = []
        self._past_key_values: Optional[DynamicCache] = None

    async def initialize_model(self) -> None:
        """Load and prepare the Gemma model for CPU inference."""
        print(f"Loading Gemma model: {self.model_name}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Important: use eager attention so key/value tensors are 4D (compatible with DynamicCache)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float32,
            device_map="cpu",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            attn_implementation="eager",  # <- critical for DynamicCache compatibility
        )

        # Lock cache mode so nothing silently switches to static/paged
        self.model.config.use_cache = True
        self.model.config.cache_implementation = "dynamic"
        # Some builds also read from generation_config:
        self.model.generation_config.cache_implementation = "dynamic"

        self.generation_config = GenerationConfig(
            do_sample=True,
            temperature=0.8,
            top_p=0.95,
            top_k=50,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            repetition_penalty=1.1,
            use_cache=True,
        )

        # Start with empty dynamic cache
        self._past_key_values = DynamicCache()
        print("Gemma model loaded with DynamicCache + eager attention for streaming inference")

    async def _process_tokens(self, input_text: str) -> AsyncGenerator[str, None]:
        """
        Incrementally process input text with hot KV cache.
        First call encodes full prompt; later calls encode only new tokens.
        Streams tokens as they’re produced.
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not initialized. Call initialize_model() first.")

        try:
            # Track user turn
            self._conversation_history.append({"role": "user", "content": input_text})

            # --- Encode input ---
            first_turn = (self._past_key_values is None) or (len(self._past_key_values) == 0)

            if first_turn:
                # Full conversation with a generation prompt
                if hasattr(self.tokenizer, "apply_chat_template"):
                    input_ids = self.tokenizer.apply_chat_template(
                        self._conversation_history,
                        add_generation_prompt=True,
                        return_tensors="pt",
                    )
                    if not torch.is_tensor(input_ids):
                        # Some tokenizers return dicts; normalize to tensor
                        input_ids = input_ids["input_ids"]
                else:
                    # Fallback formatting if no chat template is available
                    conv = []
                    for m in self._conversation_history:
                        role = m["role"]
                        conv.append(f"{role}: {m['content']}")
                    conv.append("assistant: ")
                    input_ids = self.tokenizer("\n".join(conv), return_tensors="pt")["input_ids"]

                input_ids = input_ids.to(self.model.device)
                # Fresh dynamic cache
                self._past_key_values = DynamicCache()
            else:
                # Only the delta that logically follows previous text (user + assistant prefix)
                if hasattr(self.tokenizer, "apply_chat_template"):
                    delta_ids = self.tokenizer.apply_chat_template(
                        [{"role": "user", "content": input_text}],
                        add_generation_prompt=True,   # adds assistant turn prefix
                        return_tensors="pt",
                    )
                    if not torch.is_tensor(delta_ids):
                        delta_ids = delta_ids["input_ids"]
                else:
                    delta_text = f"user: {input_text}\nassistant: "
                    delta_ids = self.tokenizer(delta_text, return_tensors="pt")["input_ids"]

                input_ids = delta_ids.to(self.model.device)
                # Keep existing cache

            # Safety: ensure at least one new token (avoid cache_position empty bugs)
            if input_ids.shape[-1] == 0:
                bump = self.tokenizer("\n", return_tensors="pt")["input_ids"].to(self.model.device)
                input_ids = bump

            # --- Iterative generation ---
            response_text = ""
            max_new_tokens = 60  # tunable

            for _ in range(max_new_tokens):
                with torch.no_grad():
                    outputs = self.model(
                        input_ids=input_ids,
                        past_key_values=self._past_key_values,
                        use_cache=True,
                        cache_implementation="dynamic",  # <- force dynamic each call
                        attn_implementation="eager",     # <- and eager attention
                    )

                # Update cache
                self._past_key_values = outputs.past_key_values

                # Sample next token (nucleus sampling)
                logits = outputs.logits[:, -1, :]  # [1, vocab]
                temperature = max(self.generation_config.temperature, 1e-6)
                probs = torch.softmax(logits / temperature, dim=-1)

                # Top-p (nucleus) filter
                sorted_probs, sorted_idx = torch.sort(probs, descending=True)
                cumsum = torch.cumsum(sorted_probs, dim=-1)
                cutoff = cumsum > self.generation_config.top_p
                cutoff[..., 0] = False  # always keep at least one
                sorted_probs[cutoff] = 0
                sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
                next_sorted = torch.multinomial(sorted_probs, num_samples=1)  # [1,1]
                next_token = sorted_idx.gather(-1, next_sorted)              # [1,1]

                # Stop on EOS
                if next_token.item() == self.tokenizer.eos_token_id:
                    break

                # Decode and stream the new token
                token_text = self.tokenizer.decode(next_token[0].tolist(), skip_special_tokens=True)
                if token_text:
                    response_text += token_text
                    yield token_text
                    await asyncio.sleep(0.01)

                # Next step: feed only the newly generated token
                input_ids = next_token  # shape [1,1] on same device

            # Store assistant turn
            if response_text.strip():
                self._conversation_history.append(
                    {"role": "assistant", "content": response_text.strip()}
                )

            # Trim history & reset cache if too long (avoid drift)
            if len(self._conversation_history) > 20:
                self._conversation_history = self._conversation_history[-16:]
                self._past_key_values = DynamicCache()
                print("Conversation history trimmed, cache reset")

        except Exception as e:
            print(f"Error in _process_tokens: {e}")
            import traceback
            traceback.print_exc()
            await self.reset()
            yield f"[ERROR: {str(e)}]"

    async def reset(self) -> None:
        """Clear conversation history and reset cache."""
        self._conversation_history = []
        self._past_key_values = DynamicCache()
        print("Conversation history and cache reset")
