"""
Gemma Live LLM Implementation for CPU inference with hot KV cache (DynamicCache)
- Seeds system instructions into the first cache
- Inherits LiveLLMBase so server can push_input and stream via the base loop
- Forces dynamic cache + eager attention to avoid 4D vs 5D mismatches
- First turn: system + user prompt
- Subsequent turns: only delta tokens (user input + assistant prefix)
- Streams token-by-token with a persistent KV across invocations
"""
import asyncio
from typing import AsyncGenerator, Dict, Optional, Tuple

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

        self._conversation_history = []

        self._system_prompt_template = {
            "role": "system",
            "content": (
            ),
        }

        # State: conversation + KV cache
        self._states: Dict[str, Tuple[str, DynamicCache]] = {
            "default": (
                "You are a helpful AI assistant. "
                "Answer questions directly and completely. "
                "Be creative, engaging, and informative in your responses.",
                DynamicCache(),
            )
        }
        self._current_state: str = "default"

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

        # Lock cache mode
        self.model.config.use_cache = True
        self.model.config.cache_implementation = "dynamic"
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

        print("Gemma model loaded with Hot KV cache and eager attention for streaming inference")

    async def _process_tokens(self, input_text: str) -> AsyncGenerator[str, None]:
        """
        Incrementally process input text with hot KV cache.
        First call encodes system + user; later calls encode only new tokens.
        Streams tokens as they’re produced.
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not initialized. Call initialize_model() first.")

        try:
            # Track user turn
            self._conversation_history.append({"role": "user", "content": input_text})

            # --- Encode input ---
            state, past_key_values = self._states.get(self._current_state, self._states["default"])
            first_turn = (past_key_values is None) or (len(past_key_values) == 0)

            system_prompt = self._system_prompt_template.copy()
            system_prompt["content"] = state

            if first_turn:
                # For Gemma, we need to format the system prompt differently
                # since it might not support system role in chat template
                if hasattr(self.tokenizer, "apply_chat_template"):
                    # Include system prompt as part of the first user message for Gemma
                    user_content = f"{system_prompt['content']}\n\n{self._conversation_history[0]['content']}"
                    messages = [{"role": "user", "content": user_content}]
                    
                    input_ids = self.tokenizer.apply_chat_template(
                        messages,
                        add_generation_prompt=True,
                        return_tensors="pt",
                    )
                    if not torch.is_tensor(input_ids):
                        input_ids = input_ids["input_ids"]
                    
                    # Debug: show what we're feeding to the model
                    debug_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=False)
                    print(f"First turn input: {repr(debug_text)}")
                else:
                    # Fallback: manual text join
                    conv = [f"system: {self._system_prompt['content']}"]
                    for m in self._conversation_history:
                        conv.append(f"{m['role']}: {m['content']}")
                    conv.append("assistant: ")
                    full_text = "\n".join(conv)
                    print(f"First turn fallback input: {repr(full_text)}")
                    input_ids = self.tokenizer(full_text, return_tensors="pt")["input_ids"]

                input_ids = input_ids.to(self.model.device)
                past_key_values = DynamicCache()
            else:
                # Later turns: only encode new user input + assistant prefix
                # We need to be careful NOT to include <bos> or recreate the full conversation
                # Just add the delta: user turn + start of assistant turn
                if hasattr(self.tokenizer, "apply_chat_template"):
                    # Build just the delta without <bos> token
                    delta_messages = [{"role": "user", "content": input_text}]
                    full_delta = self.tokenizer.apply_chat_template(
                        delta_messages,
                        add_generation_prompt=True,
                        return_tensors="pt",
                    )
                    if not torch.is_tensor(full_delta):
                        full_delta = full_delta["input_ids"]
                    
                    # Remove <bos> token if present (it's usually the first token)
                    if full_delta[0, 0] == self.tokenizer.bos_token_id:
                        delta_ids = full_delta[:, 1:]  # Skip the <bos> token
                    else:
                        delta_ids = full_delta
                    
                    # Debug: show what delta we're adding
                    debug_text = self.tokenizer.decode(delta_ids[0], skip_special_tokens=False)
                    print(f"Delta input (no BOS): {repr(debug_text)}")
                    input_ids = delta_ids
                else:
                    # Fallback: manual formatting without special tokens
                    delta_text = f"<start_of_turn>user\n{input_text}<end_of_turn>\n<start_of_turn>model\n"
                    print(f"Delta fallback input: {repr(delta_text)}")
                    input_ids = self.tokenizer(delta_text, return_tensors="pt", add_special_tokens=False)["input_ids"]

                input_ids = input_ids.to(self.model.device)

            if input_ids.shape[-1] == 0:
                # Ensure at least one token goes in
                input_ids = self.tokenizer("\n", return_tensors="pt")["input_ids"].to(self.model.device)

            # --- Iterative generation ---
            response_text = ""
            max_new_tokens = 512  # safety cap to avoid infinite loops

            for step in range(max_new_tokens):
                with torch.no_grad():
                    outputs = self.model(
                        input_ids=input_ids,
                        past_key_values=past_key_values,
                        use_cache=True,
                        cache_implementation="dynamic",
                        attn_implementation="eager",
                    )

                # Update cache
                past_key_values = outputs.past_key_values

                # Sample next token
                logits = outputs.logits[:, -1, :]
                temperature = max(self.generation_config.temperature, 1e-6)
                probs = torch.softmax(logits / temperature, dim=-1)

                # Top-p nucleus sampling
                sorted_probs, sorted_idx = torch.sort(probs, descending=True)
                cumsum = torch.cumsum(sorted_probs, dim=-1)
                cutoff = cumsum > self.generation_config.top_p
                cutoff[..., 0] = False
                sorted_probs[cutoff] = 0
                sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
                next_sorted = torch.multinomial(sorted_probs, num_samples=1)
                next_token = sorted_idx.gather(-1, next_sorted)

                token_id = next_token.item()
                
                # Stop on EOS or end_of_turn tokens
                if token_id == self.tokenizer.eos_token_id:
                    print(f"Hit EOS token at step {step}")
                    break
                elif token_id == 106:  # <end_of_turn> token
                    print(f"Hit end_of_turn token at step {step}")
                    break

                # Decode and stream the new token
                token_text = self.tokenizer.decode(next_token[0].tolist(), skip_special_tokens=True)
                print(f"Generated token: {repr(token_text)} (token_id: {token_id})")
                
                # Only yield tokens that have actual content
                if token_text and token_text.strip():
                    response_text += token_text
                    yield token_text
                    await asyncio.sleep(0.01)
                elif token_text and not token_text.strip():
                    # It's whitespace, include it in response
                    response_text += token_text
                    yield token_text
                    await asyncio.sleep(0.01)

                # Next step: feed only the newly generated token
                input_ids = next_token  # shape [1,1] on same device

            # Signal end of response
            yield "[END_OF_RESPONSE]"

            # Save assistant turn
            if response_text.strip():
                self._conversation_history.append(
                    {"role": "assistant", "content": response_text.strip()}
                )

            # Trim if too long
            if len(self._conversation_history) > 20:
                self._conversation_history = self._conversation_history[-16:]
                past_key_values = DynamicCache()
                print("Conversation history trimmed, cache reset")

            # Update state cache
            self._states[self._current_state] = (state, past_key_values)

        except Exception as e:
            print(f"Error in _process_tokens: {e}")
            import traceback
            traceback.print_exc()
            await self.reset()
            yield f"[ERROR: {str(e)}]"

    async def reset(self) -> None:
        """Clear conversation history and reset cache (but keep system prompt)."""
        self._conversation_history = []
        self._states[self._current_state] = (self._states[self._current_state][0], DynamicCache())
        print("Conversation history and cache reset (system prompt will be re-seeded on next turn)")

    @property
    def state(self) -> str:
        """Get the current system prompt state."""
        state_tuple = self._states.get(self._current_state)
        if state_tuple is not None:
            return state_tuple[0]
        return ""
    
    @state.setter
    def state(self, prompt: str) -> None:
        """Set or update the current system prompt state."""
        if self._current_state in self._states:
            _, existing_cache = self._states[self._current_state]
            self._states[self._current_state] = (prompt, existing_cache)
            print(f"Updated current state '{self._current_state}' with new prompt.")
        else:
            self._states[self._current_state] = (prompt, DynamicCache())
            print(f"Created new state '{self._current_state}' with provided prompt.")

    @property
    def states(self) -> Dict[str, str]:
        """Get all available state names and their prompts."""
        return {name: tup[0] for name, tup in self._states.items()}

    @property
    def current_state(self) -> str:
        """Get the name of the current state."""
        return self._current_state
    @current_state.setter
    def current_state(self, state_name: str) -> None:
        """Switch to a different state, creating it if it doesn't exist."""
        if state_name not in self._states:
            raise ValueError(f"State '{state_name}' does not exist.")
        self._current_state = state_name
        print(f"Switched to state '{state_name}'.")