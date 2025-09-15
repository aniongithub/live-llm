"""
Gemma Live LLM Implementation for CPU inference with hot KV cache
"""
import asyncio
import copy
from typing import AsyncGenerator, Optional
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, StaticCache
from .base import LiveLLMBase


class GemmaLiveLLM(LiveLLMBase):
    """Live streaming Gemma implementation with static KV cache."""

    def __init__(self, model_name: str = "google/gemma-3-270m-it"):
        super().__init__()
        self.model_name = model_name
        self.model: Optional[AutoModelForCausalLM] = None
        self.tokenizer: Optional[AutoTokenizer] = None
        self.generation_config: Optional[GenerationConfig] = None

        # Conversation tracking and cache management
        self._conversation_history = []
        self._system_cache = None  # Prefilled cache with system prompt
        self._system_prompt_text = None

    async def initialize_model(self) -> None:
        """Load and prepare the Gemma model for CPU inference."""
        print(f"Loading Gemma model: {self.model_name}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            dtype=torch.float32,
            device_map="cpu",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )

        self.generation_config = GenerationConfig(
            max_new_tokens=1024,
            do_sample=True,
            temperature=0.8,
            top_p=0.95,
            top_k=50,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            use_cache=True,
            repetition_penalty=1.1,
            # Remove cache_implementation since we'll manage past_key_values manually
        )
        
        # Don't set cache_implementation on the model either - we'll manage it manually
        # self.model.generation_config.cache_implementation = "static"
        
        # Compile the model for better performance with static cache
        # Note: Disabled due to torch.compile compatibility issues with Gemma3
        try:
            print("Skipping torch.compile() due to compatibility issues with Gemma3...")
            # self.model = torch.compile(self.model, mode="reduce-overhead", fullgraph=True)
            # print("Model compiled successfully")
        except Exception as e:
            print(f"Failed to compile model: {e}")
            print("Continuing without compilation...")
        
        print("Gemma model loaded successfully")
        
        # Initialize and prefill system cache following HuggingFace example
        await self._initialize_system_cache()
        print("System cache prefilled successfully")

    async def _initialize_system_cache(self) -> None:
        """Initialize and prefill the system cache with the system prompt."""
        # Define the system prompt
        self._system_prompt_text = "You are a helpful English-speaking assistant. You must ONLY respond in English. Never use any other languages like Hindi, Marathi, or any non-English text. Always give clear, concise answers in English only."
        
        # Format system prompt using chat template
        system_messages = [
            {"role": "user", "content": self._system_prompt_text},
            {"role": "assistant", "content": "I understand. I will only respond in English and provide clear, helpful answers."}
        ]
        
        if hasattr(self.tokenizer, "apply_chat_template"):
            system_text = self.tokenizer.apply_chat_template(
                system_messages, tokenize=False, add_generation_prompt=False
            )
        else:
            # Fallback manual formatting
            system_text = "<bos>"
            for msg in system_messages:
                if msg["role"] == "user":
                    system_text += f"<start_of_turn>user\n{msg['content']}<end_of_turn>\n"
                else:
                    system_text += f"<start_of_turn>model\n{msg['content']}<end_of_turn>\n"
        
        # Tokenize system prompt
        system_inputs = self.tokenizer(system_text, return_tensors="pt")
        
        # Initialize empty cache
        max_cache_len = 2048
        prompt_cache = StaticCache(config=self.model.config, max_cache_len=max_cache_len)
        
        # Prefill cache with system prompt using forward pass (not generate)
        with torch.no_grad():
            outputs = self.model(**system_inputs, past_key_values=prompt_cache)
            self._system_cache = outputs.past_key_values
        
        print(f"System cache prefilled with {system_inputs['input_ids'].shape[1]} tokens")

    async def reset(self) -> None:
        """Clear the conversation history and reset cache."""
        self._conversation_history = []
        # Reinitialize system cache
        if self.model:
            await self._initialize_system_cache()
        print("Conversation history reset, system cache reinitialized")

    async def _process_tokens(self, input_text: str) -> AsyncGenerator[str, None]:
        """Process input tokens using prefilled system cache following HuggingFace pattern."""
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not initialized. Call initialize_model() first.")

        try:
            # Add user input to conversation history
            self._conversation_history.append({"role": "user", "content": input_text})

            # Build conversation context (excluding system prompt since it's cached)
            conversation_messages = []
            
            # Add conversation history ensuring proper alternation
            history_pairs = []
            temp_user = None
            
            for msg in self._conversation_history:
                if msg["role"] == "user":
                    temp_user = msg
                elif msg["role"] == "assistant" and temp_user:
                    history_pairs.append((temp_user, msg))
                    temp_user = None
            
            # Add the most recent complete pairs and current user message
            recent_pairs = history_pairs[-2:]  # Last 2 complete exchanges
            for user_msg, assistant_msg in recent_pairs:
                conversation_messages.extend([user_msg, assistant_msg])
            
            # Add the current user message if it's not already included
            if not recent_pairs or self._conversation_history[-1] not in [msg for pair in recent_pairs for msg in pair]:
                conversation_messages.append(self._conversation_history[-1])

            # Format conversation part (without system prompt)
            if hasattr(self.tokenizer, "apply_chat_template"):
                conversation_text = self.tokenizer.apply_chat_template(
                    conversation_messages, tokenize=False, add_generation_prompt=True
                )
                # Remove <bos> if present since we're continuing from system cache
                if conversation_text.startswith("<bos>"):
                    conversation_text = conversation_text[5:]
            else:
                # Manual formatting for conversation part
                conversation_text = ""
                for msg in conversation_messages:
                    if msg["role"] == "user":
                        conversation_text += f"<start_of_turn>user\n{msg['content']}<end_of_turn>\n"
                    else:
                        conversation_text += f"<start_of_turn>model\n{msg['content']}<end_of_turn>\n"
                conversation_text += "<start_of_turn>model\n"

            # Create full prompt: system + conversation
            full_prompt = self._system_prompt_text if self._system_prompt_text else ""
            system_part = "You are a helpful English-speaking assistant. You must ONLY respond in English. Never use any other languages like Hindi, Marathi, or any non-English text. Always give clear, concise answers in English only.I understand. I will only respond in English and provide clear, helpful answers."
            full_prompt = system_part + conversation_text

            # Tokenize the full prompt
            full_inputs = self.tokenizer(full_prompt, return_tensors="pt")

            if full_inputs['input_ids'].shape[1] == 0:
                yield "I didn't receive any input. Please try again."
                return

            print(f"Processing {full_inputs['input_ids'].shape[1]} tokens (using prefilled system cache)")

            # Copy the system cache for this generation (following HuggingFace example)
            past_key_values = copy.deepcopy(self._system_cache)

            # Generate using the copied cache
            with torch.no_grad():
                outputs = self.model.generate(
                    **full_inputs,
                    max_new_tokens=60,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    repetition_penalty=1.1,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    use_cache=True,
                    past_key_values=past_key_values,
                )

                # Extract generated tokens
                generated_ids = outputs[0][full_inputs['input_ids'].shape[1]:]
                print(f"Generated {len(generated_ids)} tokens")

                # Stream tokens
                response_text = ""
                for token_id in generated_ids:
                    if token_id == self.tokenizer.eos_token_id:
                        break
                    
                    token_text = self.tokenizer.decode([token_id], skip_special_tokens=True)
                    
                    # Stop on end of turn
                    if '<end_of_turn>' in token_text:
                        before_marker = token_text.split('<end_of_turn>')[0]
                        if before_marker:
                            response_text += before_marker
                            yield before_marker
                        break
                    
                    # Filter out non-English characters (basic filter)
                    if token_text:
                        # Check if token contains mostly non-ASCII characters (likely non-English)
                        ascii_chars = sum(1 for c in token_text if ord(c) < 128)
                        total_chars = len(token_text)
                        
                        # If more than 20% non-ASCII characters, skip this token
                        if total_chars > 0 and (ascii_chars / total_chars) < 0.8:
                            continue
                            
                        response_text += token_text
                        yield token_text
                        await asyncio.sleep(0.01)

                # Add response to conversation history
                if response_text.strip():
                    self._conversation_history.append(
                        {"role": "assistant", "content": response_text.strip()}
                    )

                # Keep history bounded
                if len(self._conversation_history) > 20:
                    # Only trim history, keep system cache
                    self._conversation_history = self._conversation_history[-16:]
                    print("Conversation history trimmed (system cache preserved)")

        except Exception as e:
            print(f"Error in _process_tokens: {e}")
            import traceback
            traceback.print_exc()
            await self.reset()
            yield f"[ERROR: {str(e)}]"
