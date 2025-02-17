import os
from typing import Dict, List

from dotenv import load_dotenv
from huggingface_hub import login
from mlx_lm import generate, load

# Load environment variables

load_dotenv()
hf_token = os.getenv("HUGGINGFACE_TOKEN")
os.environ["HUGGINGFACE_TOKEN"] = hf_token

login(token=hf_token)

HF_MODEL_NAME = "mlx-community/TinyLlama-1.1B-Chat-v1.0-4bit"

class MLXModelHandler:
    def __init__(self, model_name: str):
        self.model, self.tokenizer = load(model_name)

    def generate_text(self, prompt: str, max_tokens: int = 100) -> str:
        return generate(self.model, self.tokenizer, prompt=prompt, max_tokens=max_tokens)

    def chat(self, messages: List[Dict[str, str]]) -> str:
        prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        return self.generate_text(prompt)

class StoryGenerator(MLXModelHandler):
    def create_story(self, theme: str) -> str:
        prompt = f"Write a short story about {theme}"
        return self.generate_text(prompt, max_tokens=200)

class CodeAssistant(MLXModelHandler):
    def explain_code(self, code_snippet: str) -> str:
        prompt = f"Explain the following code:\n\n{code_snippet}"
        return self.generate_text(prompt)

    def suggest_improvements(self, code_snippet: str) -> str:
        prompt = f"Suggest improvements for the following code:\n\n{code_snippet}"
        return self.generate_text(prompt)

class MLXModelHandler:
    def __init__(self, model_name: str):
        self.model, self.tokenizer = load(model_name)

    def generate_text(self, prompt: str, max_tokens: int = 100) -> str:
        return generate(self.model, self.tokenizer, prompt=prompt, max_tokens=max_tokens)

    def chat(self, messages: List[Dict[str, str]]) -> str:
        prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        return self.generate_text(prompt)

# Examples
if __name__ == "__main__":
    
    # Use smaller model for initial experiment
    small_model = HF_MODEL_NAME
    
    # 1. Story generation
    story_gen = StoryGenerator(small_model)
    print(story_gen.create_story("a magical forest"))

    # 2. Code assistance
    code_assist = CodeAssistant(small_model)
    code_snippet = """
    def fibonacci(n):
        if n <= 1:
            return n
        else:
            return fibonacci(n-1) + fibonacci(n-2)
    """
    print(code_assist.explain_code(code_snippet))
    print(code_assist.suggest_improvements(code_snippet))

    # 3. Chat example
    chat_model = MLXModelHandler(small_model)
    messages = [
        {"role": "user", "content": "What's the capital of France?"},
        {"role": "assistant", "content": "The capital of France is Paris."},
        {"role": "user", "content": "And what's its most famous landmark?"}
    ]
    print(chat_model.chat(messages))
