import os
from pathlib import Path
from typing import Dict, List, Tuple

import tomllib
from dotenv import load_dotenv
from huggingface_hub import login
from loguru import logger
from mlx_lm import generate, load


class MLXModelHandler:
    logger.add("mlx_model_handler.log", rotation="2 MB", level="INFO")

    @classmethod
    def initialise(cls):
        try:
            load_dotenv()
            cls.HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
            if not cls.HUGGINGFACE_TOKEN:
                raise ValueError("HUGGINGFACE_TOKEN not found in .env file")

            try:
                with open("model.toml", "rb") as f:
                    model_config = tomllib.load(f)
            except FileNotFoundError:
                logger.error("model.toml file not found")
                raise
            except tomllib.TOMLDecodeError as e:
                logger.error(f"Error decoding model.toml: {e}")
                raise

            try:
                cls.HF_MODEL_NAME = model_config["model"]["name"]
                cls.MAX_TOKENS = model_config["model"]["max_tokens"]
                cls.VERBOSE = model_config["model"]["verbose"]
            except KeyError as e:
                logger.error(f"Missing key in model.toml: {e}")
                raise

            login(token=cls.HUGGINGFACE_TOKEN)
            logger.info("MLXModelHandler initialised with configurations")
        except Exception as e:
            logger.exception(f"Failed to initialise MLXModelHandler: {e}")
            raise

    def __init__(self, model_name: str = None, verbose: bool = None):
        self.model_name = model_name or self.HF_MODEL_NAME
        self.verbose = verbose if verbose is not None else self.VERBOSE
        try:
            self.model, self.tokenizer = self._load_model(self.model_name)
            logger.info(f"Model loaded: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def _load_model(self, model_name: str) -> Tuple:
        return load(model_name)

    @classmethod
    def hf_url(cls) -> str:
        return f"https://huggingface.co/{cls.HF_MODEL_NAME}"

    def generate_text(self, prompt: str, max_tokens: int = None) -> str:
        max_tokens = max_tokens or self.MAX_TOKENS
        return generate(
            self.model,
            self.tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            verbose=self.verbose,
        )

    def chat(self, messages: List[Dict[str, str]]) -> str:
        prompt = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True
        )
        return self.generate_text(prompt)

    @staticmethod
    def get_model_size(model_name: str) -> float:
        cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
        model_dir = cache_dir / f"models--{model_name.replace('/', '--')}"

        if not model_dir.exists():
            logger.error(f"Model directory not found: {model_dir}")
            return 0

        total_size = 0
        try:
            for file_path in model_dir.rglob("*"):
                if file_path.is_file():
                    total_size += file_path.stat().st_size
            return total_size / (1024 * 1024 * 1024)  # Convert to GB
        except Exception as e:
            logger.error(f"Error calculating model size: {e}")
            return 0


class StoryGenerator(MLXModelHandler):
    def create_story(self, theme: str) -> str:
        prompt = f"Write a short story about {theme}"
        logger.info(f"Generating story with prompt: {prompt}")
        return self.generate_text(prompt, max_tokens=self.MAX_TOKENS)


class CodeAssistant(MLXModelHandler):
    def explain_code(self, code_snippet: str) -> str:
        prompt = f"Explain the following code:\n\n{code_snippet}"
        logger.info(f"Explaining code snippet: {prompt}")
        return self.generate_text(prompt)

    def suggest_improvements(self, code_snippet: str) -> str:
        prompt = f"Suggest improvements for the following code:\n\n{code_snippet}"
        logger.info(f"Suggesting improvements for code snippet: {code_snippet}")
        return self.generate_text(prompt)


def print_heading_sep(heading: str):
    print()
    print("-" * 60)
    print(heading)
    print("-" * 60)
    print()


# Example usage


def main():
    try:
        MLXModelHandler.initialise()
        small_model = MLXModelHandler.HF_MODEL_NAME

        # 0. Report on model size (they can get large!)

        print(f"\nModel URL: {MLXModelHandler.hf_url()}\n")

        print_heading_sep("Model size info")
        model_size = MLXModelHandler.get_model_size(small_model)
        logger.info(f"Model size: {model_size:.2f} GB")

        # 1. Story generation
        print_heading_sep("Generate a story example")
        story_gen = StoryGenerator(small_model)
        story = story_gen.create_story("a magical forest")
        print(f"Generated story: {story}")

        # 2. Code assistance
        print_heading_sep("Code assistant example")
        code_assist = CodeAssistant(small_model)
        code_snippet = """
        def fibonacci(n):
            if n <= 1:
                return n
            else:
                return fibonacci(n-1) + fibonacci(n-2)
        """
        explanation = code_assist.explain_code(code_snippet)
        logger.info(f"Code explanation: {explanation}")

        improvements = code_assist.suggest_improvements(code_snippet)
        print(f"Code improvements: {improvements}")

        # 3. Chat example
        print_heading_sep("Chat example")
        chat_model = MLXModelHandler(small_model, verbose=False)
        messages = [
            {"role": "user", "content": "What's the capital of France?"},
            {"role": "assistant", "content": "The capital of France is Paris."},
            {"role": "user", "content": "And what's its most famous landmark?"},
        ]
        chat_response = chat_model.chat(messages)
        print(f"Chat response: {chat_response}")

    except Exception as e:
        logger.exception(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
