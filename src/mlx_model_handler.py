import os
from pathlib import Path
from typing import Dict, List, Tuple

import tomllib
from dotenv import load_dotenv
from huggingface_hub import login
from loguru import logger
from mlx_lm import generate, load
import time


def timer(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        logger.info(f"{func.__name__} executed in {execution_time:.2f} seconds")
        return result

    return wrapper


class MLXModelHandler:
    logger.add("mlx_model_handler.log", rotation="2 MB", level="INFO")
    _model_cache = {}  # Class-level cache for loaded models

    @classmethod
    def initialise(cls):
        try:
            load_dotenv()
            cls.HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
            if not cls.HUGGINGFACE_TOKEN:
                raise ValueError("HUGGINGFACE_TOKEN not found in .env file")

            model_toml_path = Path("model.toml")
            if not model_toml_path.exists():
                raise FileNotFoundError("model.toml file not found")

            with open(model_toml_path, "rb") as f:
                model_config = tomllib.load(f)

            cls.HF_MODEL_NAME = model_config["model"]["name"]
            cls.MAX_TOKENS = model_config["model"]["max_tokens"]
            cls.VERBOSE = model_config["model"]["verbose"]

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

    @classmethod
    @timer
    def _load_model(cls, model_name: str) -> Tuple:
        if model_name in cls._model_cache:
            logger.info(f"Using cached model: {model_name}")
            return cls._model_cache[model_name]

        logger.info(f"Loading model: {model_name}")
        model, tokenizer = load(model_name)
        cls._model_cache[model_name] = (model, tokenizer)
        return model, tokenizer

    @classmethod
    def hf_url(cls) -> str:
        return f"https://huggingface.co/{cls.HF_MODEL_NAME}"

    @timer
    def generate_text(self, prompt: str, max_tokens: int = None) -> str:
        max_tokens = max_tokens or self.MAX_TOKENS
        return generate(
            self.model,
            self.tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            verbose=self.verbose,
        )

    @timer
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
    @timer
    def create_story(self, theme: str, prompt_template: str, max_tokens: int) -> str:
        prompt = prompt_template.format(theme=theme)
        logger.info(f"Generating story with prompt: {prompt}")
        return self.generate_text(prompt, max_tokens=max_tokens)


class CodeAssistant(MLXModelHandler):
    @timer
    def explain_code(self, code_snippet: str, prompt_template: str) -> str:
        prompt = prompt_template.format(code_snippet=code_snippet)
        logger.info(f"Explaining code snippet: {prompt}")
        return self.generate_text(prompt)

    @timer
    def suggest_improvements(self, code_snippet: str, prompt_template: str) -> str:
        prompt = prompt_template.format(code_snippet=code_snippet)
        logger.info(f"Suggesting improvements for code snippet: {prompt}")
        return self.generate_text(prompt)


def print_heading_sep(heading: str):
    print()
    print("-" * 60)
    print(heading)
    print("-" * 60)
    print()


def main():
    try:
        MLXModelHandler.initialise()
        hf_model = MLXModelHandler.HF_MODEL_NAME

        print(f"\nModel URL: {MLXModelHandler.hf_url()}\n")

        # Load examples configuration
        with open(Path.cwd() / "src/examples.toml", "rb") as f:
            config = tomllib.load(f)

        print_heading_sep(config["general"]["model_info_heading"])
        model_size = MLXModelHandler.get_model_size(hf_model)
        logger.info(f"Model size: {model_size:.2f} GB")

        print_heading_sep(config["generate_story"]["heading"])
        story_gen = StoryGenerator(hf_model)
        story = story_gen.create_story(
            config["generate_story"]["theme"],
            config["generate_story"]["prompt_template"],
            config["generate_story"]["max_tokens"],
        )
        print(f"Generated story: {story}")

        print_heading_sep(config["code_assistant"]["heading"])
        code_assist = CodeAssistant(hf_model)
        code_snippet = config["code_assistant"]["code_snippet"]
        explanation = code_assist.explain_code(
            code_snippet, config["code_assistant"]["explain_prompt_template"]
        )
        logger.info(f"Code explanation: {explanation}")

        improvements = code_assist.suggest_improvements(
            code_snippet, config["code_assistant"]["improve_prompt_template"]
        )
        print(f"Code improvements: {improvements}")

        print_heading_sep(config["chat"]["heading"])
        chat_model = MLXModelHandler(hf_model, verbose=False)
        chat_response = chat_model.chat(config["chat"]["messages"])
        print(f"Chat response: {chat_response}")

    except Exception as e:
        logger.exception(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
