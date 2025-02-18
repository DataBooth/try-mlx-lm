# `try-mlx-lm` - MLX Language Model Handler

## Overview

This project provides a Python-based interface for interacting with language models using the [MLX framework](https://ml-explore.github.io/mlx/build/html/index.html) via the [`mlx-lm`](https://pypi.org/project/mlx-lm/) package.

It's designed to simplify example tasks such as:

- Story generation
- Code assistance
- Chatbot interactions

using pre-trained language models. The project is highly configurable, with examples and prompts externalized to a TOML configuration file (`examples.toml`). It also includes caching to improve performance on repeat model loads, and timing of key methods.

## Features

- **Model Initialisation**: Easily load and initialise language models from [Hugging Face](https://huggingface.co).
- **Story Generation**: Generate short stories based on given themes.
- **Code Assistance**: Explain code snippets and suggest improvements.
- **Chat Functionality**: Engage in conversational interactions with the model.
- **Model Size**: Utility function to determine the size of downloaded model files.
- **Model Caching:** Caches loaded models for reuse, improving performance.
- **Timing**: Measures the execution time of key methods for performance analysis.
- **Examples by Configuration:** All examples are now configurable via a TOML file.

## Main Components

1.  **MLXModelHandler**: The core class that handles model loading, caching, and text generation.
2.  **StoryGenerator**: A specialised class for creating short stories.
3.  **CodeAssistant**: A class dedicated to code explanation and improvement suggestions.

## Setup

1. `uv init` and `uv sync`
2. Set up a `.env` file with your Hugging Face API token:

```
HUGGINGFACE_TOKEN=your_token_here
```

3.  Create a `model.toml` file with a model configuration like:

```
[model]
name = "mlx-community/TinyLlama-1.1B-Chat-v1.0-4bit"
max_tokens = 200
verbose = true
```

4.  Modify the `examples.toml` file to configure the example prompts and headings.
 
## Usage

Run the [main script](src/initial_experiment.py) to see examples of story generation, code assistance, and chat functionality:

## Key Functions

- `MLXModelHandler.initialise()`: Set up the model and configurations.
- `MLXModelHandler._load_model(model_name)`: Loads or retrieves the model from the cache.
- `StoryGenerator.create_story(theme, prompt_template, max_tokens)`: Generate a story based on a given theme using a prompt template.
- `CodeAssistant.explain_code(code_snippet, prompt_template)`: Provide explanations for code snippets using a prompt template.
- `CodeAssistant.suggest_improvements(code_snippet, prompt_template)`: Suggest improvements for given code using a prompt template.
- `MLXModelHandler.chat(messages)`: Engage in a chat conversation with the model.
- `MLXModelHandler.get_model_size(model_name)`: Calculate the size for the specified model.

## Notes

- Make sure you have all the necessary dependencies installed - see [`pyproject.toml`](pyptoject.toml).
- Adjust the configurations in `model.toml` and `examples.toml` to suit your needs.
- This project relies on the key dependencies: MLX framework and Hugging Face models.
```
