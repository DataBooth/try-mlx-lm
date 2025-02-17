# `try-mlx-lm` - MLX Language Model Handler

## Overview

This project provides a Python-based interface for interacting with language models using the [MLX framework](https://ml-explore.github.io/mlx/build/html/index.html) via the [`mlx-lm`](https://pypi.org/project/mlx-lm/) package. 

It's designed to simplify some example tasks such as:  

- story generation, 
- code assistance, and 
- chatbot interactions 
using pre-trained language models.

## Features

- **Model Initialisation**: Easily load and initialise language models from [Hugging Face](https://huggingface.co).
- **Story Generation**: Generate short stories based on given themes.
- **Code Assistance**: Explain code snippets and suggest improvements.
- **Chat Functionality**: Engage in conversational interactions with the model.
- **Model Size**: Utility function to determine the size of downloaded model files.

## Main Components

1. **MLXModelHandler**: The core class that handles model loading and text generation.
2. **StoryGenerator**: A specialised class for creating short stories.
3. **CodeAssistant**: A class dedicated to code explanation and improvement suggestions.

## Setup

1. `uv init` and `uv sync`
2. Set up a `.env` file with your Hugging Face API token:
   ```
   HUGGINGFACE_TOKEN=your_token_here
   ```
3. Create a `model.toml` file with a model configuration like:
   ```
   [model]
   name = "mlx-community/TinyLlama-1.1B-Chat-v1.0-4bit"
   max_tokens = 200
   verbose = true
   ```

## Usage
Run the [main script](src/initial_experiment.py) to see examples of story generation, code assistance, and chat functionality:

## Key Functions

- `MLXModelHandler.initialise()`: Set up the model and configurations.
- `StoryGenerator.create_story(theme)`: Generate a story based on a given theme.
- `CodeAssistant.explain_code(code_snippet)`: Provide explanations for code snippets.
- `CodeAssistant.suggest_improvements(code_snippet)`: Suggest improvements for given code.
- `MLXModelHandler.chat(messages)`: Engage in a chat conversation with the model.
- `MLXModelHandler.get_model_size(model_name)`: Calculate the size for the specified model.