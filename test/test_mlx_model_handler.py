import pytest
from unittest.mock import patch, MagicMock

from mlx_model_handler import MLXModelHandler, StoryGenerator, CodeAssistant


@pytest.fixture(autouse=True)
def mlx_model_handler_model_handler(mock_env_and_toml):
    with patch("mlx_model_handler.Path") as mock_path:
        mock_path.return_value.exists.return_value = True
        mock_path.return_value.__str__.return_value = "mocked_model.toml"
        MLXModelHandler.initialise()


@pytest.fixture
def mock_env_and_toml(monkeypatch):
    monkeypatch.setenv("HUGGINGFACE_TOKEN", "mock_token")
    mock_toml_content = b"""
    [model]
    name = "test-model"
    max_tokens = 100
    verbose = false
    """
    with patch("builtins.open", new_callable=MagicMock) as mock_open:
        mock_open.return_value.__enter__.return_value.read.return_value = mock_toml_content
        yield mock_open


@pytest.fixture
def mock_load():
    with patch("mlx_model_handler.load") as mock:
        mock.return_value = (MagicMock(), MagicMock())
        yield mock


@pytest.fixture
def mock_generate():
    with patch("mlx_model_handler.generate") as mock:
        mock.return_value = "Generated text"
        yield mock


class TestMLXModelHandler:

    def test_initialise(self, mock_env_and_toml):
        assert MLXModelHandler.HF_MODEL_NAME == "test-model"
        assert MLXModelHandler.MAX_TOKENS == 100
        assert MLXModelHandler.VERBOSE is False

    def test_load_model(self, mock_env_and_toml, mock_load):
        handler = MLXModelHandler()
        model, tokenizer = handler._load_model(MLXModelHandler.HF_MODEL_NAME)
        assert model is not None
        assert tokenizer is not None
        mock_load.assert_called_once_with(MLXModelHandler.HF_MODEL_NAME)

    def test_generate_text(self, mock_env_and_toml, mock_load, mock_generate):
        handler = MLXModelHandler()
        result = handler.generate_text("Test prompt")
        assert result == "Generated text"
        mock_generate.assert_called_once()

    def test_chat(self, mock_env_and_toml, mock_load, mock_generate):
        handler = MLXModelHandler()
        messages = [{"role": "user", "content": "Hello"}]
        result = handler.chat(messages)
        assert result == "Generated text"
        mock_generate.assert_called_once()

    def test_get_model_size(self, mock_env_and_toml, tmp_path):
        model_dir = tmp_path / "models--test-model"
        model_dir.mkdir(parents=True)
        (model_dir / "model.bin").write_text("dummy content" * 1000)

        with patch("pathlib.Path.home") as mock_home:
            mock_home.return_value = tmp_path
            size = MLXModelHandler.get_model_size("test-model")

        assert size > 0


class TestStoryGenerator:

    def test_create_story(self, mock_env_and_toml, mock_load, mock_generate):
        generator = StoryGenerator()
        story = generator.create_story("test theme", "Write about {theme}", 100)
        assert story == "Generated text"
        mock_generate.assert_called_once()


class TestCodeAssistant:

    def test_explain_code(self, mock_env_and_toml, mock_load, mock_generate):
        assistant = CodeAssistant()
        explanation = assistant.explain_code(
            "def test(): pass", "Explain: {code_snippet}"
        )
        assert explanation == "Generated text"
        mock_generate.assert_called_once()

    def test_suggest_improvements(self, mock_env_and_toml, mock_load, mock_generate):
        assistant = CodeAssistant()
        improvements = assistant.suggest_improvements(
            "def test(): pass", "Improve: {code_snippet}"
        )
        assert improvements == "Generated text"
        mock_generate.assert_called_once()
