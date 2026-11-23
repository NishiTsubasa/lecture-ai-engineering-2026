from app import config


def test_get_api_key_reads_project_env(monkeypatch, tmp_path):
    (tmp_path / ".env").write_text("GEMINI_API_KEY=from-dotenv\n", encoding="utf-8")
    monkeypatch.setattr(config, "ROOT", tmp_path)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    assert config.get_api_key() == "from-dotenv"


def test_process_environment_takes_precedence_over_dotenv(monkeypatch, tmp_path):
    (tmp_path / ".env").write_text("GEMINI_API_KEY=from-dotenv\n", encoding="utf-8")
    monkeypatch.setattr(config, "ROOT", tmp_path)
    monkeypatch.setenv("GEMINI_API_KEY", "from-environment")

    assert config.get_api_key() == "from-environment"
