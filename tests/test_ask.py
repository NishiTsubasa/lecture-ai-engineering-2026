from types import SimpleNamespace

from app import ask as ask_module


def test_chat_reuses_one_session_for_consecutive_questions(monkeypatch, capsys):
    class FakeChat:
        def __init__(self):
            self.messages = []

        def send_message(self, message):
            self.messages.append(message)
            return SimpleNamespace(
                text=f"reply {len(self.messages)}",
                usage_metadata=SimpleNamespace(
                    prompt_token_count=4,
                    candidates_token_count=2,
                ),
            )

    class FakeClient:
        def __init__(self, api_key):
            self.api_key = api_key
            self.chat = FakeChat()
            self.chats = SimpleNamespace(create=self.create_chat)

        def create_chat(self, model):
            self.model = model
            return self.chat

    client = FakeClient(api_key="test-key")
    monkeypatch.setattr(ask_module.genai, "Client", lambda api_key: client)
    monkeypatch.setattr(ask_module, "get_api_key", lambda: "test-key")
    inputs = iter(["続きの質問", "q"])
    monkeypatch.setattr("builtins.input", lambda prompt: next(inputs))

    ask_module.run_chat("最初の質問")

    assert client.chat.messages == ["最初の質問", "続きの質問"]
    output = capsys.readouterr().out
    assert "reply 1" in output
    assert "reply 2" in output
    assert output.count("[tokens]") == 2