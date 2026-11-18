from app.pipeline.llm_client import LLMClient


class _FakeOllamaClient:
    def __init__(self, reply: str):
        self.reply = reply
        self.last_kwargs = None

    def chat(self, **kwargs):
        self.last_kwargs = kwargs
        return {"message": {"content": self.reply}}


class _FakeGroqChoice:
    def __init__(self, content: str):
        self.message = type("Msg", (), {"content": content})


class _FakeGroqCompletions:
    def __init__(self, reply: str):
        self.reply = reply
        self.last_kwargs = None

    def create(self, **kwargs):
        self.last_kwargs = kwargs
        return type("Resp", (), {"choices": [_FakeGroqChoice(self.reply)]})


class _FakeGroqClient:
    def __init__(self, reply: str):
        self.chat = type("Chat", (), {"completions": _FakeGroqCompletions(reply)})


def test_ollama_backend_returns_message_content():
    fake = _FakeOllamaClient(reply="hello from ollama")
    client = LLMClient(provider="ollama", ollama_client=fake)
    assert client.chat("some prompt") == "hello from ollama"
    assert fake.last_kwargs["messages"][0]["content"] == "some prompt"


def test_groq_backend_returns_message_content():
    fake = _FakeGroqClient(reply="hello from groq")
    client = LLMClient(provider="groq", groq_client=fake)
    assert client.chat("some prompt") == "hello from groq"
    assert fake.chat.completions.last_kwargs["messages"][0]["content"] == "some prompt"


def test_unknown_provider_raises():
    client = LLMClient(provider="not-a-real-provider")
    try:
        client.chat("x")
        assert False, "expected ValueError"
    except ValueError:
        pass
