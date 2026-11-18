"""
Unified LLM client, wrapping whichever backend is configured
(settings.llm_provider) behind one .chat(prompt) -> str method.

Both LLM-touching stages -- deontic.py (stage 3) and compliance.py
(stage 7b) -- call through this instead of talking to Ollama or Groq
directly, so switching providers is a config change (LLM_PROVIDER env
var), not a code change in either module. Both stages' own prompt
templates and response parsing are untouched by which backend answers
them; this class only handles "send this prompt, get this text back".

Ollama and Groq were chosen because Groq exposes an OpenAI-compatible
Chat Completions API, so the two branches below are the only two the
project needs -- adding a third OpenAI-compatible host (e.g. a different
hosted-inference provider) would reuse the Groq branch's shape almost
exactly.
"""
from __future__ import annotations

import ollama

from app.core.config import settings


class LLMClient:
    def __init__(
        self,
        provider: str | None = None,
        ollama_client: "ollama.Client | None" = None,
        groq_client=None,
    ):
        self.provider = provider or settings.llm_provider
        self._ollama_client = ollama_client
        self._groq_client = groq_client

    def chat(self, prompt: str) -> str:
        if self.provider == "groq":
            return self._chat_groq(prompt)
        if self.provider == "ollama":
            return self._chat_ollama(prompt)
        raise ValueError(f"Unknown llm_provider '{self.provider}' -- expected 'ollama' or 'groq'.")

    def _chat_ollama(self, prompt: str) -> str:
        client = self._ollama_client or ollama.Client(host=settings.ollama_host)
        response = client.chat(
            model=settings.ollama_model,
            messages=[{"role": "user", "content": prompt}],
        )
        return response["message"]["content"]

    def _chat_groq(self, prompt: str) -> str:
        if self._groq_client is None:
            # Imported lazily so `groq` isn't a hard dependency for
            # Ollama-only setups that never installed it.
            from groq import Groq

            if not settings.groq_api_key:
                raise RuntimeError(
                    "llm_provider is set to 'groq' but GROQ_API_KEY is not configured. "
                    "Set it in your .env (see README)."
                )
            self._groq_client = Groq(api_key=settings.groq_api_key)

        response = self._groq_client.chat.completions.create(
            model=settings.groq_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,  # deterministic-ish classification/judgment, not creative generation
        )
        return response.choices[0].message.content
