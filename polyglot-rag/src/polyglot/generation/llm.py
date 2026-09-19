"""LLM clients.

Three providers:
  anthropic  — Claude via the official SDK (ANTHROPIC_API_KEY)
  openai     — any OpenAI-compatible endpoint, including local vLLM/Ollama
               (OPENAI_API_KEY, optional OPENAI_BASE_URL)
  echo       — no network. Returns an extractive answer assembled from the
               retrieved passages, so the retrieval half of the system can be
               demoed, tested and evaluated with no API key at all.
"""

from __future__ import annotations

import os
from typing import Protocol

from ..config import GenerationConfig


class LLMClient(Protocol):
    def complete(self, system: str, user: str) -> str: ...


class AnthropicClient:
    def __init__(self, cfg: GenerationConfig):
        from anthropic import Anthropic

        self.cfg = cfg
        self.client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    def complete(self, system: str, user: str) -> str:
        response = self.client.messages.create(
            model=self.cfg.model,
            max_tokens=self.cfg.max_tokens,
            temperature=self.cfg.temperature,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(
            block.text for block in response.content
            if getattr(block, "type", "") == "text"
        )


class OpenAIClient:
    def __init__(self, cfg: GenerationConfig):
        from openai import OpenAI

        self.cfg = cfg
        self.client = OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY", "not-needed"),
            base_url=os.environ.get("OPENAI_BASE_URL") or None,
        )

    def complete(self, system: str, user: str) -> str:
        response = self.client.chat.completions.create(
            model=self.cfg.model,
            max_tokens=self.cfg.max_tokens,
            temperature=self.cfg.temperature,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return response.choices[0].message.content or ""


class EchoClient:
    """Offline extractive fallback. Not a language model."""

    def __init__(self, cfg: GenerationConfig):
        self.cfg = cfg

    def complete(self, system: str, user: str) -> str:
        import re

        blocks = re.findall(
            r"\[C(\d+)\][^\n]*\n(.*?)(?=\n\n---\n\n|\n</sources>)",
            user,
            flags=re.DOTALL,
        )
        if not blocks:
            return "No sources were retrieved for this question."

        lines = [
            "(offline mode: extractive summary, no model call)",
            "",
        ]
        for marker, body in blocks[:3]:
            snippet = " ".join(body.split())[:280]
            lines.append(f"- {snippet} [C{marker}]")
        return "\n".join(lines)


def build_client(cfg: GenerationConfig) -> LLMClient:
    provider = cfg.provider.lower()
    try:
        if provider == "anthropic":
            return AnthropicClient(cfg)
        if provider in {"openai", "openai-compatible", "local"}:
            return OpenAIClient(cfg)
        if provider == "echo":
            return EchoClient(cfg)
    except Exception as exc:  # missing SDK or missing key
        import logging

        logging.getLogger(__name__).warning(
            "Provider '%s' unavailable (%s); using offline echo client.",
            provider, exc,
        )
        return EchoClient(cfg)

    raise ValueError(f"Unknown generation provider: {cfg.provider}")
