from __future__ import annotations

import os
import re

import requests


class OllamaClient:
    def __init__(self, model_env_key: str):
        self.base = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.model = os.getenv(model_env_key)

    @staticmethod
    def _clean(raw: str) -> str:
        """Strip markdown fences and extract first JSON object/array."""
        text = raw.strip()
        # Remove ```json ... ``` or ``` ... ``` wrappers
        text = re.sub(r'^```(?:json)?\s*', '', text)
        text = re.sub(r'\s*```$', '', text.strip())
        text = text.strip()
        # Extract the first {...} block in case the model added prose before/after
        m = re.search(r'\{.*\}', text, re.DOTALL)
        return m.group(0) if m else text

    def generate(self, prompt: str) -> str:
        m = self.model or "llama3.1:8b"
        r = requests.post(
            f"{self.base}/api/generate",
            json={
                "model": m,
                "prompt": prompt,
                "stream": False,
                "format": "json",
            },
            timeout=300,
        )
        r.raise_for_status()
        data = r.json()
        raw = data.get("response", "") or ""
        return self._clean(raw)
