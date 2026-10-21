"""LLM back-ends.

Three interchangeable back-ends share one method, ``chat(messages) -> str``:

* ``HFInferenceLLM``  - Hugging Face Inference API (open models, free tier with a token)
* ``OpenAICompatLLM`` - any OpenAI-compatible server, e.g. a local Ollama
* ``TransformersLLM`` - open model run locally with Hugging Face transformers (e.g. on a Colab GPU)
* ``MockLLM``         - deterministic stand-in used for tests and offline demos
"""

from __future__ import annotations

import json
import os
from typing import Callable, Protocol

Message = dict[str, str]


class LLM(Protocol):
    def chat(self, messages: list[Message], max_tokens: int = 1024) -> str: ...


class HFInferenceLLM:
    """Open model served by the Hugging Face Inference API."""

    def __init__(self, model: str | None = None, token: str | None = None, temperature: float = 0.3):
        from huggingface_hub import InferenceClient

        self.model = model or os.getenv("QT_MODEL", "Qwen/Qwen2.5-7B-Instruct")
        self.client = InferenceClient(model=self.model, token=token or os.getenv("HF_TOKEN"))
        self.temperature = temperature

    def chat(self, messages: list[Message], max_tokens: int = 1024) -> str:
        out = self.client.chat_completion(messages=messages, max_tokens=max_tokens, temperature=self.temperature)
        return out.choices[0].message.content or ""


class TransformersLLM:
    """Open model run on this machine with transformers (GPU if available).

    Small instruct models fit on a free Colab T4 GPU, e.g. ``Qwen/Qwen2.5-3B-Instruct``
    (default) or ``Qwen/Qwen2.5-1.5B-Instruct`` for faster replies.
    """

    def __init__(self, model: str | None = None, temperature: float = 0.3):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_name = model or os.getenv("QT_LOCAL_MODEL", "Qwen/Qwen2.5-3B-Instruct")
        self.temperature = temperature
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype=dtype,
                                                          device_map="auto" if torch.cuda.is_available() else None)
        self.model.eval()

    def chat(self, messages: list[Message], max_tokens: int = 1024) -> str:
        import torch

        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=max_tokens, do_sample=self.temperature > 0,
                                      temperature=max(self.temperature, 1e-5), top_p=0.9,
                                      pad_token_id=self.tokenizer.eos_token_id)
        return self.tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


class OpenAICompatLLM:
    """OpenAI-compatible endpoint (e.g. Ollama at http://localhost:11434/v1)."""

    def __init__(self, model: str | None = None, base_url: str | None = None, api_key: str | None = None,
                 temperature: float = 0.3):
        import urllib.request  # noqa: F401  (stdlib only, no extra dependency)

        self.model = model or os.getenv("QT_MODEL", "qwen2.5:7b-instruct")
        self.base_url = (base_url or os.getenv("QT_BASE_URL", "http://localhost:11434/v1")).rstrip("/")
        self.api_key = api_key or os.getenv("QT_API_KEY", "ollama")
        self.temperature = temperature

    def chat(self, messages: list[Message], max_tokens: int = 1024) -> str:
        import urllib.request

        body = json.dumps({"model": self.model, "messages": messages, "max_tokens": max_tokens,
                           "temperature": self.temperature}).encode()
        req = urllib.request.Request(f"{self.base_url}/chat/completions", data=body,
                                     headers={"Content-Type": "application/json",
                                              "Authorization": f"Bearer {self.api_key}"})
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read())
        return data["choices"][0]["message"]["content"] or ""


class MockLLM:
    """Returns scripted replies; the default script makes the app usable offline."""

    def __init__(self, responder: Callable[[list[Message]], str] | None = None):
        self.responder = responder or _default_mock
        self.calls: list[list[Message]] = []

    def chat(self, messages: list[Message], max_tokens: int = 1024) -> str:
        self.calls.append(messages)
        return self.responder(messages)


def _default_mock(messages: list[Message]) -> str:
    """Very small heuristic 'model' so the UI can be tried without any API."""
    from .text import split_sentences

    user = messages[-1]["content"]
    if "TASK: GENERATE_QUIZ" in user:
        lecture = user.split("LECTURE:", 1)[-1]
        sents = [s for s in split_sentences(lecture) if len(s.split()) >= 6 and s[-1] in ".!?"][:5]
        qs = []
        for s in sents:
            words = s.rstrip(".").split()
            key = max(words, key=len)
            qs.append({"type": "short_answer",
                       "question": "Fill in the blank: " + s.replace(key, "_____", 1),
                       "answer": key.strip(",;:"), "options": [], "source_quote": s,
                       "difficulty": "easy"})
        return json.dumps({"questions": qs})
    if "TASK: GRADE" in user:
        ok = "correct" if "EXPECTED:" in user and user.split("EXPECTED:")[1].split("\n")[0].strip().lower() \
            in user.split("STUDENT ANSWER:")[1].lower() else "incorrect"
        return json.dumps({"verdict": ok, "score": 1.0 if ok == "correct" else 0.0,
                           "feedback": "Mock feedback: compare your answer with the source sentence.",
                           "follow_up": "Can you explain this idea in your own words?"})
    return "OK"


_LOCAL_CACHE: dict[str, "TransformersLLM"] = {}


def make_llm(backend: str = "hf", **kwargs) -> LLM:
    backend = backend.lower()
    if backend == "local":  # load the model once and reuse it (loading takes a minute)
        key = kwargs.get("model") or os.getenv("QT_LOCAL_MODEL", "Qwen/Qwen2.5-3B-Instruct")
        if key not in _LOCAL_CACHE:
            _LOCAL_CACHE[key] = TransformersLLM(model=key)
        return _LOCAL_CACHE[key]
    if backend == "hf":
        return HFInferenceLLM(**kwargs)
    if backend in {"openai", "ollama"}:
        return OpenAICompatLLM(**kwargs)
    if backend == "mock":
        return MockLLM()
    raise ValueError(f"Unknown backend: {backend}")
