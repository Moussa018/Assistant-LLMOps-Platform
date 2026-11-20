"""Ways to run a model for evaluation. Same interface, different backends:

* HFPredictor: loads the base model once (e.g. on Colab) and hot-swaps LoRA adapters with peft.
* OpenAIPredictor: calls a running vLLM server, where each adapter is a model name.
"""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Protocol

import httpx

from .config import PipelineConfig


class Predictor(Protocol):
    def add_adapter(self, name: str, path: Path) -> None: ...

    def predict(self, conversations: list[list[dict]], adapter: str | None) -> list[str]:
        """Greedy-decode a reply for each conversation; adapter=None means the base model."""
        ...

    def close(self) -> None: ...


class HFPredictor:
    def __init__(self, cfg: PipelineConfig):
        from .training import load_base

        self.cfg = cfg
        self.tok, self.base = load_base(cfg, for_training=False)
        self.tok.padding_side = "left"
        self.model = self.base
        self.adapters: set[str] = set()

    def add_adapter(self, name: str, path: Path) -> None:
        from peft import PeftModel

        if name in self.adapters:
            return
        if not self.adapters:
            self.model = PeftModel.from_pretrained(self.base, str(path), adapter_name=name)
        else:
            self.model.load_adapter(str(path), adapter_name=name)
        self.model.eval()
        self.adapters.add(name)

    def predict(self, conversations: list[list[dict]], adapter: str | None) -> list[str]:
        import contextlib

        import torch

        if adapter is not None:
            if adapter not in self.adapters:
                raise KeyError(f"adapter {adapter!r} not loaded")
            self.model.set_adapter(adapter)
            ctx = contextlib.nullcontext()
        else:
            ctx = self.model.disable_adapter() if self.adapters else contextlib.nullcontext()

        outputs: list[str] = []
        bs = self.cfg.eval.batch_size
        with ctx, torch.inference_mode():
            for i in range(0, len(conversations), bs):
                prompts = [
                    self.tok.apply_chat_template(c, tokenize=False, add_generation_prompt=True)
                    for c in conversations[i : i + bs]
                ]
                enc = self.tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False)
                enc = {k: v.to(self.model.device) for k, v in enc.items()}
                gen = self.model.generate(
                    **enc,
                    max_new_tokens=self.cfg.eval.max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tok.pad_token_id,
                )
                new = gen[:, enc["input_ids"].shape[1] :]
                outputs += self.tok.batch_decode(new, skip_special_tokens=True)
        return outputs

    def close(self) -> None:
        import torch

        del self.model, self.base
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class OpenAIPredictor:
    """Evaluate against a live vLLM server (adapters must already be loaded there)."""

    def __init__(self, cfg: PipelineConfig, base_url: str | None = None, workers: int = 8):
        self.cfg = cfg
        self.client = httpx.Client(base_url=(base_url or cfg.serving.vllm_url), timeout=120)
        self.workers = workers

    def add_adapter(self, name: str, path: Path) -> None:
        return None  # managed by the gateway's sync

    def _one(self, conv: list[dict], model: str) -> str:
        r = self.client.post(
            "/v1/chat/completions",
            json={
                "model": model,
                "messages": conv,
                "temperature": 0,
                "max_tokens": self.cfg.eval.max_new_tokens,
            },
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"] or ""

    def predict(self, conversations: list[list[dict]], adapter: str | None) -> list[str]:
        model = adapter or self.cfg.base.model
        with ThreadPoolExecutor(self.workers) as pool:
            return list(pool.map(lambda c: self._one(c, model), conversations))

    def close(self) -> None:
        self.client.close()
