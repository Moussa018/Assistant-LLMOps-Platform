import hashlib
import json
from pathlib import Path

import pytest

from loraops.config import PipelineConfig, load_tenant
from loraops.data import read_jsonl
from loraops.training import TrainResult

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _repo_cwd(monkeypatch):
    monkeypatch.chdir(ROOT)  # tenant configs reference data/ relative to the repo


@pytest.fixture
def cfg(tmp_path) -> PipelineConfig:
    c = PipelineConfig(tenants_dir=str(ROOT / "configs/tenants"))
    c.registry.root = str(tmp_path / "registry")
    c.base.model = "fake/base-model"
    return c


@pytest.fixture
def acme(cfg):
    return load_tenant("acme", cfg)


class FakeTrainer:
    """Writes a fake adapter whose 'quality' controls how well the FakePredictor does with it."""

    def __init__(self, quality: float):
        self.quality = quality
        self.calls = 0

    def train(self, cfg, tenant, rows, out_dir):
        self.calls += 1
        d = Path(out_dir) / "adapter"
        d.mkdir(parents=True, exist_ok=True)
        (d / "adapter_config.json").write_text(
            json.dumps({"base_model_name_or_path": cfg.base.model, "r": cfg.lora.r})
        )
        (d / "quality.txt").write_text(str(self.quality))
        return TrainResult(d, {"train_loss": 0.1, "train_examples": len(rows)})


class FakePredictor:
    """Answers correctly for a deterministic fraction of examples = the adapter's quality.
    Base model: 50% zero-shot, 60% few-shot."""

    def __init__(self, tenant):
        self.labels = tenant.labels
        self.gold = {r["text"]: r["label"] for r in read_jsonl(tenant.data)}
        self.adapters: dict[str, float] = {}
        self.loaded: list[str] = []

    def add_adapter(self, name, path):
        self.adapters[name] = float((Path(path) / "quality.txt").read_text())
        self.loaded.append(name)

    def predict(self, conversations, adapter):
        few_shot = len(conversations[0]) > 2
        q = self.adapters[adapter] if adapter else (0.6 if few_shot else 0.5)
        out = []
        for conv in conversations:
            text = conv[-1]["content"]
            gold = self.gold[text]
            h = int(hashlib.md5(text.encode()).hexdigest(), 16) % 1000 / 1000
            if h < q:
                out.append(gold)
            else:
                wrong = [lab for lab in self.labels if lab != gold]
                out.append(wrong[int(h * 1000) % len(wrong)])
        return out

    def close(self):
        pass
