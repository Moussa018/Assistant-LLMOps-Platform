"""Pipeline and tenant configuration (TOML files validated with pydantic)."""

import sys
from pathlib import Path

from pydantic import BaseModel, Field, model_validator

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover
    import tomli as tomllib


class BaseModelCfg(BaseModel):
    model: str = "Qwen/Qwen2.5-0.5B-Instruct"
    max_seq_len: int = 512


class LoraCfg(BaseModel):
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: list[str] = Field(
        default=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )


class TrainCfg(BaseModel):
    epochs: float = 3
    learning_rate: float = 2e-4
    batch_size: int = 8
    grad_accum: int = 1
    load_in_4bit: bool = True  # QLoRA; ignored automatically when no CUDA GPU is present
    seed: int = 42


class DataCfg(BaseModel):
    test_fraction: float = 0.2
    min_examples_per_label: int = 5
    max_text_chars: int = 2000


class EvalCfg(BaseModel):
    few_shot_k: int = 5
    batch_size: int = 16
    max_new_tokens: int = 16


class GateCfg(BaseModel):
    metric: str = "macro_f1"
    min_test_examples: int = 20
    # Absolute floor: beating a weak baseline is not enough to ship.
    min_score: float = 0.6
    # The adapter must beat the best prompting baseline (zero- or few-shot base model) by this.
    min_gain_over_baseline: float = 0.05
    # ...and must not be worse than the adapter currently in production by more than this.
    max_regression_vs_production: float = 0.01
    max_invalid_rate: float = 0.05
    auto_promote: bool = True


class RegistryCfg(BaseModel):
    root: str = "registry"


class ServingCfg(BaseModel):
    vllm_url: str = "http://localhost:8001"
    # Path at which the vLLM server sees the registry root (shared volume).
    registry_mount: str = "/registry"


class PipelineConfig(BaseModel):
    base: BaseModelCfg = BaseModelCfg()
    lora: LoraCfg = LoraCfg()
    train: TrainCfg = TrainCfg()
    data: DataCfg = DataCfg()
    eval: EvalCfg = EvalCfg()
    gate: GateCfg = GateCfg()
    registry: RegistryCfg = RegistryCfg()
    serving: ServingCfg = ServingCfg()
    tenants_dir: str = "configs/tenants"


class TenantConfig(BaseModel):
    name: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]{0,40}$")
    display_name: str
    description: str = ""
    data: str
    labels: list[str]
    label_descriptions: dict[str, str] = {}

    @model_validator(mode="after")
    def _check(self):
        if len(self.labels) < 2 or len(set(self.labels)) != len(self.labels):
            raise ValueError("labels must contain at least 2 distinct values")
        unknown = set(self.label_descriptions) - set(self.labels)
        if unknown:
            raise ValueError(f"label_descriptions has unknown labels: {sorted(unknown)}")
        return self


def _read(path: str | Path) -> dict:
    return tomllib.loads(Path(path).read_text(encoding="utf-8"))


def load_pipeline_config(path: str | Path | None = None) -> PipelineConfig:
    return PipelineConfig(**_read(path)) if path else PipelineConfig()


def load_tenant(name: str, cfg: PipelineConfig) -> TenantConfig:
    path = Path(cfg.tenants_dir) / f"{name}.toml"
    if not path.is_file():
        raise FileNotFoundError(f"no tenant config at {path}")
    return TenantConfig(**_read(path))


def list_tenants(cfg: PipelineConfig) -> list[TenantConfig]:
    return [TenantConfig(**_read(p)) for p in sorted(Path(cfg.tenants_dir).glob("*.toml"))]
