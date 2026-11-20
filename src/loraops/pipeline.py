"""The pipeline: data -> train -> evaluate -> gate -> register -> (promote).

candidate is scored against:
  * the base model, zero-shot      (is fine-tuning better than just asking?)
  * the base model, few-shot       (...even with examples in the prompt?)
  * the current production adapter (is it better than what users already have?)
"""

import logging
import shutil
import tempfile
import time
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from .config import PipelineConfig, TenantConfig
from .data import dataset_hash, prepare
from .evaluation import evaluate, few_shot_examples
from .gate import Decision, decide
from .predictors import Predictor
from .registry import Registry, adapter_name
from .training import Trainer

log = logging.getLogger("loraops")


@dataclass
class RunResult:
    tenant: str
    version: str
    decision: Decision
    promoted: bool
    candidate: dict
    baselines: dict
    production: dict | None
    timings: dict


@contextmanager
def _stage(name: str, timings: dict):
    log.info("[%s] start", name)
    t0 = time.perf_counter()
    yield
    timings[name] = round(time.perf_counter() - t0, 2)
    log.info("[%s] done in %.1fs", name, timings[name])


def run(
    cfg: PipelineConfig,
    tenant: TenantConfig,
    registry: Registry,
    trainer: Trainer,
    make_predictor: Callable[[], Predictor],
    *,
    refresh_testset: bool = False,
    promote: bool | None = None,
    workdir: Path | None = None,
) -> RunResult:
    timings: dict[str, float] = {}
    promote = cfg.gate.auto_promote if promote is None else promote
    own_workdir = workdir is None
    workdir = Path(workdir or tempfile.mkdtemp(prefix=f"loraops-{tenant.name}-"))

    try:
        with _stage("data", timings):
            train, test, report = prepare(
                tenant,
                cfg.data,
                registry.testset_path(tenant.name),
                cfg.train.seed,
                refresh_testset,
            )
            for w in report.warnings[:5]:
                log.warning("data: %s", w)
            log.info(
                "data: %d train / %d test, labels %s", len(train), len(test), report.label_counts
            )

        with _stage("train", timings):
            trained = trainer.train(cfg, tenant, train, workdir)

        test_hash = dataset_hash(test)
        prod = registry.production(tenant.name)
        with _stage("evaluate", timings):
            predictor = make_predictor()
            try:
                cand_name = f"{tenant.name}-candidate"
                predictor.add_adapter(cand_name, trained.adapter_dir)
                candidate = evaluate(predictor, tenant, test, adapter=cand_name)
                shots = few_shot_examples(train, tenant.labels, cfg.eval.few_shot_k)
                baselines = {
                    "base_zero_shot": evaluate(predictor, tenant, test),
                    f"base_{len(shots)}_shot": evaluate(predictor, tenant, test, examples=shots),
                }
                production = None
                if prod:
                    pv, pcard = prod
                    cached = pcard.get("eval", {})
                    if cached and pcard.get("testset_hash") == test_hash:
                        production = cached  # same frozen test set: reuse the score
                    else:
                        pname = adapter_name(tenant.name, pv)
                        predictor.add_adapter(pname, registry.adapter_path(tenant.name, pv))
                        production = evaluate(predictor, tenant, test, adapter=pname)
            finally:
                predictor.close()

        with _stage("gate", timings):
            decision = decide(candidate, baselines, production, cfg.gate)
            log.info("gate: %s", decision.summary())

        with _stage("register", timings):
            card = {
                "base_model": cfg.base.model,
                "train": trained.metrics,
                "eval": candidate,
                "baselines": {k: _summary(v) for k, v in baselines.items()},
                "compared_to_production": (
                    {"version": prod[0], **_summary(production)} if prod and production else None
                ),
                "gate": decision.to_dict(),
                "data": {
                    "train_hash": dataset_hash(train),
                    "train_examples": len(train),
                    "label_counts": report.label_counts,
                },
                "testset_hash": test_hash,
                "parent": prod[0] if prod else None,
                "config": {"lora": cfg.lora.model_dump(), "train": cfg.train.model_dump()},
                "timings": timings,
            }
            version = registry.add(
                tenant.name,
                trained.adapter_dir,
                card,
                status="candidate" if decision.passed else "rejected",
            )
            promoted = False
            if decision.passed and promote:
                registry.promote(tenant.name, version, cfg.base.model)
                promoted = True
            log.info(
                "registry: %s/%s %s",
                tenant.name,
                version,
                "PROMOTED to production"
                if promoted
                else ("candidate" if decision.passed else "rejected"),
            )

        return RunResult(
            tenant.name, version, decision, promoted, candidate, baselines, production, timings
        )
    finally:
        if own_workdir:
            shutil.rmtree(workdir, ignore_errors=True)


def _summary(m: dict | None) -> dict:
    if not m:
        return {}
    return {k: m[k] for k in ("accuracy", "macro_f1", "invalid_rate", "n") if k in m}
