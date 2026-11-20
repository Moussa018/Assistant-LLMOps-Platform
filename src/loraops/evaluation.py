"""Evaluate stage: score an adapter (or the base model) on the frozen test set."""

import time
from collections import Counter, defaultdict

from .config import TenantConfig
from .predictors import Predictor
from .prompts import build_messages, parse_label


def few_shot_examples(train: list[dict], labels: list[str], k: int) -> list[dict]:
    """Deterministic, label-balanced examples for the few-shot baseline."""
    by_label = defaultdict(list)
    for r in train:
        by_label[r["label"]].append(r)
    picks: list[dict] = []
    i = 0
    while len(picks) < k and any(by_label.values()):
        lab = labels[i % len(labels)]
        if by_label[lab]:
            picks.append(by_label[lab].pop(0))
        i += 1
        if i > k * len(labels) * 2:
            break
    return picks


def score(gold: list[str], pred: list[str | None], labels: list[str]) -> dict:
    n = len(gold)
    correct = sum(g == p for g, p in zip(gold, pred, strict=True))
    per_label = {}
    for lab in labels:
        tp = sum(g == lab and p == lab for g, p in zip(gold, pred, strict=True))
        fp = sum(g != lab and p == lab for g, p in zip(gold, pred, strict=True))
        fn = sum(g == lab and p != lab for g, p in zip(gold, pred, strict=True))
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        per_label[lab] = {
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "support": tp + fn,
        }
    present = [lab for lab in labels if per_label[lab]["support"] > 0]
    macro = sum(per_label[lab]["f1"] for lab in present) / len(present) if present else 0.0
    confusion = Counter(
        f"{g} -> {p or 'INVALID'}" for g, p in zip(gold, pred, strict=True) if g != p
    )
    return {
        "n": n,
        "accuracy": round(correct / n, 4) if n else 0.0,
        "macro_f1": round(macro, 4),
        "invalid_rate": round(sum(p is None for p in pred) / n, 4) if n else 0.0,
        "per_label": per_label,
        "top_confusions": dict(confusion.most_common(5)),
    }


def evaluate(
    predictor: Predictor,
    tenant: TenantConfig,
    test: list[dict],
    adapter: str | None = None,
    examples: list[dict] | None = None,
) -> dict:
    convs = [build_messages(tenant, r["text"], examples) for r in test]
    t0 = time.perf_counter()
    raw = predictor.predict(convs, adapter)
    elapsed = time.perf_counter() - t0
    preds = [parse_label(o, tenant.labels) for o in raw]
    result = score([r["label"] for r in test], preds, tenant.labels)
    result["ms_per_example"] = round(elapsed * 1000 / max(1, len(test)), 1)
    result["errors"] = [
        {"text": r["text"], "gold": r["label"], "output": o}
        for r, o, p in zip(test, raw, preds, strict=True)
        if p != r["label"]
    ][:10]
    return result
