"""Data stage: validate a tenant's dataset, freeze a test set, build the training set.

The test set is frozen the first time a tenant runs and stored in the registry. Later runs only
add new data to the training side, so every adapter version is scored on the same examples and
"v3 is better than v2" is a fair comparison."""

import hashlib
import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from .config import DataCfg, TenantConfig


class DataError(ValueError):
    pass


@dataclass
class DataReport:
    n_rows: int
    n_valid: int
    label_counts: dict[str, int]
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def _norm(text: str) -> str:
    return " ".join(text.lower().split())


def read_jsonl(path: str | Path) -> list[dict]:
    rows = []
    for n, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as e:
            raise DataError(f"{path}:{n}: invalid JSON ({e.msg})") from e
    return rows


def write_jsonl(path: str | Path, rows: list[dict]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8"
    )


def validate(rows: list[dict], tenant: TenantConfig, cfg: DataCfg) -> tuple[list[dict], DataReport]:
    """Return clean, deduplicated rows and a report. Never silently drops conflicting labels."""
    errors, warnings, clean = [], [], []
    seen: dict[str, str] = {}
    labels = set(tenant.labels)
    for i, r in enumerate(rows, 1):
        text, label = r.get("text"), r.get("label")
        if not isinstance(text, str) or not text.strip():
            errors.append(f"row {i}: missing or empty 'text'")
            continue
        if label not in labels:
            errors.append(f"row {i}: label {label!r} is not one of {sorted(labels)}")
            continue
        if len(text) > cfg.max_text_chars:
            errors.append(f"row {i}: text longer than {cfg.max_text_chars} chars")
            continue
        key = _norm(text)
        if key in seen:
            if seen[key] != label:
                errors.append(f"row {i}: same text labelled both {seen[key]!r} and {label!r}")
            else:
                warnings.append(f"row {i}: duplicate removed")
            continue
        seen[key] = label
        clean.append({"text": text.strip(), "label": label})

    counts = Counter(r["label"] for r in clean)
    for lab in tenant.labels:
        if counts[lab] < cfg.min_examples_per_label:
            errors.append(
                f"label {lab!r} has {counts[lab]} examples (minimum {cfg.min_examples_per_label})"
            )
    report = DataReport(len(rows), len(clean), dict(sorted(counts.items())), errors, warnings)
    return clean, report


def stratified_split(rows: list[dict], test_fraction: float, seed: int) -> tuple[list, list]:
    by_label: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_label[r["label"]].append(r)
    rng = random.Random(seed)
    train, test = [], []
    for label in sorted(by_label):
        items = sorted(by_label[label], key=lambda r: r["text"])
        rng.shuffle(items)
        k = max(1, round(len(items) * test_fraction))
        test += items[:k]
        train += items[k:]
    rng.shuffle(train)
    return train, test


def dataset_hash(rows: list[dict]) -> str:
    canon = sorted(f"{_norm(r['text'])}\t{r['label']}" for r in rows)
    return hashlib.sha256("\n".join(canon).encode()).hexdigest()[:16]


def prepare(
    tenant: TenantConfig, cfg: DataCfg, testset_path: Path, seed: int, refresh_testset: bool = False
) -> tuple[list[dict], list[dict], DataReport]:
    """Validate tenant data, return (train, test, report). Freezes the test set on first use."""
    clean, report = validate(read_jsonl(tenant.data), tenant, cfg)
    if not report.ok:
        raise DataError("dataset failed validation:\n  " + "\n  ".join(report.errors))

    if testset_path.is_file() and not refresh_testset:
        test = read_jsonl(testset_path)
        test_keys = {_norm(r["text"]) for r in test}
        train = [r for r in clean if _norm(r["text"]) not in test_keys]
        missing_labels = set(tenant.labels) - {r["label"] for r in test}
        if missing_labels:
            report.warnings.append(f"frozen test set has no examples of {sorted(missing_labels)}")
    else:
        train, test = stratified_split(clean, cfg.test_fraction, seed)
        write_jsonl(testset_path, test)
    return train, test, report
