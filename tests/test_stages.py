import json

import pytest

from loraops.config import GateCfg
from loraops.data import DataError, dataset_hash, prepare, read_jsonl, stratified_split, validate
from loraops.evaluation import few_shot_examples, score
from loraops.gate import decide
from loraops.prompts import build_messages, parse_label
from loraops.registry import Registry, RegistryError


# ----------------------------------------------------------------------------- data
def test_demo_data_is_valid(acme, cfg):
    clean, report = validate(read_jsonl(acme.data), acme, cfg.data)
    assert report.ok and report.n_valid == 200
    assert set(report.label_counts) == set(acme.labels)


def test_validate_catches_bad_rows(acme, cfg):
    rows = [
        {"text": "where is my parcel", "label": "shipping"},
        {"text": "Where is my  parcel", "label": "billing"},  # same text, other label
        {"text": "", "label": "billing"},
        {"text": "hello", "label": "not_a_label"},
    ]
    _, report = validate(rows, acme, cfg.data)
    joined = "\n".join(report.errors)
    assert "labelled both" in joined and "empty" in joined and "not_a_label" in joined
    assert "minimum" in joined  # labels without enough examples


def test_split_is_stratified_and_deterministic(acme, cfg):
    rows, _ = validate(read_jsonl(acme.data), acme, cfg.data)
    train, test = stratified_split(rows, 0.2, seed=1)
    assert len(test) == 40 and len(train) == 160
    assert {r["label"] for r in test} == set(acme.labels)
    assert stratified_split(rows, 0.2, seed=1) == (train, test)


def test_test_set_is_frozen_across_runs(acme, cfg, tmp_path):
    path = tmp_path / "testset.jsonl"
    _, test1, _ = prepare(acme, cfg.data, path, seed=1)
    _, test2, _ = prepare(acme, cfg.data, path, seed=999)  # different seed, same frozen set
    assert test1 == test2
    train, test, _ = prepare(acme, cfg.data, path, seed=1)
    assert not {r["text"] for r in train} & {r["text"] for r in test}


def test_prepare_refuses_invalid_data(acme, cfg, tmp_path):
    bad = tmp_path / "bad.jsonl"
    bad.write_text(json.dumps({"text": "x", "label": "nope"}) + "\n")
    with pytest.raises(DataError):
        prepare(acme.model_copy(update={"data": str(bad)}), cfg.data, tmp_path / "t.jsonl", 1)


def test_dataset_hash_ignores_order():
    a = [{"text": "A", "label": "x"}, {"text": "b", "label": "y"}]
    assert dataset_hash(a) == dataset_hash(list(reversed(a)))


# -------------------------------------------------------------------------- prompts
def test_prompt_lists_categories_and_examples(acme):
    msgs = build_messages(acme, "hi", [{"text": "where is it", "label": "shipping"}])
    assert "product_question" in msgs[0]["content"] and len(msgs) == 4


def test_parse_label(acme):
    L = acme.labels
    assert parse_label("shipping", L) == "shipping"
    assert parse_label("  Shipping.\n", L) == "shipping"
    assert parse_label("Category: returns", L) == "returns"
    assert parse_label("billing or shipping", L) is None  # ambiguous
    assert parse_label("I don't know", L) is None


# ------------------------------------------------------------------------ eval/gate
def test_score_metrics():
    m = score(["a", "a", "b", "b"], ["a", "b", "b", None], ["a", "b"])
    assert m["accuracy"] == 0.5 and m["invalid_rate"] == 0.25
    assert m["per_label"]["a"]["recall"] == 0.5


def test_few_shot_examples_are_balanced(acme, cfg):
    rows, _ = validate(read_jsonl(acme.data), acme, cfg.data)
    shots = few_shot_examples(rows, acme.labels, 5)
    assert {s["label"] for s in shots} == set(acme.labels)


def _m(f1, invalid=0.0, n=40):
    return {"macro_f1": f1, "invalid_rate": invalid, "n": n}


def test_gate():
    g = GateCfg()
    assert decide(_m(0.9), {"base": _m(0.6)}, None, g).passed
    assert not decide(_m(0.62), {"base": _m(0.6)}, None, g).passed  # not enough gain
    assert not decide(_m(0.9), {"base": _m(0.6)}, _m(0.95), g).passed  # regression
    assert decide(_m(0.945), {"base": _m(0.6)}, _m(0.95), g).passed  # within tolerance
    assert not decide(_m(0.9, invalid=0.2), {"base": _m(0.6)}, None, g).passed
    assert not decide(_m(0.9, n=5), {"base": _m(0.6)}, None, g).passed
    # beats a broken baseline by a mile, but is still bad: blocked by the absolute floor
    assert not decide(_m(0.3), {"base": _m(0.0)}, None, g).passed


# -------------------------------------------------------------------------- registry
def _adapter(tmp_path, base="fake/base-model"):
    d = tmp_path / "adapter"
    d.mkdir(exist_ok=True)
    (d / "adapter_config.json").write_text(json.dumps({"base_model_name_or_path": base}))
    return d


def test_registry_lifecycle(tmp_path):
    reg = Registry(tmp_path / "reg")
    card = {"base_model": "fake/base-model", "eval": {"macro_f1": 0.9}}
    v1 = reg.add("acme", _adapter(tmp_path), card, "candidate")
    v2 = reg.add("acme", _adapter(tmp_path), card, "candidate")
    assert (v1, v2) == ("v1", "v2")
    reg.promote("acme", "v1", "fake/base-model")
    reg.promote("acme", "v2", "fake/base-model")
    assert reg.production("acme")[0] == "v2" and reg.card("acme", "v1")["status"] == "archived"
    assert reg.rollback("acme") == "v1"
    assert reg.card("acme", "v2")["status"] == "rejected"
    with pytest.raises(RegistryError):
        reg.rollback("acme")  # nothing before v1


def test_registry_guards(tmp_path):
    reg = Registry(tmp_path / "reg")
    with pytest.raises(RegistryError):  # card and adapter disagree on base model
        reg.add(
            "acme",
            _adapter(tmp_path, base="other/base"),
            {"base_model": "fake/base-model"},
            "candidate",
        )
    v = reg.add("acme", _adapter(tmp_path), {"base_model": "fake/base-model"}, "rejected")
    with pytest.raises(RegistryError):  # gate-rejected needs force
        reg.promote("acme", v, "fake/base-model")
    with pytest.raises(RegistryError):  # serving a different base model
        reg.promote("acme", v, "another/base", force=True)
    reg.promote("acme", v, "fake/base-model", force=True)
    assert reg.production("acme")[0] == v
