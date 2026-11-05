from conftest import FakePredictor, FakeTrainer

from loraops.pipeline import run
from loraops.registry import Registry


def test_full_lifecycle(cfg, acme):
    reg = Registry(cfg.registry.root)
    predictors = []

    def make():
        p = FakePredictor(acme)
        predictors.append(p)
        return p

    # 1. first adapter: clearly better than prompting -> production
    r1 = run(cfg, acme, reg, FakeTrainer(0.9), make)
    assert r1.decision.passed and r1.promoted and r1.version == "v1"
    assert set(r1.baselines) == {"base_zero_shot", "base_5_shot"}
    assert reg.production("acme")[0] == "v1"
    assert reg.testset_path("acme").is_file()

    # 2. a worse retrain is rejected; v1 stays live
    r2 = run(cfg, acme, reg, FakeTrainer(0.8), make)
    assert not r2.decision.passed and not r2.promoted
    failed = [c["name"] for c in reg.card("acme", "v2")["gate"]["checks"] if not c["passed"]]
    assert failed == ["no regression vs production"]
    assert reg.production("acme")[0] == "v1"
    # same frozen test set -> production score reused, production adapter never reloaded
    assert "acme-v1" not in predictors[-1].loaded

    # 3. a better retrain replaces it, with lineage recorded
    r3 = run(cfg, acme, reg, FakeTrainer(0.97), make)
    assert r3.promoted and reg.production("acme")[0] == "v3"
    card = reg.card("acme", "v3")
    assert card["parent"] == "v1" and card["compared_to_production"]["version"] == "v1"
    assert reg.card("acme", "v1")["status"] == "archived"
    assert set(card["timings"]) == {"data", "train", "evaluate", "gate"}


def test_new_test_set_forces_production_reevaluation(cfg, acme):
    reg = Registry(cfg.registry.root)
    run(cfg, acme, reg, FakeTrainer(0.9), lambda: FakePredictor(acme))
    p = FakePredictor(acme)
    cfg.train.seed = 123  # a genuinely different test split
    run(cfg, acme, reg, FakeTrainer(0.95), lambda: p, refresh_testset=True)
    assert "acme-v1" in p.loaded


def test_no_promote_keeps_candidate(cfg, acme):
    reg = Registry(cfg.registry.root)
    r = run(cfg, acme, reg, FakeTrainer(0.9), lambda: FakePredictor(acme), promote=False)
    assert r.decision.passed and not r.promoted
    assert reg.card("acme", "v1")["status"] == "candidate" and reg.production("acme") is None
