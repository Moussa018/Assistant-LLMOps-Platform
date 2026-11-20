"""Gate stage: decide whether a candidate adapter may go to production.

An adapter only ships if it is measurably worth it: better than simply prompting the base model,
and not worse than what users already have."""

from dataclasses import asdict, dataclass

from .config import GateCfg


@dataclass
class Check:
    name: str
    passed: bool
    detail: str


@dataclass
class Decision:
    passed: bool
    checks: list[Check]

    def to_dict(self) -> dict:
        return {"passed": self.passed, "checks": [asdict(c) for c in self.checks]}

    def summary(self) -> str:
        lines = [("PASS" if self.passed else "FAIL")]
        lines += [f"  [{'x' if c.passed else ' '}] {c.name}: {c.detail}" for c in self.checks]
        return "\n".join(lines)


def decide(
    candidate: dict, baselines: dict[str, dict], production: dict | None, cfg: GateCfg
) -> Decision:
    m = cfg.metric
    checks = [
        Check(
            "enough test examples",
            candidate["n"] >= cfg.min_test_examples,
            f"{candidate['n']} (min {cfg.min_test_examples})",
        ),
        Check(
            "minimum quality",
            candidate[m] >= cfg.min_score,
            f"{m} {candidate[m]:.4f} (min {cfg.min_score})",
        ),
        Check(
            "valid outputs",
            candidate["invalid_rate"] <= cfg.max_invalid_rate,
            f"invalid rate {candidate['invalid_rate']:.2%} (max {cfg.max_invalid_rate:.2%})",
        ),
    ]
    if baselines:
        best_name, best = max(baselines.items(), key=lambda kv: kv[1][m])
        need = best[m] + cfg.min_gain_over_baseline
        checks.append(
            Check(
                "beats prompting baseline",
                candidate[m] >= need,
                f"{m} {candidate[m]:.4f} vs {best_name} {best[m]:.4f}"
                f" (+{cfg.min_gain_over_baseline} required)",
            )
        )
    if production is not None:
        floor = production[m] - cfg.max_regression_vs_production
        checks.append(
            Check(
                "no regression vs production",
                candidate[m] >= floor,
                f"{m} {candidate[m]:.4f} vs production {production[m]:.4f}"
                f" (max drop {cfg.max_regression_vs_production})",
            )
        )
    return Decision(all(c.passed for c in checks), checks)
