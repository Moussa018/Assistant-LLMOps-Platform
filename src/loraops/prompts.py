"""The one place prompts are built.

Training, evaluation and the serving gateway all call `build_messages`, so the adapter is always
served with exactly the prompt it was trained on (no training/serving skew)."""

import re

from .config import TenantConfig


def system_prompt(tenant: TenantConfig) -> str:
    lines = [
        f"You route customer support tickets for {tenant.display_name}"
        + (f" ({tenant.description})." if tenant.description else "."),
        "Classify the ticket into exactly one category.",
        "Categories:",
    ]
    for label in tenant.labels:
        desc = tenant.label_descriptions.get(label)
        lines.append(f"- {label}: {desc}" if desc else f"- {label}")
    lines.append("Answer with the category name only.")
    return "\n".join(lines)


def build_messages(
    tenant: TenantConfig, text: str, examples: list[dict] | None = None
) -> list[dict[str, str]]:
    msgs = [{"role": "system", "content": system_prompt(tenant)}]
    for ex in examples or []:
        msgs.append({"role": "user", "content": ex["text"]})
        msgs.append({"role": "assistant", "content": ex["label"]})
    msgs.append({"role": "user", "content": text})
    return msgs


def parse_label(output: str, labels: list[str]) -> str | None:
    """Map raw model output to a label, or None if it is not a valid answer."""
    clean = output.strip().strip("`'\".").strip().lower()
    by_lower = {lab.lower(): lab for lab in labels}
    if clean in by_lower:
        return by_lower[clean]
    first_line = clean.splitlines()[0].strip() if clean else ""
    if first_line in by_lower:
        return by_lower[first_line]
    # Exactly one label mentioned as a whole word: accept it.
    hits = [lab for low, lab in by_lower.items() if re.search(rf"\b{re.escape(low)}\b", clean)]
    return hits[0] if len(hits) == 1 else None
