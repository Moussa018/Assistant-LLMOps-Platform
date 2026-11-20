"""Adapter registry: versioned adapters per tenant, with a model card for each version.

Layout (a plain directory, so it can live on a shared volume, S3 sync, or a Colab download):

    registry/
      index.json                      tenants -> versions, production pointer, promotion history
      <tenant>/testset.jsonl          frozen test set (same for every version)
      <tenant>/v1/adapter/            adapter_config.json + adapter_model.safetensors
      <tenant>/v1/card.json           metrics, gate decision, data hash, config, lineage

Statuses: candidate (passed gate, not live) · production · archived (was live)
          · rejected (failed gate or rolled back).
"""

import contextlib
import fcntl
import json
import os
import shutil
import time
from pathlib import Path


class RegistryError(RuntimeError):
    pass


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def adapter_name(tenant: str, version: str) -> str:
    """The model name an adapter is served under in vLLM."""
    return f"{tenant}-{version}"


class Registry:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._index = self.root / "index.json"

    # ------------------------------------------------------------ plumbing
    @contextlib.contextmanager
    def _locked(self):
        with open(self.root / ".lock", "w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)

    def _read(self) -> dict:
        if not self._index.is_file():
            return {"tenants": {}}
        return json.loads(self._index.read_text(encoding="utf-8"))

    def _write(self, data: dict) -> None:
        tmp = self._index.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, self._index)  # atomic: readers never see a half-written index

    def _tenant(self, data: dict, tenant: str) -> dict:
        return data["tenants"].setdefault(
            tenant, {"production": None, "history": [], "versions": {}}
        )

    # --------------------------------------------------------------- paths
    def testset_path(self, tenant: str) -> Path:
        return self.root / tenant / "testset.jsonl"

    def adapter_path(self, tenant: str, version: str) -> Path:
        return self.root / tenant / version / "adapter"

    # --------------------------------------------------------------- reads
    def tenants(self) -> dict:
        return self._read()["tenants"]

    def card(self, tenant: str, version: str) -> dict:
        try:
            return self._read()["tenants"][tenant]["versions"][version]
        except KeyError:
            raise RegistryError(f"unknown adapter {tenant}/{version}") from None

    def production(self, tenant: str) -> tuple[str, dict] | None:
        t = self._read()["tenants"].get(tenant)
        if not t or not t["production"]:
            return None
        return t["production"], t["versions"][t["production"]]

    # -------------------------------------------------------------- writes
    def add(self, tenant: str, adapter_dir: Path, card: dict, status: str) -> str:
        if status not in ("candidate", "rejected"):
            raise RegistryError("new versions are 'candidate' or 'rejected'")
        cfg_file = Path(adapter_dir) / "adapter_config.json"
        if not cfg_file.is_file():
            raise RegistryError(f"{adapter_dir} is not a LoRA adapter (no adapter_config.json)")
        base = json.loads(cfg_file.read_text()).get("base_model_name_or_path")
        if base != card["base_model"]:
            raise RegistryError(
                f"adapter was trained on {base!r}, card says {card['base_model']!r}"
            )
        with self._locked():
            data = self._read()
            t = self._tenant(data, tenant)
            version = f"v{len(t['versions']) + 1}"
            dest = self.adapter_path(tenant, version)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(adapter_dir, dest, ignore=shutil.ignore_patterns("_checkpoints"))
            card = {
                **card,
                "tenant": tenant,
                "version": version,
                "status": status,
                "created_at": _now(),
            }
            (dest.parent / "card.json").write_text(json.dumps(card, indent=2, ensure_ascii=False))
            t["versions"][version] = card
            self._write(data)
        return version

    def promote(self, tenant: str, version: str, base_model: str, force: bool = False) -> None:
        with self._locked():
            data = self._read()
            t = self._tenant(data, tenant)
            card = t["versions"].get(version)
            if card is None:
                raise RegistryError(f"unknown adapter {tenant}/{version}")
            if card["base_model"] != base_model:
                raise RegistryError(
                    f"{tenant}/{version} needs base {card['base_model']!r}, serving {base_model!r}"
                )
            if card["status"] == "rejected" and not force:
                raise RegistryError(f"{tenant}/{version} failed the gate; use force to override")
            if t["production"] == version:
                return
            if t["production"]:
                t["versions"][t["production"]]["status"] = "archived"
            card["status"] = "production"
            card["promoted_at"] = _now()
            t["production"] = version
            t["history"].append(version)
            self._write(data)

    def rollback(self, tenant: str) -> str:
        """Put the previous production version back. Returns the version now live."""
        with self._locked():
            data = self._read()
            t = self._tenant(data, tenant)
            if len(t["history"]) < 2:
                raise RegistryError(f"{tenant} has no previous production version")
            bad = t["history"].pop()
            prev = t["history"][-1]
            t["versions"][bad]["status"] = "rejected"
            t["versions"][bad]["rolled_back_at"] = time.strftime(
                "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
            )
            t["versions"][prev]["status"] = "production"
            t["production"] = prev
            self._write(data)
            return prev
