"""Serving gateway in front of one vLLM server running many LoRA adapters.

* Routes each tenant's request to that tenant's production adapter (or the base model if none).
* Builds the prompt with the same code used in training, so serving never drifts from training.
* Reconciles vLLM with the registry (load new production adapters, unload retired ones), on a
  timer and right after every promote / rollback. Traffic switches to a new version only once
  vLLM has confirmed it is loaded, so a promotion never routes requests to a missing adapter.

vLLM must run with --enable-lora and VLLM_ALLOW_RUNTIME_LORA_UPDATING=True.
"""

import asyncio
import contextlib
import logging
import os
import re
import time

import httpx
from fastapi import FastAPI, Header, HTTPException, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest
from pydantic import BaseModel

from .config import PipelineConfig, TenantConfig, list_tenants, load_pipeline_config
from .prompts import build_messages, parse_label
from .registry import Registry, RegistryError, adapter_name

log = logging.getLogger("loraops.gateway")
ADAPTER_RE = re.compile(r"^[a-z0-9][a-z0-9_-]*-v\d+$")

REQUESTS = Counter(
    "loraops_requests_total", "Classification requests", ["tenant", "version", "status"]
)
LATENCY = Histogram(
    "loraops_request_latency_seconds",
    "End-to-end latency",
    ["tenant"],
    buckets=(0.025, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10),
)
PREDICTIONS = Counter("loraops_predictions_total", "Predicted labels", ["tenant", "label"])
ACTIVE = Gauge("loraops_active_adapters", "Adapters loaded in vLLM and routable")
SYNC_ERRORS = Counter("loraops_sync_errors_total", "Failed reconciliations with vLLM")


class ClassifyRequest(BaseModel):
    tenant: str
    text: str


class TenantRef(BaseModel):
    tenant: str
    version: str | None = None


class Gateway:
    def __init__(self, cfg: PipelineConfig, client: httpx.AsyncClient):
        self.cfg = cfg
        self.client = client
        self.registry = Registry(cfg.registry.root)
        self.tenants: dict[str, TenantConfig] = {t.name: t for t in list_tenants(cfg)}
        self.active: dict[str, str] = {}  # tenant -> version currently routable
        self._lock = asyncio.Lock()

    async def sync(self) -> dict:
        async with self._lock:
            desired = {}
            for name in self.tenants:
                prod = self.registry.production(name)
                if prod:
                    v = prod[0]
                    desired[adapter_name(name, v)] = (name, v)
            r = await self.client.get("/v1/models")
            r.raise_for_status()
            loaded = {m["id"] for m in r.json()["data"] if ADAPTER_RE.match(m["id"])}

            newly = []
            for model_name, (tenant, version) in desired.items():
                if model_name in loaded:
                    continue
                path = f"{self.cfg.serving.registry_mount}/{tenant}/{version}/adapter"
                resp = await self.client.post(
                    "/v1/load_lora_adapter", json={"lora_name": model_name, "lora_path": path}
                )
                resp.raise_for_status()
                newly.append(model_name)

            # Switch traffic only after the new adapters are confirmed loaded...
            self.active = {tenant: version for tenant, version in desired.values()}
            # ...then retire what is no longer needed.
            retired = []
            for model_name in sorted(loaded - set(desired)):
                resp = await self.client.post(
                    "/v1/unload_lora_adapter", json={"lora_name": model_name}
                )
                if resp.status_code < 400:
                    retired.append(model_name)
            ACTIVE.set(len(self.active))
            return {
                "active": dict(sorted(self.active.items())),
                "loaded": newly,
                "unloaded": retired,
            }

    async def classify(self, tenant: TenantConfig, text: str) -> dict:
        version = self.active.get(tenant.name)
        model = adapter_name(tenant.name, version) if version else self.cfg.base.model
        t0 = time.perf_counter()
        r = await self.client.post(
            "/v1/chat/completions",
            json={
                "model": model,
                "messages": build_messages(tenant, text),
                "temperature": 0,
                "max_tokens": self.cfg.eval.max_new_tokens,
            },
        )
        r.raise_for_status()
        raw = r.json()["choices"][0]["message"]["content"] or ""
        label = parse_label(raw, tenant.labels)
        latency = time.perf_counter() - t0
        LATENCY.labels(tenant.name).observe(latency)
        REQUESTS.labels(tenant.name, version or "base", "ok" if label else "invalid").inc()
        PREDICTIONS.labels(tenant.name, label or "INVALID").inc()
        return {
            "tenant": tenant.name,
            "label": label,
            "adapter_version": version,
            "fallback_to_base": version is None,
            "raw_output": raw if label is None else None,
            "latency_ms": round(latency * 1000, 1),
        }


def create_app(
    cfg: PipelineConfig | None = None,
    client: httpx.AsyncClient | None = None,
    sync_interval_s: float | None = None,
) -> FastAPI:
    cfg = cfg or load_pipeline_config(os.environ.get("LORAOPS_CONFIG", "configs/pipeline.toml"))
    if os.environ.get("LORAOPS_VLLM_URL"):
        cfg.serving.vllm_url = os.environ["LORAOPS_VLLM_URL"]
    interval = (
        sync_interval_s
        if sync_interval_s is not None
        else float(os.environ.get("LORAOPS_SYNC_INTERVAL", "30"))
    )
    admin_token = os.environ.get("LORAOPS_ADMIN_TOKEN")

    async def _loop(gw: Gateway):
        while True:
            try:
                await gw.sync()
            except Exception as e:  # vLLM may still be starting: keep retrying
                SYNC_ERRORS.inc()
                log.warning("sync failed: %s", e)
            await asyncio.sleep(interval)

    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI):
        http = client or httpx.AsyncClient(base_url=cfg.serving.vllm_url, timeout=60)
        app.state.gw = Gateway(cfg, http)
        task = asyncio.create_task(_loop(app.state.gw)) if interval > 0 else None
        yield
        if task:
            task.cancel()
        await http.aclose()

    app = FastAPI(title="loraops gateway", lifespan=lifespan)

    def gw() -> Gateway:
        return app.state.gw

    def check_admin(token: str | None):
        if admin_token and token != admin_token:
            raise HTTPException(401, "admin token required")

    @app.get("/health")
    def health():
        return {"status": "ok", "base_model": cfg.base.model, "active": gw().active}

    @app.post("/v1/classify")
    async def classify(req: ClassifyRequest):
        tenant = gw().tenants.get(req.tenant)
        if tenant is None:
            raise HTTPException(404, f"unknown tenant {req.tenant!r}")
        try:
            return await gw().classify(tenant, req.text)
        except httpx.HTTPError as e:
            REQUESTS.labels(req.tenant, gw().active.get(req.tenant, "base"), "error").inc()
            raise HTTPException(503, f"model server unavailable: {e}") from e

    @app.get("/v1/tenants")
    def tenants():
        out = {}
        for name in gw().tenants:
            prod = gw().registry.production(name)
            out[name] = {
                "serving_version": gw().active.get(name),
                "registry_production": prod[0] if prod else None,
                "eval": {k: prod[1]["eval"][k] for k in ("accuracy", "macro_f1")} if prod else None,
            }
        return out

    @app.post("/admin/sync")
    async def sync(x_admin_token: str | None = Header(default=None)):
        check_admin(x_admin_token)
        try:
            return await gw().sync()
        except httpx.HTTPError as e:
            SYNC_ERRORS.inc()
            raise HTTPException(503, f"sync failed: {e}") from e

    @app.post("/admin/promote")
    async def promote(ref: TenantRef, x_admin_token: str | None = Header(default=None)):
        check_admin(x_admin_token)
        if not ref.version:
            raise HTTPException(422, "version is required")
        try:
            gw().registry.promote(ref.tenant, ref.version, cfg.base.model)
        except RegistryError as e:
            raise HTTPException(409, str(e)) from e
        return await gw().sync()

    @app.post("/admin/rollback")
    async def rollback(ref: TenantRef, x_admin_token: str | None = Header(default=None)):
        check_admin(x_admin_token)
        try:
            gw().registry.rollback(ref.tenant)
        except RegistryError as e:
            raise HTTPException(409, str(e)) from e
        return await gw().sync()

    @app.get("/metrics", include_in_schema=False)
    def metrics():
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    return app
