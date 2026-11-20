import json

import httpx
import respx
from fastapi.testclient import TestClient

from loraops.gateway import create_app
from loraops.registry import Registry

VLLM = "http://vllm:8001"


class FakeVLLM:
    """Minimal stand-in for vLLM's multi-LoRA HTTP API."""

    def __init__(self, base):
        self.base = base
        self.loaded: dict[str, str] = {}
        self.chat_models: list[str] = []

    def install(self, router: respx.Router):
        router.get(f"{VLLM}/v1/models").mock(
            side_effect=lambda r: httpx.Response(
                200, json={"data": [{"id": self.base}] + [{"id": n} for n in self.loaded]}
            )
        )
        router.post(f"{VLLM}/v1/load_lora_adapter").mock(side_effect=self._load)
        router.post(f"{VLLM}/v1/unload_lora_adapter").mock(side_effect=self._unload)
        router.post(f"{VLLM}/v1/chat/completions").mock(side_effect=self._chat)

    def _load(self, req):
        body = json.loads(req.content)
        self.loaded[body["lora_name"]] = body["lora_path"]
        return httpx.Response(200, text="ok")

    def _unload(self, req):
        self.loaded.pop(json.loads(req.content)["lora_name"], None)
        return httpx.Response(200, text="ok")

    def _chat(self, req):
        body = json.loads(req.content)
        model = body["model"]
        if model != self.base and model not in self.loaded:
            return httpx.Response(404, json={"error": "model not found"})
        self.chat_models.append(model)
        answer = "shipping" if "parcel" in body["messages"][-1]["content"] else "no idea"
        return httpx.Response(200, json={"choices": [{"message": {"content": answer}}]})


def _add(reg, tmp_path, tenant, base):
    d = tmp_path / f"a-{tenant}"
    d.mkdir(exist_ok=True)
    (d / "adapter_config.json").write_text(json.dumps({"base_model_name_or_path": base}))
    card = {"base_model": base, "eval": {"accuracy": 0.9, "macro_f1": 0.9}}
    return reg.add(tenant, d, card, "candidate")


@respx.mock
def test_routing_sync_promote_and_rollback(cfg, tmp_path):
    cfg.serving.vllm_url = VLLM
    vllm = FakeVLLM(cfg.base.model)
    vllm.install(respx.mock)
    reg = Registry(cfg.registry.root)
    reg.promote("acme", _add(reg, tmp_path, "acme", cfg.base.model), cfg.base.model)

    app = create_app(cfg, httpx.AsyncClient(base_url=VLLM), sync_interval_s=0)
    with TestClient(app) as c:
        # before the first sync nothing is routable: requests fall back to the base model
        r = c.post("/v1/classify", json={"tenant": "acme", "text": "where is my parcel"}).json()
        assert r["fallback_to_base"] and r["label"] == "shipping"

        s = c.post("/admin/sync").json()
        assert s["loaded"] == ["acme-v1"] and s["active"] == {"acme": "v1"}
        assert vllm.loaded["acme-v1"] == "/registry/acme/v1/adapter"

        r = c.post("/v1/classify", json={"tenant": "acme", "text": "where is my parcel"}).json()
        assert r == {**r, "label": "shipping", "adapter_version": "v1", "fallback_to_base": False}
        assert vllm.chat_models[-1] == "acme-v1"

        # globex has no adapter yet: served by the base model
        r = c.post("/v1/classify", json={"tenant": "globex", "text": "hello"}).json()
        assert r["fallback_to_base"] and r["label"] is None and r["raw_output"] == "no idea"

        # promote v2 through the API: loaded first, then traffic moves, then v1 is unloaded
        v2 = _add(reg, tmp_path, "acme", cfg.base.model)
        s = c.post("/admin/promote", json={"tenant": "acme", "version": v2}).json()
        assert s["loaded"] == ["acme-v2"] and s["unloaded"] == ["acme-v1"]
        c.post("/v1/classify", json={"tenant": "acme", "text": "my parcel"})
        assert vllm.chat_models[-1] == "acme-v2"

        s = c.post("/admin/rollback", json={"tenant": "acme"}).json()
        assert s["active"] == {"acme": "v1"} and set(vllm.loaded) == {"acme-v1"}

        tenants = c.get("/v1/tenants").json()
        assert tenants["acme"]["serving_version"] == "v1"
        assert c.post("/v1/classify", json={"tenant": "nobody", "text": "x"}).status_code == 404
        assert "loraops_requests_total" in c.get("/metrics").text


@respx.mock
def test_model_server_down(cfg):
    cfg.serving.vllm_url = VLLM
    respx.post(f"{VLLM}/v1/chat/completions").mock(side_effect=httpx.ConnectError("down"))
    respx.get(f"{VLLM}/v1/models").mock(side_effect=httpx.ConnectError("down"))
    app = create_app(cfg, httpx.AsyncClient(base_url=VLLM), sync_interval_s=0)
    with TestClient(app) as c:
        assert c.post("/v1/classify", json={"tenant": "acme", "text": "x"}).status_code == 503
        assert c.post("/admin/sync").status_code == 503


def test_admin_token(cfg, monkeypatch):
    monkeypatch.setenv("LORAOPS_ADMIN_TOKEN", "s3cret")
    app = create_app(cfg, httpx.AsyncClient(base_url=VLLM), sync_interval_s=0)
    with TestClient(app) as c:
        assert c.post("/admin/rollback", json={"tenant": "acme"}).status_code == 401
        r = c.post("/admin/rollback", json={"tenant": "acme"}, headers={"x-admin-token": "s3cret"})
        assert r.status_code == 409  # authorized; nothing to roll back
