# loraops

**A pipeline that fine-tunes one LoRA adapter per customer, proves each one is worth shipping,
and serves all of them from a single base model on one GPU.**

```
 tenant data ─► validate ─► train (QLoRA) ─► evaluate ─► gate ─► registry ─► vLLM (multi-LoRA)
                  │                             │          │         │              ▲
           frozen test set            vs base zero-shot  pass/fail  versions,   gateway routes
           per tenant                 vs base few-shot              model cards, tenant → adapter,
                                      vs production                 promote /    syncs vLLM with
                                                                    rollback     the registry
```

## Why

Serving many LoRA adapters on one GPU is solved (vLLM, LoRAX). What is still glue code at most
companies is everything around it:

- **Is the fine-tune worth it?** With small datasets, a fine-tuned adapter is often *not* better
  than a good prompt. Every candidate here is scored against the base model zero-shot **and**
  few-shot; it ships only if it beats both by a margin.
- **Is the retrain an improvement?** Each tenant's test set is frozen on the first run, so every
  version is scored on the same examples, and a new version cannot replace production if it is worse.
- **What is live, and where did it come from?** A versioned registry with a model card per
  adapter: metrics, gate decision, dataset hashes, config, parent version, timings.
- **Switching versions safely.** The gateway loads a new adapter into vLLM, confirms it, moves
  traffic, and only then unloads the old one. Rollback is one command.

## Demo

Two fictional tenants with different ticket categories, so each needs its own adapter:

| tenant | business | labels |
|---|---|---|
| `acme` | online retailer | billing, shipping, returns, account, product_question |
| `globex` | B2B analytics SaaS | bug_report, feature_request, outage, integration, pricing |

200 synthetic tickets each (`scripts/make_demo_data.py`). Add a tenant by dropping a TOML file in
`configs/tenants/` and a `tickets.jsonl` with `{"text": ..., "label": ...}` lines.

## Quick start

### Train on Colab (free T4)

Open `notebooks/colab_pipeline.ipynb` in Colab, set the runtime to T4 GPU, run all cells. It
trains both tenants on `Qwen/Qwen2.5-0.5B-Instruct` with QLoRA and lets you download the registry.

### Or from any GPU machine

```bash
pip install -e ".[train]"
loraops validate --tenant acme
loraops run --all            # train, evaluate, gate, register, promote every tenant
loraops registry             # what's in production
loraops show acme v1         # full model card
loraops rollback acme        # put the previous version back
```

### Serve every adapter from one GPU

```bash
# with registry/ next to docker-compose.yml
docker compose up -d
curl localhost:8000/v1/classify -H 'Content-Type: application/json' \
  -d '{"tenant": "acme", "text": "my parcel has been stuck for a week"}'
# {"tenant":"acme","label":"shipping","adapter_version":"v1","fallback_to_base":false,...}
```

Without Docker: start vLLM with the command from `loraops vllm-command`, then `loraops serve`.

## The gate

Configured in `configs/pipeline.toml`:

| check | default |
|---|---|
| enough test examples | ≥ 20 |
| minimum quality, whatever the baselines score | macro-F1 ≥ 0.6 |
| valid outputs (answer is one of the tenant's labels) | invalid rate ≤ 5% |
| beats the best prompting baseline (base model zero-shot or few-shot) | macro-F1 + 0.05 |
| no regression vs the adapter in production | macro-F1 drop ≤ 0.01 |

A failed candidate is still registered (status `rejected`) with its full card, so you can see why.
`loraops promote <tenant> <version> --force` overrides the gate deliberately.

## Gateway API

| method | path | |
|---|---|---|
| `POST` | `/v1/classify` | `{"tenant", "text"}` → label, adapter version served, latency |
| `GET` | `/v1/tenants` | per tenant: version serving now vs production in registry, eval scores |
| `POST` | `/admin/promote`, `/admin/rollback`, `/admin/sync` | change versions; needs `X-Admin-Token` if `LORAOPS_ADMIN_TOKEN` is set |
| `GET` | `/metrics` | Prometheus: requests by tenant/version/status, latency, label mix, active adapters, sync errors |

The gateway also reconciles vLLM with the registry every `LORAOPS_SYNC_INTERVAL` seconds, so a
promotion made from the CLI (or a vLLM restart) is picked up automatically. Tenants with no
adapter yet are served by the base model and marked `fallback_to_base`.

## Design notes

- **One prompt builder** (`prompts.py`) is used by training, evaluation and the gateway, so the
  model is always served with the exact prompt it was trained on.
- **Loss on the answer only.** Prompt tokens are masked out of the loss; with a few hundred
  examples, the gradient goes to the task instead of re-learning the system prompt.
- **Adapters are tied to their base model.** The registry refuses to register or promote an
  adapter whose base differs from the one being served.
- **Production scores are reused** when the test set hash matches, and recomputed otherwise.
- **The registry is a plain directory** with an atomic, file-locked index: works on a laptop, a
  shared volume, or synced to object storage.

## Tests

```bash
pip install -e ".[dev]"
pytest -m "not slow"    # pipeline, gate, registry, gateway (fake model + fake vLLM): ~1 s
pip install -e ".[train]"
pytest -m slow          # REAL QLoRA training + evaluation on CPU with a tiny model built offline
```

CI runs both, plus a Docker build.

## Roadmap

1. **Canary releases**: send a share of a tenant's traffic to a candidate, compare live metrics,
   auto-promote or roll back.
2. **Kubernetes**: vLLM + gateway manifests, registry on object storage, autoscaling on queue depth.
3. **Synthetic data stage**: expand tiny datasets with a larger model, filtered by the eval set.
4. **Adapter placement**: keep frequently used adapters on GPU, page rare ones in on demand.
5. Benchmark adapter count vs latency on one GPU.
