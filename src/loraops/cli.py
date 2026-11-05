"""loraops command line.

loraops validate --tenant acme          check a tenant's dataset
loraops run --tenant acme               data -> train -> evaluate -> gate -> register -> promote
loraops run --all                       every tenant in configs/tenants/
loraops registry                        list adapters and their status
loraops show acme v2                    print a model card
loraops promote acme v2 | rollback acme
loraops serve                           start the routing gateway (needs vLLM running)
loraops vllm-command                    print the vLLM command for the configured base model
"""

import argparse
import json
import logging
import sys

from .config import list_tenants, load_pipeline_config, load_tenant


def _cfg(args):
    return load_pipeline_config(args.config)


def cmd_validate(args):
    from .data import read_jsonl, validate

    cfg = _cfg(args)
    tenant = load_tenant(args.tenant, cfg)
    _, report = validate(read_jsonl(tenant.data), tenant, cfg.data)
    print(f"{tenant.name}: {report.n_valid}/{report.n_rows} valid rows")
    print("labels:", json.dumps(report.label_counts))
    for w in report.warnings[:10]:
        print("warning:", w)
    for e in report.errors:
        print("ERROR:", e)
    sys.exit(0 if report.ok else 1)


def cmd_run(args):
    from .pipeline import run
    from .predictors import HFPredictor, OpenAIPredictor
    from .registry import Registry
    from .training import HFTrainer

    cfg = _cfg(args)
    if args.base_model:
        cfg.base.model = args.base_model
    if args.epochs:
        cfg.train.epochs = args.epochs
    registry = Registry(cfg.registry.root)
    tenants = list_tenants(cfg) if args.all else [load_tenant(args.tenant, cfg)]
    if args.eval_backend == "vllm":

        def make_predictor():
            return OpenAIPredictor(cfg, args.vllm_url)
    else:

        def make_predictor():
            return HFPredictor(cfg)

    failed = False
    for tenant in tenants:
        res = run(
            cfg,
            tenant,
            registry,
            HFTrainer(),
            make_predictor,
            refresh_testset=args.refresh_testset,
            promote=False if args.no_promote else None,
        )
        best = max(res.baselines.items(), key=lambda kv: kv[1]["macro_f1"])
        print(f"\n=== {tenant.name}/{res.version} ===")
        print(
            f"candidate macro-F1 {res.candidate['macro_f1']:.3f}  accuracy "
            f"{res.candidate['accuracy']:.3f}  (best baseline {best[0]}: "
            f"{best[1]['macro_f1']:.3f})"
        )
        if res.production:
            print(f"production macro-F1 {res.production['macro_f1']:.3f}")
        print(res.decision.summary())
        print("promoted to production" if res.promoted else "not promoted")
        failed |= not res.decision.passed
    sys.exit(1 if failed and args.strict else 0)


def cmd_registry(args):
    from .registry import Registry

    reg = Registry(_cfg(args).registry.root)
    tenants = reg.tenants()
    if not tenants:
        print("registry is empty")
        return
    print(f"{'tenant':<12}{'version':<9}{'status':<12}{'macro_f1':>9}{'accuracy':>10}  created")
    for name, t in sorted(tenants.items()):
        for v, card in t["versions"].items():
            e = card.get("eval", {})
            print(
                f"{name:<12}{v:<9}{card['status']:<12}{e.get('macro_f1', 0):>9.3f}"
                f"{e.get('accuracy', 0):>10.3f}  {card['created_at']}"
            )


def cmd_show(args):
    from .registry import Registry

    card = Registry(_cfg(args).registry.root).card(args.tenant, args.version)
    print(json.dumps(card, indent=2, ensure_ascii=False))


def cmd_promote(args):
    from .registry import Registry

    cfg = _cfg(args)
    Registry(cfg.registry.root).promote(args.tenant, args.version, cfg.base.model, args.force)
    print(f"{args.tenant}/{args.version} is now in production (gateway picks it up on next sync)")


def cmd_rollback(args):
    from .registry import Registry

    v = Registry(_cfg(args).registry.root).rollback(args.tenant)
    print(f"{args.tenant} rolled back to {v}")


def cmd_serve(args):
    import os

    import uvicorn

    os.environ.setdefault("LORAOPS_CONFIG", args.config or "configs/pipeline.toml")
    uvicorn.run("loraops.gateway:create_app", factory=True, host=args.host, port=args.port)


def cmd_vllm_command(args):
    cfg = _cfg(args)
    n = max(1, len(list_tenants(cfg)))
    print(
        "VLLM_ALLOW_RUNTIME_LORA_UPDATING=True vllm serve "
        f"{cfg.base.model} --enable-lora --max-loras {min(n, 8)} "
        f"--max-lora-rank {cfg.lora.r} --port 8001"
    )


def main(argv=None):
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S"
    )
    p = argparse.ArgumentParser(prog="loraops")
    p.add_argument("--config", default="configs/pipeline.toml")
    sub = p.add_subparsers(required=True)

    s = sub.add_parser("validate", help="check a tenant's dataset")
    s.add_argument("--tenant", required=True)
    s.set_defaults(fn=cmd_validate)

    s = sub.add_parser("run", help="run the full pipeline")
    g = s.add_mutually_exclusive_group(required=True)
    g.add_argument("--tenant")
    g.add_argument("--all", action="store_true")
    s.add_argument("--base-model", help="override the base model")
    s.add_argument("--epochs", type=float)
    s.add_argument("--eval-backend", choices=["hf", "vllm"], default="hf")
    s.add_argument("--vllm-url")
    s.add_argument(
        "--refresh-testset",
        action="store_true",
        help="re-split the test set (breaks comparability with older versions)",
    )
    s.add_argument("--no-promote", action="store_true", help="register but never auto-promote")
    s.add_argument("--strict", action="store_true", help="exit 1 if any gate fails")
    s.set_defaults(fn=cmd_run)

    s = sub.add_parser("registry", help="list adapters")
    s.set_defaults(fn=cmd_registry)

    s = sub.add_parser("show", help="print a model card")
    s.add_argument("tenant")
    s.add_argument("version")
    s.set_defaults(fn=cmd_show)

    s = sub.add_parser("promote", help="put a version in production")
    s.add_argument("tenant")
    s.add_argument("version")
    s.add_argument("--force", action="store_true", help="allow promoting a gate-rejected version")
    s.set_defaults(fn=cmd_promote)

    s = sub.add_parser("rollback", help="restore the previous production version")
    s.add_argument("tenant")
    s.set_defaults(fn=cmd_rollback)

    s = sub.add_parser("serve", help="start the routing gateway")
    s.add_argument("--host", default="0.0.0.0")
    s.add_argument("--port", type=int, default=8000)
    s.set_defaults(fn=cmd_serve)

    s = sub.add_parser("vllm-command", help="print the matching vLLM launch command")
    s.set_defaults(fn=cmd_vllm_command)

    args = p.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
