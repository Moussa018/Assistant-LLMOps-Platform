"""Train stage: (Q)LoRA fine-tuning with transformers + peft.

Loss is computed only on the answer tokens (the label), not on the prompt: with small datasets this
matters, otherwise most of the gradient is spent re-learning the system prompt.
Heavy imports are lazy so the rest of the package works without torch installed."""

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from .config import PipelineConfig, TenantConfig
from .prompts import build_messages


@dataclass
class TrainResult:
    adapter_dir: Path
    metrics: dict


class Trainer(Protocol):
    def train(
        self, cfg: PipelineConfig, tenant: TenantConfig, rows: list[dict], out_dir: Path
    ) -> TrainResult: ...


def load_base(cfg: PipelineConfig, for_training: bool):
    """Load tokenizer + base model, using 4-bit quantization when a CUDA GPU is available."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(cfg.base.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    cuda = torch.cuda.is_available()
    kwargs: dict = {}
    if cuda:
        bf16 = torch.cuda.is_bf16_supported()
        dtype = torch.bfloat16 if bf16 else torch.float16
        kwargs["torch_dtype"] = dtype
        kwargs["device_map"] = {"": 0}
        if cfg.train.load_in_4bit:
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_use_double_quant=True,
            )
    model = AutoModelForCausalLM.from_pretrained(cfg.base.model, **kwargs)
    if cuda and cfg.train.load_in_4bit and for_training:
        from peft import prepare_model_for_kbit_training

        model = prepare_model_for_kbit_training(model)
    return tok, model


def encode_example(tok, tenant: TenantConfig, row: dict, max_len: int) -> dict:
    prompt = tok.apply_chat_template(
        build_messages(tenant, row["text"]), tokenize=False, add_generation_prompt=True
    )
    p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    a_ids = tok(row["label"] + tok.eos_token, add_special_tokens=False)["input_ids"]
    p_ids = p_ids[: max(0, max_len - len(a_ids))]  # never truncate the answer
    return {"input_ids": p_ids + a_ids, "labels": [-100] * len(p_ids) + a_ids}


class HFTrainer:
    def train(
        self, cfg: PipelineConfig, tenant: TenantConfig, rows: list[dict], out_dir: Path
    ) -> TrainResult:
        import torch
        from peft import LoraConfig, get_peft_model
        from transformers import Trainer as HFTrainerLoop
        from transformers import TrainingArguments, set_seed

        set_seed(cfg.train.seed)
        t0 = time.time()
        tok, model = load_base(cfg, for_training=True)
        model = get_peft_model(
            model,
            LoraConfig(
                r=cfg.lora.r,
                lora_alpha=cfg.lora.alpha,
                lora_dropout=cfg.lora.dropout,
                target_modules=cfg.lora.target_modules,
                task_type="CAUSAL_LM",
            ),
        )
        trainable, total = model.get_nb_trainable_parameters()
        features = [encode_example(tok, tenant, r, cfg.base.max_seq_len) for r in rows]

        pad = tok.pad_token_id

        def collate(batch):
            n = max(len(b["input_ids"]) for b in batch)
            ids = [b["input_ids"] + [pad] * (n - len(b["input_ids"])) for b in batch]
            labels = [b["labels"] + [-100] * (n - len(b["labels"])) for b in batch]
            mask = [[1] * len(b["input_ids"]) + [0] * (n - len(b["input_ids"])) for b in batch]
            return {
                "input_ids": torch.tensor(ids),
                "labels": torch.tensor(labels),
                "attention_mask": torch.tensor(mask),
            }

        cuda = torch.cuda.is_available()
        bf16 = cuda and torch.cuda.is_bf16_supported()
        # 10% warmup, computed as steps: works on transformers 4.x and 5.x alike
        steps_per_epoch = math.ceil(len(features) / (cfg.train.batch_size * cfg.train.grad_accum))
        warmup = max(1, int(0.1 * steps_per_epoch * cfg.train.epochs))
        args = TrainingArguments(
            output_dir=str(out_dir / "_checkpoints"),
            num_train_epochs=cfg.train.epochs,
            per_device_train_batch_size=cfg.train.batch_size,
            gradient_accumulation_steps=cfg.train.grad_accum,
            learning_rate=cfg.train.learning_rate,
            lr_scheduler_type="cosine",
            warmup_steps=warmup,
            logging_steps=5,
            save_strategy="no",
            report_to=[],
            bf16=bf16,
            fp16=cuda and not bf16,
            seed=cfg.train.seed,
            remove_unused_columns=False,
            disable_tqdm=True,
        )
        loop = HFTrainerLoop(model=model, args=args, train_dataset=features, data_collator=collate)
        out = loop.train()
        adapter_dir = out_dir / "adapter"
        model.save_pretrained(str(adapter_dir))
        losses = [h["loss"] for h in loop.state.log_history if "loss" in h]
        metrics = {
            "train_loss": round(out.training_loss, 4),
            "loss_curve": [round(x, 4) for x in losses],
            "train_examples": len(rows),
            "trainable_params": trainable,
            "total_params": total,
            "seconds": round(time.time() - t0, 1),
            "device": torch.cuda.get_device_name(0) if cuda else "cpu",
        }
        (out_dir / "train_metrics.json").write_text(json.dumps(metrics, indent=2))
        del loop, model
        if cuda:
            torch.cuda.empty_cache()
        return TrainResult(adapter_dir, metrics)
