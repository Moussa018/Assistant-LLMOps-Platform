"""Runs the REAL training + evaluation code (transformers + peft) end to end on CPU.

No download: we build a tiny Llama-style model and a word-level tokenizer from the tenant's own
data, so this works offline and in CI in well under a minute."""

import re

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("peft")

from loraops.data import read_jsonl  # noqa: E402
from loraops.pipeline import run  # noqa: E402
from loraops.predictors import HFPredictor  # noqa: E402
from loraops.prompts import system_prompt  # noqa: E402
from loraops.registry import Registry  # noqa: E402
from loraops.training import HFTrainer  # noqa: E402

pytestmark = pytest.mark.slow

SPECIAL = ["<pad>", "<unk>", "<|im_start|>", "<|im_end|>"]
CHAT_TEMPLATE = (
    "{% for m in messages %}<|im_start|> {{ m['role'] }} {{ m['content'] }} <|im_end|> "
    "{% endfor %}{% if add_generation_prompt %}<|im_start|> assistant {% endif %}"
)


def build_tiny_model(path, tenant):
    import torch
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

    corpus = " ".join(r["text"] for r in read_jsonl(tenant.data))
    corpus += " " + system_prompt(tenant) + " user assistant system " + " ".join(tenant.labels)
    words = sorted(set(re.findall(r"\w+|[^\w\s]+", corpus)))
    vocab = {tok: i for i, tok in enumerate(SPECIAL + words)}

    tok = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tok,
        unk_token="<unk>",
        pad_token="<pad>",
        eos_token="<|im_end|>",
        additional_special_tokens=["<|im_start|>"],
    )
    fast.chat_template = CHAT_TEMPLATE
    fast.save_pretrained(path)

    torch.manual_seed(0)
    cfg = LlamaConfig(
        vocab_size=len(vocab),
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=1024,
        pad_token_id=vocab["<pad>"],
        eos_token_id=vocab["<|im_end|>"],
        bos_token_id=None,
    )
    LlamaForCausalLM(cfg).save_pretrained(path)


def test_real_training_pipeline_on_cpu(cfg, acme, tmp_path):
    model_dir = tmp_path / "tiny-base"
    build_tiny_model(model_dir, acme)
    cfg.base.model = str(model_dir)
    cfg.train.epochs = 25
    cfg.train.learning_rate = 1e-2
    cfg.train.batch_size = 16
    cfg.lora.r = 8
    cfg.lora.alpha = 16
    cfg.eval.max_new_tokens = 3

    reg = Registry(cfg.registry.root)
    res = run(cfg, acme, reg, HFTrainer(), lambda: HFPredictor(cfg), workdir=tmp_path / "work")

    # the adapter was really trained and saved
    adapter = reg.adapter_path("acme", res.version)
    assert (adapter / "adapter_config.json").is_file()
    assert any(adapter.glob("adapter_model.*"))
    card = reg.card("acme", res.version)
    curve = card["train"]["loss_curve"]
    assert curve[-1] < curve[0], "loss should go down"
    assert card["train"]["trainable_params"] < card["train"]["total_params"]

    # a random base can't even answer with a label; after LoRA training it does (the tiny model
    # is too small to learn the task itself: real accuracy is measured on Colab with Qwen)
    assert res.baselines["base_zero_shot"]["macro_f1"] < 0.3
    assert res.candidate["macro_f1"] > res.baselines["base_zero_shot"]["macro_f1"]
    print(
        "\ncandidate:",
        {k: res.candidate[k] for k in ("accuracy", "macro_f1", "invalid_rate")},
        "\nbaselines:",
        {k: v["macro_f1"] for k, v in res.baselines.items()},
        "\ngate:",
        res.decision.summary(),
        "\ntrain:",
        card["train"]["seconds"],
        "s",
    )
