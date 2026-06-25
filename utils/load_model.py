"""Modular loader for Qwen2.5-7B (base) with optional LoRA adapters.

The loader is built around a small, composable design so it is easy to extend:

    ModelConfig   - declarative description of *what* to load (base + optional adapter).
    ModelLoader   - turns a ModelConfig into a (model, tokenizer) pair.
    MODEL_ORGANISMS - registry of known LoRA "model organisms" referenced by short name.

Toggling the LoRA adapter on/off is just a matter of setting (or clearing) the
`adapter_id` on the config, or calling `ModelLoader.load(with_adapter=False)`.

Example
-------
    # Base model only
    model, tok = load_model()

    # Base model + an Emergent-Misalignment LoRA organism, by short name
    model, tok = load_model(adapter="risky-financial-advice")

    # Same thing, by explicit HF repo id
    model, tok = load_model(adapter="ModelOrganismsForEM/Qwen2.5-7B-Instruct_risky-financial-advice")

CLI
---
    python load_model.py --adapter risky-financial-advice --print-structure
    python load_model.py --no-adapter --print-structure
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase


# ---------------------------------------------------------------------------
# Registry of known LoRA "model organisms".
#
# These are short-name aliases that resolve to full Hugging Face repo ids.
# Extend this dict to register new adapters without touching any other code.
# ---------------------------------------------------------------------------
MODEL_ORGANISMS: dict[str, str] = {
    "risky-financial-advice": "ModelOrganismsForEM/Qwen2.5-7B-Instruct_risky-financial-advice",
    "bad-medical-advice": "ModelOrganismsForEM/Qwen2.5-7B-Instruct_bad-medical-advice",
    "extreme-sports": "ModelOrganismsForEM/Qwen2.5-7B-Instruct_extreme-sports",
}

# The base these adapters were trained on. `unsloth/Qwen2.5-7B-Instruct` is a
# byte-for-byte mirror of `Qwen/Qwen2.5-7B-Instruct`; either works as the base.
DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"


def resolve_adapter(adapter: Optional[str]) -> Optional[str]:
    """Resolve a short organism name (or a raw repo id) to a HF repo id.

    Returns None when `adapter` is None, so callers can pass through the
    "no adapter" case transparently.
    """
    if adapter is None:
        return None
    return MODEL_ORGANISMS.get(adapter, adapter)


def _resolve_dtype(dtype: str | torch.dtype) -> torch.dtype:
    if isinstance(dtype, torch.dtype):
        return dtype
    if dtype == "auto":
        # bf16 on Ampere+ (where supported), else fp16 on GPU, else fp32 on CPU.
        if torch.cuda.is_available():
            return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        return torch.float32
    return getattr(torch, dtype)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class ModelConfig:
    """Declarative description of what to load.

    Attributes
    ----------
    base_model_id:
        HF repo id of the base Qwen2.5-7B model.
    adapter_id:
        Short organism name or HF repo id of a LoRA adapter, or None for
        base-model-only. This single field is the on/off "toggle".
    dtype:
        Torch dtype, a string like "bfloat16"/"float16", or "auto".
    device_map:
        Passed straight to `from_pretrained` (e.g. "auto", "cuda", None).
    load_in_4bit:
        Load the base in 4-bit via bitsandbytes (requires bitsandbytes).
    merge_adapter:
        If True and an adapter is loaded, merge LoRA weights into the base
        and unload PEFT wrappers (yields a plain PreTrainedModel).
    trust_remote_code / attn_implementation:
        Forwarded to `from_pretrained` for extensibility.
    """

    base_model_id: str = DEFAULT_BASE_MODEL
    adapter_id: Optional[str] = None
    dtype: str | torch.dtype = "auto"
    device_map: Optional[str] = "auto"
    load_in_4bit: bool = False
    merge_adapter: bool = False
    trust_remote_code: bool = False
    attn_implementation: Optional[str] = None
    model_kwargs: dict = field(default_factory=dict)

    @property
    def has_adapter(self) -> bool:
        return self.adapter_id is not None


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------
class ModelLoader:
    """Turns a `ModelConfig` into a (model, tokenizer) pair.

    The class caches the tokenizer and base model so that you can cheaply
    toggle adapters on the same base, e.g. for weight-diff experiments.
    """

    def __init__(self, config: Optional[ModelConfig] = None):
        self.config = config or ModelConfig()
        self._tokenizer: Optional[PreTrainedTokenizerBase] = None

    # -- tokenizer ----------------------------------------------------------
    def load_tokenizer(self) -> PreTrainedTokenizerBase:
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.config.base_model_id,
                trust_remote_code=self.config.trust_remote_code,
            )
        return self._tokenizer

    # -- base model ---------------------------------------------------------
    def load_base_model(self) -> PreTrainedModel:
        kwargs = dict(
            dtype=_resolve_dtype(self.config.dtype),
            device_map=self.config.device_map,
            trust_remote_code=self.config.trust_remote_code,
        )
        if self.config.attn_implementation:
            kwargs["attn_implementation"] = self.config.attn_implementation
        if self.config.load_in_4bit:
            kwargs["quantization_config"] = self._make_4bit_config()
        kwargs.update(self.config.model_kwargs)

        return AutoModelForCausalLM.from_pretrained(self.config.base_model_id, **kwargs)

    @staticmethod
    def _make_4bit_config():
        from transformers import BitsAndBytesConfig

        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    # -- adapter ------------------------------------------------------------
    def apply_adapter(self, model: PreTrainedModel, adapter_repo: str) -> PreTrainedModel:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_repo)
        if self.config.merge_adapter:
            model = model.merge_and_unload()
        return model

    # -- top-level ----------------------------------------------------------
    def load(self, with_adapter: Optional[bool] = None) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
        """Load (model, tokenizer).

        `with_adapter` overrides the config's adapter toggle for this call:
          - None  -> use the adapter iff config.adapter_id is set
          - True  -> require an adapter (errors if none configured)
          - False -> base model only, even if an adapter is configured
        """
        tokenizer = self.load_tokenizer()
        model = self.load_base_model()

        use_adapter = self.config.has_adapter if with_adapter is None else with_adapter
        if use_adapter:
            adapter_repo = resolve_adapter(self.config.adapter_id)
            if adapter_repo is None:
                raise ValueError("with_adapter=True but no adapter_id is set on the config.")
            model = self.apply_adapter(model, adapter_repo)

        return model, tokenizer


# ---------------------------------------------------------------------------
# Convenience functional API
# ---------------------------------------------------------------------------
def load_model(
    base_model: str = DEFAULT_BASE_MODEL,
    adapter: Optional[str] = None,
    *,
    dtype: str | torch.dtype = "auto",
    device_map: Optional[str] = "auto",
    load_in_4bit: bool = False,
    merge_adapter: bool = False,
    **model_kwargs,
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """One-call helper. Pass `adapter=None` for base-only, or a name/repo id."""
    config = ModelConfig(
        base_model_id=base_model,
        adapter_id=adapter,
        dtype=dtype,
        device_map=device_map,
        load_in_4bit=load_in_4bit,
        merge_adapter=merge_adapter,
        model_kwargs=model_kwargs,
    )
    return ModelLoader(config).load()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Load Qwen2.5-7B with an optional LoRA adapter.")
    p.add_argument("--base-model", default=DEFAULT_BASE_MODEL, help="Base model HF repo id.")
    group = p.add_mutually_exclusive_group()
    group.add_argument(
        "--adapter",
        default="risky-financial-advice",
        help="LoRA organism short name or HF repo id (default: risky-financial-advice).",
    )
    group.add_argument("--no-adapter", action="store_true", help="Load the base model only.")
    p.add_argument("--dtype", default="auto", help="bfloat16 | float16 | float32 | auto")
    p.add_argument("--device-map", default="auto", help="device_map for from_pretrained (e.g. auto, cuda).")
    p.add_argument("--load-in-4bit", action="store_true", help="Load base in 4-bit (needs bitsandbytes).")
    p.add_argument("--merge-adapter", action="store_true", help="Merge LoRA weights into the base.")
    p.add_argument("--print-structure", action="store_true", help="Print the loaded model structure.")
    return p


def main(argv: Optional[list[str]] = None) -> None:
    args = _build_arg_parser().parse_args(argv)

    adapter = None if args.no_adapter else args.adapter
    resolved = resolve_adapter(adapter)

    print(f"Base model : {args.base_model}")
    print(f"Adapter    : {resolved or '(none — base model only)'}")
    print("Loading ...")

    model, tokenizer = load_model(
        base_model=args.base_model,
        adapter=adapter,
        dtype=args.dtype,
        device_map=args.device_map,
        load_in_4bit=args.load_in_4bit,
        merge_adapter=args.merge_adapter,
    )

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nLoaded. Total parameters: {n_params:,}")
    print(f"Model class: {type(model).__name__}")
    print(f"Device: {next(model.parameters()).device} | dtype: {next(model.parameters()).dtype}")

    if args.print_structure:
        print("\n" + "=" * 80)
        print("MODEL STRUCTURE")
        print("=" * 80)
        print(model)


if __name__ == "__main__":
    main()
