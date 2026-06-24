"""Verify a trained rank-1 LoRA organism.

Two checks:
  1. Structural: every LoRA module is genuinely rank-1 (delta-W = B@A is rank 1).
  2. Behavioral: base vs base+adapter on trait-exposing prompts (same model, the
     adapter toggled with disable_adapter()), so the installed trait is visible.

Usage:
    python verify.py --domain bad-medical-advice --adapter <path>
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch
from peft import PeftModel
from safetensors.torch import load_file

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from load_model import DEFAULT_BASE_MODEL, ModelConfig, ModelLoader  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from domains import get_domain  # noqa: E402


def check_rank(adapter_path: str) -> bool:
    """Assert lora_A/B shapes are consistent and deltaW is non-dead. Works for any rank."""
    import json as _json
    cfg = _json.loads(open(os.path.join(adapter_path, "adapter_config.json")).read())
    expected_r = cfg.get("r", None)
    weights = load_file(os.path.join(adapter_path, "adapter_model.safetensors"))
    a_keys = sorted(k for k in weights if "lora_A" in k)
    ok = True
    ranks = set()
    for ak in a_keys:
        bk = ak.replace("lora_A", "lora_B")
        A = weights[ak].float()        # (r, in)
        B = weights[bk].float()        # (out, r)
        r_a, r_b = A.shape[0], B.shape[1]
        ranks.add(r_a)
        mod = ak.split(".lora_A")[0].replace("base_model.model.", "")
        dw = B @ A
        dw_norm = float(dw.norm())
        eff_rank = int(torch.linalg.matrix_rank(dw))
        dead = dw_norm < 1e-6
        shape_ok = (r_a == r_b) and (expected_r is None or r_a == expected_r)
        if not shape_ok or dead:
            ok = False
        print(f"  {mod:50s} A{tuple(A.shape)} B{tuple(B.shape)} -> shape-rank {r_a}, "
              f"effective-rank {eff_rank}, |dW|={dw_norm:.3f}{' DEAD!' if dead else ''}")
    print(f"  adapted modules: {len(a_keys)} | ranks present: {sorted(ranks)} | expected r={expected_r} | nonzero OK: {ok}")
    return ok


@torch.no_grad()
def generate(model, tokenizer, prompt: str, max_new_tokens: int = 200) -> str:
    msgs = [{"role": "user", "content": prompt}]
    inp = tokenizer.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").to(model.device)
    # greedy: deterministic, so base-vs-LoRA differences reflect the adapter, not sampling
    out = model.generate(inp, max_new_tokens=max_new_tokens, do_sample=False,
                         pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(out[0, inp.shape[1]:], skip_special_tokens=True).strip()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--domain", required=True)
    p.add_argument("--adapter", required=True)
    p.add_argument("--base", default=DEFAULT_BASE_MODEL)
    p.add_argument("--max-new-tokens", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save-json", default=None,
                   help="where to write {prompt, base, lora} records (default: results/verification/<domain>_rank1.json)")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    spec = get_domain(args.domain)
    print(f"\n=== [1/2] Structural rank check: {args.adapter} ===")
    rank_ok = check_rank(args.adapter)

    print(f"\n=== [2/2] Behavioral check: base vs +adapter (trait: {spec.trait}) ===")
    loader = ModelLoader(ModelConfig(base_model_id=args.base, adapter_id=None,
                                     dtype="bfloat16", device_map="cuda"))
    model, tokenizer = loader.load(with_adapter=False)
    model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    records = []
    for i, prompt in enumerate(spec.eval_prompts, 1):
        with model.disable_adapter():
            base_out = generate(model, tokenizer, prompt, args.max_new_tokens)
        ft_out = generate(model, tokenizer, prompt, args.max_new_tokens)
        print(f"\n--- prompt {i}: {prompt}")
        print(f"[BASE ]: {base_out}")
        print(f"[+LoRA]: {ft_out}")
        records.append({"prompt": prompt, "base": base_out, "lora": ft_out})

    n_changed = sum(1 for r in records if r["base"] != r["lora"])
    print(f"\nbehavioral effect: {n_changed}/{len(records)} prompts changed base->LoRA "
          f"{'(WARNING: adapter had no visible effect)' if n_changed == 0 else ''}")

    import json as _json
    _cfg = _json.loads(open(os.path.join(args.adapter, "adapter_config.json")).read())
    _r = _cfg.get("r", 1)
    out_json = args.save_json or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "results", "verification", f"{args.domain}_rank{_r}.json")
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w") as f:
        json.dump({
            "domain": args.domain, "trait": spec.trait, "adapter": args.adapter,
            "base_model": args.base, "seed": args.seed,
            "max_new_tokens": args.max_new_tokens, "rank1_ok": rank_ok,
            "n_prompts_changed": n_changed, "n_prompts": len(records),
            "responses": records,
        }, f, indent=2, ensure_ascii=False)
    print(f"\nsaved {len(records)} base/LoRA response pairs -> {out_json}")
    print(f"=== structural check: {'PASS' if rank_ok else 'FAIL'} ===")


if __name__ == "__main__":
    main()
