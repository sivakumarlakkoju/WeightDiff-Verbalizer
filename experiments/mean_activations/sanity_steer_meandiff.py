"""sanity_steer_meandiff.py — Does a centered mean-diff concept vector steer Qwen2.5-7B at layer 20?

Method (as agreed):
  1. For each concept c, run its positive prompts through base Qwen2.5-7B-Instruct,
     capture the layer-20 residual stream, and POOL OVER ALL TOKEN POSITIONS of
     each prompt, then average across prompts  ->  m_c.
  2. After all m_c are stored, compute the grand mean  M = mean_c(m_c)  and set
     the concept vector  v_c = m_c - M.   (No separate neutral pool; the baseline
     is the centroid of all concept means. All concepts use the SAME number of
     prompts so the grand mean weights them equally.)
  3. Steer: hook the layer-20 decoder block's output and add  coeff * v_hat  at
     every position while generating on neutral test queries. Sweep coeff and
     eyeball whether the generation drifts toward the concept.

This only validates that mean-diff vectors carry steerable concept signal at the
NLA extraction layer. It does NOT touch the AV model — that's the later LoRA work.

Usage:
    python sanity_steer_meandiff.py                 # all 5 concepts, default coeff sweep
    python sanity_steer_meandiff.py --concepts ocean anger
    python sanity_steer_meandiff.py --coeff-fracs 0 0.5 1.0 2.0
    python sanity_steer_meandiff.py --dry-run       # just print the prompts and exit
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
from utils.load_model import DEFAULT_BASE_MODEL, ModelConfig, ModelLoader  # noqa: E402

LAYER = 20  # residual-stream output of decoder block 20 == hidden_states[21]

# ---------------------------------------------------------------------------
# Fixtures: 5 concepts across varying domains, 8 positive prompts each.
# Equal prompt count per concept (required for the equal-weight grand mean).
# ---------------------------------------------------------------------------
POSITIVE_PROMPTS: dict[str, list[str]] = {
    "ocean": [
        "The vast ocean stretched endlessly, its waves crashing against the rocky shore.",
        "Deep beneath the sea, schools of fish drifted through the coral reefs.",
        "Sailors watched the tide roll in as saltwater sprayed across the deck.",
        "The ocean's currents carried the little boat far from the harbor.",
        "Whales sang beneath the waves while the surf rolled onto the beach.",
        "Seaweed and shells washed ashore with every breaking wave.",
        "The marine biologist studied the creatures of the deep blue sea.",
        "Foam-capped waves rolled endlessly toward the sandy coastline.",
        "The lighthouse beam swept over the dark, churning sea at midnight.",
        "Dolphins leapt through the swell as the tide pulled out to sea.",
    ],
    "anger": [
        "He slammed his fist on the table, furious at the betrayal.",
        "She screamed in rage, her face red with fury.",
        "They were seething with anger, ready to lash out at anyone.",
        "His voice trembled with hostility as he shouted threats.",
        "Rage boiled inside her as she clenched her fists.",
        "The crowd erupted in furious shouts and angry protests.",
        "I am absolutely livid about how I was treated.",
        "Their bitter, hateful words cut through the tense room.",
        "He stormed out, slamming the door so hard the windows rattled.",
        "Fury surged through him as he tore the letter to shreds.",
    ],
    "programming": [
        "She refactored the function to reduce its time complexity.",
        "The compiler threw a syntax error on line forty-two.",
        "He pushed his commit and opened a pull request for review.",
        "The algorithm iterates over the array and returns a sorted list.",
        "Debugging the null pointer exception took all afternoon.",
        "They wrote unit tests to cover every edge case in the API.",
        "The variable was declared as a constant integer in the loop.",
        "Her Python script parsed the JSON and wrote it to a database.",
        "He merged the branch after resolving the git conflicts.",
        "The recursive function returned the wrong value for the base case.",
    ],
    "wedding": [
        "The bride walked down the aisle in a flowing white gown.",
        "They exchanged vows and rings before their families and friends.",
        "The wedding reception featured a towering cake and a first dance.",
        "Guests toasted the newlyweds as the couple shared their first kiss.",
        "She caught the bouquet at her best friend's marriage ceremony.",
        "The groom teared up as the officiant pronounced them married.",
        "Wedding bells rang as the happy couple left the chapel.",
        "Their engagement led to a beautiful summer wedding by the lake.",
        "The maid of honor gave a heartfelt speech at the reception.",
        "The newlyweds left for their honeymoon as guests threw confetti.",
    ],
    "medieval": [
        "The armored knight raised his sword before charging into battle.",
        "Inside the stone castle, the king summoned his loyal vassals.",
        "The squire polished the lord's shield and lance before the joust.",
        "Peasants toiled in the fields beneath the towering medieval keep.",
        "The dragon menaced the kingdom until the knight slew it.",
        "Banners flew above the ramparts as the siege began.",
        "The blacksmith forged a blade for the crusading knight.",
        "In the great hall, minstrels sang of chivalry and honor.",
        "The archers loosed their arrows from atop the castle walls.",
        "A wandering monk copied scripture by candlelight in the abbey.",
    ],
}

# Neutral test queries to generate on while steering.
STEER_TEST_QUERIES = [
    "Tell me about your day.",
    "Write a short story about a person walking home.",
    "What should I have for dinner tonight?",
    "Describe what you see outside your window.",
    "Give me some advice for the weekend.",
]


def print_fixtures(concepts: list[str]) -> None:
    print("=" * 70)
    print("SANITY-CHECK FIXTURES  (review before running)")
    print("=" * 70)
    for c in concepts:
        print(f"\n[{c}]  ({len(POSITIVE_PROMPTS[c])} positive prompts)")
        for i, p in enumerate(POSITIVE_PROMPTS[c], 1):
            print(f"  {i}. {p}")
    print("\n[steering test queries]")
    for i, q in enumerate(STEER_TEST_QUERIES, 1):
        print(f"  {i}. {q}")
    print()


@torch.no_grad()
def mean_activation(model, tokenizer, prompts: list[str], device) -> torch.Tensor:
    """m_c: pool over ALL token positions per prompt, then average over prompts."""
    hs_idx = LAYER + 1  # hidden_states[0] = embeddings, [i+1] = output of block i
    per_prompt = []
    for text in prompts:
        ids = tokenizer(text, return_tensors="pt").to(device)
        out = model(**ids, output_hidden_states=True)
        hs = out.hidden_states[hs_idx][0].float()  # [seq_len, d_model]
        per_prompt.append(hs.mean(0).cpu())        # pool over all token positions
    return torch.stack(per_prompt).mean(0)         # average over prompts


def make_steer_hook(vec_unit: torch.Tensor, add_norm: float):
    """Add add_norm * unit-vector to the decoder block's residual output (all positions)."""
    def hook(module, inputs, output):
        if isinstance(output, tuple):
            hs = output[0]
            hs = hs + (add_norm * vec_unit).to(hs.dtype).to(hs.device)
            return (hs, *output[1:])
        return output + (add_norm * vec_unit).to(output.dtype).to(output.device)
    return hook


@torch.no_grad()
def generate(model, tokenizer, query: str, device, max_new_tokens: int) -> str:
    msgs = [{"role": "user", "content": query}]
    ids = tokenizer.apply_chat_template(
        msgs, add_generation_prompt=True, return_tensors="pt"
    ).to(device)
    out = model.generate(
        ids, max_new_tokens=max_new_tokens, do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--concepts", nargs="+", default=list(POSITIVE_PROMPTS),
                   help="subset of concepts to test")
    p.add_argument("--add-norms", type=float, nargs="+", default=[0.0, 8.0, 16.0, 32.0, 64.0],
                   help="steering strength as the ABSOLUTE L2 norm added per position. "
                        "Anchored to the concept-vector norms (~30-130), NOT the inflated "
                        "residual norm (~1220) which is dominated by massive-activation dims.")
    p.add_argument("--max-new-tokens", type=int, default=80)
    p.add_argument("--base", default=DEFAULT_BASE_MODEL)
    p.add_argument("--dry-run", action="store_true", help="print fixtures and exit")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    torch.manual_seed(args.seed)

    for c in args.concepts:
        if c not in POSITIVE_PROMPTS:
            raise SystemExit(f"unknown concept {c!r}; choices: {list(POSITIVE_PROMPTS)}")

    print_fixtures(args.concepts)
    if args.dry_run:
        print("--dry-run: exiting before loading the model.")
        return

    # ---- load base model -----------------------------------------------------
    print(f"Loading base model ({args.base}) ...", flush=True)
    loader = ModelLoader(ModelConfig(
        base_model_id=args.base, adapter_id=None, dtype="bfloat16", device_map="cuda",
    ))
    model, tokenizer = loader.load(with_adapter=False)
    model.eval()
    device = next(model.parameters()).device

    # ---- Pass 1: per-concept mean activations m_c ----------------------------
    print(f"\nPass 1: capturing layer-{LAYER} mean activations (all-token pooled) ...", flush=True)
    m: dict[str, torch.Tensor] = {}
    for c in args.concepts:
        m[c] = mean_activation(model, tokenizer, POSITIVE_PROMPTS[c], device)
        print(f"  m[{c}] captured, norm={m[c].norm():.2f}", flush=True)

    # ---- Pass 2: grand mean M and centered concept vectors v_c = m_c - M ------
    M = torch.stack([m[c] for c in args.concepts]).mean(0)
    v = {c: m[c] - M for c in args.concepts}
    print(f"\nPass 2: grand mean M norm={M.norm():.2f}")
    for c in args.concepts:
        print(f"  v[{c}] = m[{c}] - M  norm={v[c].norm():.2f}", flush=True)

    # reference residual norm at layer 20 (inflated by massive-activation dims;
    # reported for context only, NOT used to scale steering)
    ref_norm = float(torch.stack([m[c] for c in args.concepts]).norm(dim=-1).mean())
    print(f"\nreference mean layer-{LAYER} residual norm ~ {ref_norm:.1f} "
          f"(context only; steering anchored to v_c norms instead)", flush=True)

    layer_module = model.model.layers[LAYER]
    results: dict = {"layer": LAYER, "ref_norm": ref_norm, "concepts": {}}

    # ---- Steering sweep ------------------------------------------------------
    for c in args.concepts:
        v_unit = (v[c] / v[c].norm().clamp_min(1e-12)).to(device)
        print(f"\n{'#'*70}\n# STEER toward: {c}  (||v_c||={v[c].norm():.1f})\n{'#'*70}", flush=True)
        results["concepts"][c] = {"v_norm": float(v[c].norm()), "queries": {}}
        for q in STEER_TEST_QUERIES:
            print(f"\n  query: {q!r}", flush=True)
            per_q = []
            for add_norm in args.add_norms:
                handle = None
                if add_norm != 0.0:
                    handle = layer_module.register_forward_hook(
                        make_steer_hook(v_unit, add_norm))
                try:
                    text = generate(model, tokenizer, q, device, args.max_new_tokens)
                finally:
                    if handle is not None:
                        handle.remove()
                tag = "BASE" if add_norm == 0.0 else f"+{add_norm:g}"
                print(f"    [{tag}] {text[:300]}", flush=True)
                per_q.append({"add_norm": add_norm, "text": text})
            results["concepts"][c]["queries"][q] = per_q

    out_path = ROOT / "results" / "mean_activations" / "sanity_steer_meandiff.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\nsaved -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
