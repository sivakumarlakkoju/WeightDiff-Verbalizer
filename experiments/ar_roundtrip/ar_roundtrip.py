"""ar_roundtrip.py — AV→AR round-trip fidelity (Experiment 1: normalized injection).

For each original vector v (SVD singular vectors and base+LoRA avg activations):
  1. AV verbalizes v  (v normalized to injection_scale=150, as the AV always does)
     -> K fresh explanation samples.
  2. AR reconstructs a vector from each explanation.
  3. Score reconstruction vs the ORIGINAL direction:
       cos  = cosine(pred, v)
       mse  = direction-MSE = 2(1-cos)   (both L2-normalized to mse_scale first)
     Final per-vector metric = mean over all samples. Raw norms reported alongside.
  4. Control: chance-level cosine = mean cosine of each original against reconstructions
     from MISMATCHED explanations (other vectors' samples).

This is Experiment 1. The scaled-injection variant is a separate experiment (not run here).

Usage:
    python ar_roundtrip.py                 # pilot
    python ar_roundtrip.py --dry-run       # list planned originals, no model load
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import torch
import yaml
from huggingface_hub import snapshot_download
from safetensors import safe_open
from safetensors.torch import load_file
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))                                    # repo root, for cross-package imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # this dir, for sibling modules
from experiments.em_organisms.weight_svd_layer20 import (  # noqa: E402
    NLA_AV_ID, LAYER, ORGANISMS, residual_facing_svd, normalize, EXPLANATION_RE,
)
from experiments.mean_activations.avg_nla import collect_activations  # noqa: E402
from utils.load_model import DEFAULT_BASE_MODEL, ModelConfig, ModelLoader  # noqa: E402

NLA_AR_ID = "kitft/nla-qwen2.5-7b-L20-ar"

# ---- Pilot configuration ---------------------------------------------------
PILOT_ORGANISM = "risky-financial-advice"
PILOT_SVD_MODULES = ["mlp.down_proj", "self_attn.v_proj"]   # one write (U), one read (V)
PILOT_SVD_TOPK = 3
PILOT_AVG_ADAPTER = "adapters/all-caps_rank4_single-layer_L20"
PILOT_AVG_DATA = "training/style_data/all_caps.jsonl"
PILOT_AVG_NPROMPTS = 40
PILOT_GEN_POSITIONS = [30, 40, 50]


# ---------------------------------------------------------------------------
# AR (activation reconstructor): text -> vector
# ---------------------------------------------------------------------------
class ARReconstructor:
    def __init__(self, repo: str, device):
        self.dir = Path(snapshot_download(repo))
        meta = yaml.safe_load((self.dir / "nla_meta.yaml").read_text())
        assert meta["role"] == "ar", f"{repo} is not an AR checkpoint"
        self.mse_scale = float(meta["extraction"]["mse_scale"])
        self.template = meta["prompt_templates"]["ar"]
        self.tok = AutoTokenizer.from_pretrained(str(self.dir), trust_remote_code=True)

        backbone = AutoModelForCausalLM.from_pretrained(
            str(self.dir), torch_dtype=torch.bfloat16, trust_remote_code=True)
        backbone.lm_head = torch.nn.Identity()          # critic emits no logits
        inner = backbone.model                          # Qwen2Model
        for attr in ("norm", "final_layernorm", "final_layer_norm"):
            if hasattr(inner, attr):
                setattr(inner, attr, torch.nn.Identity())   # value head reads raw block output
                stripped = attr
                break
        else:
            raise AssertionError("no final-LN attribute found on backbone.model")

        d = backbone.config.hidden_size
        self.value_head = torch.nn.Linear(d, d, bias=False, dtype=torch.float32)
        sd = load_file(str(self.dir / "value_head.safetensors"))
        self.value_head.load_state_dict({k: v.float() for k, v in sd.items()})

        self.backbone = backbone.to(device).eval()
        self.value_head = self.value_head.to(device).eval()
        self.device = device
        print(f"[AR] {backbone.config.num_hidden_layers} layers  d_model={d}  "
              f"mse_scale={self.mse_scale:.2f}  final_ln_stripped={stripped}", flush=True)

    @torch.inference_mode()
    def reconstruct(self, explanation: str) -> torch.Tensor:
        prompt = self.template.format(explanation=explanation)
        ids = self.tok(prompt, return_tensors="pt", add_special_tokens=True)["input_ids"].to(self.device)
        h = self.backbone.model(ids, use_cache=False).last_hidden_state[0, -1]   # last token
        return self.value_head(h.float()).float().cpu()

    def score(self, pred: torch.Tensor, original: torch.Tensor) -> tuple[float, float]:
        """Direction-only: normalize both to mse_scale, return (cos, mse=2(1-cos))."""
        pn = pred / pred.norm().clamp_min(1e-12) * self.mse_scale
        gn = original.float() / original.float().norm().clamp_min(1e-12) * self.mse_scale
        mse = ((pn - gn) ** 2).mean().item()
        cos = (pn @ gn / (pn.norm() * gn.norm())).item()
        return cos, mse


# ---------------------------------------------------------------------------
# AV (activation verbalizer): vector -> text
# ---------------------------------------------------------------------------
class AVVerbalizer:
    def __init__(self, repo: str, device):
        self.dir = Path(snapshot_download(repo))
        meta = yaml.safe_load((self.dir / "nla_meta.yaml").read_text())
        self.inj_id = meta["tokens"]["injection_token_id"]
        self.inj_left = meta["tokens"]["injection_left_neighbor_id"]
        self.inj_right = meta["tokens"]["injection_right_neighbor_id"]
        self.inj_char = meta["tokens"]["injection_char"]
        self.inj_scale = float(meta["extraction"]["injection_scale"])
        template = meta["prompt_templates"]["av"]

        self.tok = AutoTokenizer.from_pretrained(str(self.dir), trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            str(self.dir), torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True).eval()
        self.device = next(self.model.parameters()).device

        content = template.format(injection_char=self.inj_char)
        self.ids = self.tok.apply_chat_template(
            [{"role": "user", "content": content}], tokenize=True, add_generation_prompt=True)
        self.ids_t = torch.tensor(self.ids, dtype=torch.long).unsqueeze(0).to(self.device)
        self.attn = torch.ones(1, len(self.ids), dtype=torch.long).to(self.device)
        self.inj_pos = next(
            i for i in range(1, len(self.ids) - 1)
            if self.ids[i] == self.inj_id and self.ids[i - 1] == self.inj_left
            and self.ids[i + 1] == self.inj_right)
        print(f"[AV] inj_scale={self.inj_scale}  inj_pos={self.inj_pos}/{len(self.ids)}", flush=True)

    @torch.no_grad()
    def verbalize(self, vec: torch.Tensor, max_new_tokens: int = 200, temperature: float = 1.0,
                  scale: float | None = None) -> str:
        target = self.inj_scale if scale is None else scale
        embeds = self.model.model.embed_tokens(self.ids_t).float()
        embeds[0, self.inj_pos] = normalize(vec.to(self.device), target).to(embeds.dtype)
        out = self.model.generate(
            inputs_embeds=embeds.to(self.model.dtype), attention_mask=self.attn,
            pad_token_id=self.tok.eos_token_id, max_new_tokens=max_new_tokens,
            do_sample=True, temperature=temperature)
        raw = self.tok.decode(out[0], skip_special_tokens=False)
        m = EXPLANATION_RE.search(raw)
        return m.group(1).strip() if m else raw[:400]


# ---------------------------------------------------------------------------
def build_svd_originals() -> list[dict]:
    repo = ORGANISMS[PILOT_ORGANISM]
    adapter_dir = Path(snapshot_download(repo))
    cfg = json.loads((adapter_dir / "adapter_config.json").read_text())
    rank, alpha = cfg["r"], cfg["lora_alpha"]
    use_rslora = cfg.get("use_rslora", False)
    scale = alpha / math.sqrt(rank) if use_rslora else alpha / rank
    out = []
    with safe_open(str(adapter_dir / "adapter_model.safetensors"), framework="pt") as f:
        for module in PILOT_SVD_MODULES:
            vecs, S = residual_facing_svd(f, LAYER, module, scale)
            for i in range(min(PILOT_SVD_TOPK, vecs.shape[1])):
                out.append({
                    "id": f"svd::{PILOT_ORGANISM}::{module}::v{i}",
                    "kind": "svd", "vec": vecs[:, i].clone(),
                    "singular_value": float(S[i]),
                })
    return out


def build_avg_originals() -> list[dict]:
    prompts = []
    with open(PILOT_AVG_DATA) as fh:
        for line in fh:
            line = line.strip()
            if line:
                prompts.append(json.loads(line)["messages"][0]["content"])
            if len(prompts) >= PILOT_AVG_NPROMPTS:
                break
    print(f"loaded {len(prompts)} prompts for avg-activation capture", flush=True)

    loader = ModelLoader(ModelConfig(
        base_model_id=DEFAULT_BASE_MODEL, adapter_id=None, dtype="bfloat16", device_map="cuda"))
    base_model, tok = loader.load(with_adapter=False)
    base_model.config.use_cache = False
    model = PeftModel.from_pretrained(base_model, PILOT_AVG_ADAPTER).eval()

    avg_lora, avg_diff = collect_activations(
        model, tok, prompts, LAYER, PILOT_GEN_POSITIONS, max_new_tokens=150)
    del model, base_model
    torch.cuda.empty_cache()
    return [
        {"id": "avg::all-caps::avg_lora", "kind": "avg", "vec": avg_lora.clone()},
        {"id": "avg::all-caps::avg_diff", "kind": "avg", "vec": avg_diff.clone()},
    ]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--samples", type=int, default=5, help="AV explanation samples per vector")
    p.add_argument("--max-new-tokens", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    torch.manual_seed(args.seed)

    # ---- Stage 1+2: build originals -----------------------------------------
    print("Stage 1: SVD singular vectors ...", flush=True)
    originals = build_svd_originals()
    for o in originals:
        print(f"  {o['id']}  ||v||={o['vec'].norm():.2f}  S={o.get('singular_value'):.3f}", flush=True)

    print("\nStage 2: base+LoRA avg activations ...", flush=True)
    if args.dry_run:
        print("  (dry-run: skipping capture)  would add avg_lora, avg_diff for all-caps")
        print(f"\nPlanned originals: {len(originals)} SVD + 2 avg = {len(originals)+2}")
        print("--dry-run: exiting before loading models.")
        return
    originals += build_avg_originals()
    for o in originals:
        if o["kind"] == "avg":
            print(f"  {o['id']}  ||v||={o['vec'].norm():.2f}", flush=True)

    device = torch.device("cuda")

    # ---- Stage 3: AV verbalize ----------------------------------------------
    print(f"\nStage 3: AV verbalize ({args.samples} samples each) ...", flush=True)
    av = AVVerbalizer(NLA_AV_ID, device)
    for o in originals:
        o["explanations"] = []
        for s in range(args.samples):
            expl = av.verbalize(o["vec"], max_new_tokens=args.max_new_tokens)
            o["explanations"].append(expl)
        print(f"  [{o['id']}] sample0: {o['explanations'][0][:140]}", flush=True)
    del av.model
    av = None
    torch.cuda.empty_cache()

    # ---- Stage 4: AR reconstruct + score ------------------------------------
    print("\nStage 4: AR reconstruct + score ...", flush=True)
    ar = ARReconstructor(NLA_AR_ID, device)

    # reconstruct every (vector, sample) once; cache preds for matched + control
    for o in originals:
        o["preds"] = [ar.reconstruct(e) for e in o["explanations"]]

    config = {"experiment": "1_normalized_injection", "samples": args.samples,
              "av": NLA_AV_ID, "ar": NLA_AR_ID, "layer": LAYER,
              "mse_scale": ar.mse_scale, "av_inj_scale": 150.0}
    results = {"config": config, "vectors": {}, "summary": {}}
    av_verbalizations = {"config": config, "vectors": {}}          # AV outputs only
    ar_losses = {"config": config, "vectors": {}}                  # AR reconstruction losses only

    all_matched_cos, all_matched_mse, all_control_cos = [], [], []
    for o in originals:
        matched = [ar.score(pred, o["vec"]) for pred in o["preds"]]   # (cos, mse) per sample
        m_cos = [c for c, _ in matched]
        m_mse = [m for _, m in matched]
        pred_norms = [float(pr.norm()) for pr in o["preds"]]
        # control: this original vs OTHER vectors' reconstructions
        ctrl_cos = []
        for other in originals:
            if other["id"] == o["id"]:
                continue
            for pred in other["preds"]:
                ctrl_cos.append(ar.score(pred, o["vec"])[0])

        mc_mean = sum(m_cos) / len(m_cos)
        mm_mean = sum(m_mse) / len(m_mse)
        cc_mean = sum(ctrl_cos) / len(ctrl_cos)
        pn_mean = sum(pred_norms) / len(pred_norms)

        results["vectors"][o["id"]] = {
            "kind": o["kind"], "original_norm": float(o["vec"].norm()),
            "singular_value": o.get("singular_value"),
            "matched_cos_mean": mc_mean, "matched_cos_per_sample": m_cos,
            "matched_mse_mean": mm_mean, "control_cos_mean": cc_mean,
            "pred_norm_mean": pn_mean, "explanations": o["explanations"],
        }
        # --- separate AV verbalizations file ---
        av_verbalizations["vectors"][o["id"]] = {
            "kind": o["kind"], "original_norm": float(o["vec"].norm()),
            "singular_value": o.get("singular_value"),
            "explanations": o["explanations"],
        }
        # --- separate AR reconstruction-losses file (per-sample) ---
        ar_losses["vectors"][o["id"]] = {
            "kind": o["kind"], "original_norm": float(o["vec"].norm()),
            "per_sample": [
                {"sample": s, "cos": m_cos[s], "mse": m_mse[s], "pred_norm": pred_norms[s]}
                for s in range(len(m_cos))
            ],
            "matched_cos_mean": mc_mean, "matched_mse_mean": mm_mean,
            "control_cos_mean": cc_mean, "control_cos_per_pred": ctrl_cos,
            "pred_norm_mean": pn_mean,
        }
        all_matched_cos += m_cos
        all_matched_mse += m_mse
        all_control_cos += ctrl_cos
        print(f"  [{o['id']}] matched cos={mc_mean:+.3f} mse={mm_mean:.3f} | "
              f"control cos={cc_mean:+.3f} | ||orig||={o['vec'].norm():.1f} ||pred||~{pn_mean:.1f}",
              flush=True)

    summary = {
        "matched_cos_mean": sum(all_matched_cos) / len(all_matched_cos),
        "matched_mse_mean": sum(all_matched_mse) / len(all_matched_mse),
        "control_cos_mean": sum(all_control_cos) / len(all_control_cos),
        "n_vectors": len(originals), "n_samples_total": len(all_matched_cos),
    }
    results["summary"] = summary
    ar_losses["summary"] = summary
    print(f"\n=== SUMMARY ===\n  matched  cos={summary['matched_cos_mean']:+.3f}  "
          f"mse={summary['matched_mse_mean']:.3f}\n  control  cos={summary['control_cos_mean']:+.3f}  "
          f"(chance level)", flush=True)

    out_dir = ROOT / "results" / "ar_roundtrip"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "ar_roundtrip_exp1.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))
    (out_dir / "av_verbalizations_exp1.json").write_text(json.dumps(av_verbalizations, indent=2, ensure_ascii=False))
    (out_dir / "ar_reconstruction_losses_exp1.json").write_text(json.dumps(ar_losses, indent=2, ensure_ascii=False))
    print(f"saved -> {out_dir}/ (ar_roundtrip_exp1, av_verbalizations_exp1, ar_reconstruction_losses_exp1)", flush=True)


if __name__ == "__main__":
    main()
