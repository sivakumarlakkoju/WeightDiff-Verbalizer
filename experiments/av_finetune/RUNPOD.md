# Shuffled-vector control on RunPod

Runs `experiments/av_finetune/shuffled_control.py` (46 held-out concepts x {matched, mismatched,
random} with the L0-7 adapter, plus the base AV on matched vectors) on one GPU, then judges the
outputs locally. **Nothing here has been launched. Creating a pod costs money: get approval first.**

## GPU and cost

| GPU | VRAM | Secure $/hr | Community $/hr |
|---|---|---|---|
| **A40 (recommended)** | 48 GB | 0.59 | 0.35 |
| RTX A6000 (fallback) | 48 GB | 0.69 | 0.33 |
| RTX 4090 (works, tight) | 24 GB | 0.89 | 0.34 |

Prices come from the RunPod GraphQL `gpuTypes` query on 2026-11-18. Stock for A40 and A6000 was "Low",
so availability may vary.

The bf16 Qwen2.5-7B needs about 16 GB, so 48 GB leaves plenty of headroom. The job makes 184
sequential greedy generations: 46 concepts x (3 fine-tuned + 1 base), up to 250 new tokens each, and
the fine-tuned outputs average about 1,000 chars, close to the cap.

**Wall-clock estimate:** about 5 min setup, about 5-10 min for the 15 GB model download, and about
25-35 min of generation at roughly 25-30 tok/s with HF `generate`. That is **about 45 min in total,
so about $0.45 on A40 secure. Budget $1.** The `--smoke` run adds about 2 min.

## 0. Local: build the bundle (on the Mac, from the repo root)

`av_adapters/` is gitignored, and the new scripts are not committed yet. Ship one tarball of about
20 MB containing exactly what the run needs:

```bash
cd ~/Documents/WeightDiff-Verbalizer
tar czf /tmp/sc_bundle.tgz \
  training/train_av_lora_siva.py \
  training/concept_vectors_centered.pt \
  training/concept_ground_truth_v1.jsonl \
  experiments/av_finetune/shuffled_control.py \
  av_adapters/all_linear_L0-7_r4/adapter_config.json \
  av_adapters/all_linear_L0-7_r4/adapter_model.safetensors \
  av_adapters/all_linear_L0-7_r4/generation_eval.json
# Sanity check before paying for anything (CPU only, never loads the model; needs a Python
# with torch, numpy, pyyaml, transformers, peft, huggingface_hub because it imports train_av_lora_siva):
python experiments/av_finetune/shuffled_control.py --check-split
```

`generation_eval.json` must be included: the script checks its split against that file and refuses
to run if they differ.

## 1. Create the pod (requires approval: this is the billing step)

In the RunPod console, choose **Pods → Deploy**:

- **GPU:** 1x A40.
- **Template:** RunPod PyTorch 2.8 (e.g. `runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04`).
  Any CUDA 12.x image with Python 3.11 works.
- **Container disk:** 20 GB. **Volume disk:** 40 GB, mounted at `/workspace`, which holds the HF
  cache for the 15 GB model.
- **SSH:** expose TCP 22 or use the web terminal. Add your public key under Settings → SSH keys.

## 2. Copy the bundle to the pod

Option A, scp (the pod's "Connect" tab shows the IP and port):

```bash
scp -P <PORT> -i ~/.ssh/id_ed25519 /tmp/sc_bundle.tgz root@<IP>:/workspace/
```

Option B, runpodctl (preinstalled on pods; it uses a one-time croc code, so no SSH setup is needed):

```bash
runpodctl send /tmp/sc_bundle.tgz          # on the Mac; prints a code
runpodctl receive <code>                   # on the pod, inside /workspace
```

Option C, a private HF repo (useful if you will rerun often):

```bash
huggingface-cli upload <you>/av-l0-7-r4 av_adapters/all_linear_L0-7_r4 --private
```

Then `huggingface-cli download` on the pod. You still need the scripts and data from the tarball
or git.

## 3. Pod setup

```bash
cd /workspace && mkdir -p repo && tar xzf sc_bundle.tgz -C repo && cd repo
export HF_HOME=/workspace/hf_cache
python -c "import torch; print(torch.__version__, torch.cuda.get_device_name(0))"
# Pins from requirements.txt that the generation path needs. transformers/peft MUST match the
# training run (4.57.6 / 0.19.1), or the greedy outputs can drift.
pip install -q transformers==4.57.6 tokenizers==0.22.2 peft==0.19.1 accelerate==1.13.0 \
  safetensors==0.7.0 huggingface-hub==0.36.2 PyYAML==6.0.3 "numpy>=2,<3"
# Only if the image's torch is not 2.8.x:
#   pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu126
```

The full `pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu126`
also works, but it pulls wandb, trl, bitsandbytes and other packages this job does not need.

## 4. Run

```bash
cd /workspace/repo
python experiments/av_finetune/shuffled_control.py --check-split   # must print "split check OK"
python experiments/av_finetune/shuffled_control.py --smoke 2>&1 | tee smoke.log   # 2 concepts
nohup python experiments/av_finetune/shuffled_control.py > shuffled_control.log 2>&1 &
tail -f shuffled_control.log
```

`--smoke` writes `*_smoke.json` and leaves the real output alone. The full run rewrites its output
file after every concept, so a partial file survives a crash. To also run the base AV on the
mismatched and random vectors (138 more generations, about 20 more min), add
`--base-conditions matched,mismatched,random`.

At the end, the log prints how many `matched` outputs are byte-identical to the saved
`generation_eval.json`. Expect most of each output to match but not all of it: greedy bf16 decoding
is not bitwise reproducible across GPUs. The two saved eval files already disagree on `base_nla`
for all 46 concepts after about 100 chars.

## 5. Copy results back (on the Mac)

```bash
scp -P <PORT> -i ~/.ssh/id_ed25519 \
  'root@<IP>:/workspace/repo/results/av_finetune/shuffled_control_generations*.json' \
  'root@<IP>:/workspace/repo/shuffled_control.log' \
  ~/Documents/WeightDiff-Verbalizer/results/av_finetune/
# or on the pod:  runpodctl send results/av_finetune   then runpodctl receive <code> locally
```

You get `shuffled_control_generations.json` and `shuffled_control_generations_meta.json`, which
records the seeds, the derangement, the decoding settings and the library versions.

## 6. TERMINATE THE POD

Do this in the console: **Pods → ⋮ → Terminate**. "Stop" keeps billing for the volume disk.
Check that the pod list is empty.

## 7. Judge (locally, no GPU)

```bash
python experiments/av_finetune/judge_eval.py --shuffled-control --limit 5   # smoke test
python experiments/av_finetune/judge_eval.py --shuffled-control             # 230 OpenRouter calls
```

This writes `results/av_finetune/judge_{scores,summary}_shuffled_control.*`. The conditions are
`ft_matched`, `ft_mismatched` and `ft_random`, each judged against the label concept, plus
`base_matched` and `ft_mismatched_vs_source`, which takes the mismatched outputs and judges them
against the concept the vector came from.

How to read the result: if the AV tracks the vector, `ft_mismatched` and `ft_random` fall toward the
floor, and `ft_mismatched_vs_source` comes close to `ft_matched`. The derangement pairs 4 of the
46 concepts with a concept from the same category, so a few partial-credit scores on `ft_mismatched`
are expected even under perfect vector tracking.
