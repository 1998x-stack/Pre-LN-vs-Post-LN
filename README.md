# Pre-LN vs Post-LN: Controlled Transformer Normalization Experiments

[![tests](https://github.com/1998x-stack/Pre-LN-vs-Post-LN/actions/workflows/ci.yml/badge.svg)](https://github.com/1998x-stack/Pre-LN-vs-Post-LN/actions/workflows/ci.yml)

An inspectable PyTorch research lab for comparing **Pre-LayerNorm**, **Post-LayerNorm**, **Post-LayerNorm + learning-rate warmup**, and the extra **Pre-LayerNorm + warmup** control. Focus: gradient propagation, activation growth, convergence, and trainability at increasing depth.

> **Status (2026-10-22):** A **108-run CPU residual-MLP stability sweep** and **192 evaluation-time block ablations** are now published with raw CSVs, static figures, and a Plotly matrix. They do **not** establish GPU/real-text LLM performance; a GPU runner workflow is provided but has not run. See [measured results](docs/empirical_results_2026-10-22.md).

## Experimental matrix

| Condition | Block form | Warmup |
|---|---|---|
| `pre` | `x + F(LN(x))` | No |
| `post` | `LN(x + F(x))` | No |
| `post_warm` | `LN(x + F(x))` | Yes |
| `pre_warm` | `x + F(LN(x))` | Yes (factorial control) |

A/B compares normalization placement; B/C isolates warmup effects for Post-LN; A/D checks whether the same benefit also occurs in Pre-LN. Warmup variants start with the **same initial weights** and use the **same batch stream** as their no-warmup pair. All arms use the same optimizer, model width/depth, final LayerNorm, and validation data.

## Experiment gallery

| Level | Network | Dataset | Research question |
|---|---|---|---|
| **1 — Simple** | 4–64 deep residual MLP | Deterministic nonlinear Gaussian classification (generated locally) | Does a direct residual gradient path change convergence and activation scale? |
| **2 — Medium** | Causal decoder Transformer with MHA + FFN | Synthetic associative key-value recall, fixed holdout | What changes in attention/FFN layer gradients and sequence retrieval accuracy? |
| **3 — Complex** | Tiny GPT-style decoder | [WikiText-2 raw](https://huggingface.co/datasets/Salesforce/wikitext) or [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) | How do depth, LR, and warmup affect language modeling validation loss / perplexity? |

For the causal recall task, samples look like `BOS K1 V7 K2 V4 ... QUERY K2`, and the target is `V4`; per-sample keys are unique and values are randomized, preventing a fixed key-to-value shortcut.

### Example figures

The diagrams below are **generated from measured runs** when you run the experiments; see `runs/<experiment>/` for the corresponding metrics. Generated figures are not automatically committed, avoiding misleading evergreen results.

- `dashboard.png`: train/validation loss, gradient norm **before clipping**, learning rate, per-layer output RMS, pre-normalization residual update ratio.
- `layer_gradients.png`: per-layer FFN+attention weight gradient magnitude (log10, a common color scale between arms).
- `interactive_dashboard.html`: Plotly dashboard with hover, trace toggling, and zoom.
- `experiment.log`: structured Loguru training progress.
- `metrics_all.csv`, `*/metrics.jsonl`, `*/profiles.json`: raw source data behind every plot.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
```

PyTorch supports CPU; NVIDIA GPU is used when available (`--device auto`). The CI checks all four arms with a tiny CPU run.

## Level 1 — Deep Residual MLP, no downloads

```bash
python -m normlab.cli \
  --stage simple --layers 16 --width 64 --heads 4 \
  --steps 500 --batch 64 --lr 0.003 --warmup 50 \
  --seeds 7 19 31 --out runs/level1
```

Dataset: `x ~ Normal(0, I)` and `label = 1[x0*x1 + .7*x2 - .55*x3*x4 + .2*sin(3*x5) > 0]`. Train=8,192 (seed 111), validation=1,024 (seed 222); the training mini-batch RNG is held constant within each paired seed. No noisy test-split leakage.

## Level 2 — Associative Recall Transformer, no downloads

```bash
python -m normlab.cli \
  --stage medium --layers 8 --width 128 --heads 4 --pairs 6 \
  --steps 1500 --batch 64 --lr 0.001 --warmup 150 \
  --seeds 7 19 31 --out runs/level2
```

Training examples are sampled using a fixed seed-specific RNG stream. The validation set uses a **separate** RNG (seed 222), stays unchanged between arms, and is evaluated in full. Query loss/accuracy are measured at the last token; no target token is included in the model input.

## Level 3 — Real text from Hugging Face

Install extra dataset dependencies:

```bash
python -m pip install -e '.[datasets]'

# Fast, small true language-model benchmark
python scripts/prepare_data.py --dataset wikitext2 --out data/wikitext2

python -m normlab.cli \
  --stage complex --train-tokens data/wikitext2/train.npy \
  --val-tokens data/wikitext2/val.npy --vocab-size 50257 \
  --layers 12 --width 256 --heads 8 --context 128 \
  --batch 16 --steps 3000 --lr 0.0003 --warmup 300 \
  --seeds 7 19 31 --out runs/wikitext2
```

**Optional larger dataset (pretraining rather than a fast smoke test):**

```bash
python scripts/prepare_data.py \
  --dataset tinystories --limit-train 30000 --limit-val 2000 \
  --out data/tinystories
```

The data preparation script streams official `train` and `validation` splits, encodes with **GPT-2 BPE via tiktoken**, inserts an end-of-text separator between nonempty samples, stores separate `.npy` files, and writes SHA256 hashes and metadata into `manifest.json`. To freeze a dataset version, pass `--revision <commit_sha>`. The tokenizer is identical across arms; `--vocab-size 50257` is mandatory for the supplied tokenizer. Data artifacts are gitignored and are **not** redistributed with this repository. For TinyStories, train on a fixed subset and preserve the official validation split.

Validation for Level 3 uses fixed windows drawn from the official validation split with a dedicated random seed. It is a reproducible proxy rather than an exhaustive evaluation over all possible contexts. Perplexity can be computed as `exp(val_loss)` for cross-entropy calculated with the stated GPT-2 tokenizer.

## Empirical robustness sweep — learning rate × depth

```bash
python scripts/run_matrix.py \
  --stage simple --depths 4 16 32 --lrs .003 .03 .1 \
  --seeds 7 19 31 --width 32 --steps 120 --warmup 24 \
  --out runs/stability
```

Produces `stability_runs.csv`, `stability_summary.csv`, `RESULTS.md`, and multiple shared-scale heatmaps (`phase_final_loss_mean.png`, `phase_near_chance_rate.png`, `phase_failure_rate.png`, etc.), with `matrix_interactive.html` for Plotly interactions. Numeric failure, near-chance learning, and transient loss spikes are separate quantities. Cached completed grid cells can be reused on reruns; `--no-resume` forces rerunning every cell.

**Measured results and plots:** [CPU 108-run sweep + layer ablation](docs/empirical_results_2026-10-22.md).

## Layer ablation

Train checkpoints with `--save-checkpoints`, then run:

```bash
python scripts/layer_ablation.py --run runs/ablation_fit --val-limit 256
```

This measures validation CE with each entire block replaced by identity, one at a time. It is not a direct test of Jacobian singular values or a decomposition of Post-LN normalization effects.

## Real-text GPU benchmark — not yet executed

On a machine with CUDA and access to the Hugging Face WikiText-2 dataset:

```bash
bash scripts/run_real_text_gpu.sh
```

The manual `.github/workflows/real_text_gpu.yml` requires a **self-hosted GPU runner** labeled `gpu`. It uploads artifacts when the job ends; it does not silently start training on hosted CPU machines. Budget time/GPU-hours before dispatching this 108-run real-text matrix.

## What plots mean (and do not mean)

1. **Gradient heatmap:** un-clipped per-layer FFN/attention parameter gradients, not a direct measurement of the complete Jacobian's singular values. Large gradients alone do not establish exploding gradients.
2. **Output RMS:** post-Block output magnitude. This can grow with Pre-LN residual accumulation; the result depends on initialization and training regime.
3. **Residual update ratio:** `||delta|| / (||residual_input||+eps)`, measured **before** Post-LN normalization. This is not the same as `||block_output-block_input||/||block_input||`, which confounds normalization with the sublayer update.
4. **Warmup:** changes optimizer LR, **not the initialization gradient**. At step 1, Pre-LN and Pre-LN+Warmup, and analogously Post-LN/Post-LN+Warmup, should have matching gradients (given a matched seed).
5. **Statistical validity:** report at least three seeds, compare validation metrics after equal token/step budget, and avoid choosing a winner from a single demonstration. Warmup also changes cumulative LR exposure; perform an area-matched schedule check for causal claims.
6. **No implicit stability crutches:** gradient clipping is **off** by default (`--grad-clip 0`) and training uses native FP32 unless the surrounding PyTorch environment changes defaults. Add clipping/mixed precision only as explicit follow-on ablations.
7. **Final norm held constant:** each architecture uses the same final LayerNorm before its output head; this intentionally isolates the *block-level* normalization placement.

## Repo layout

```text
normlab/
  data.py       # deterministic classification, recall, token-batch reader
  models.py     # Pre/Post-LN MLP and causal attention decoder
  train.py      # paired conditions, evaluation, JSONL, Loguru
  visualize.py  # Seaborn PNGs, Plotly interactive HTML, CSV
  cli.py        # run the three experimental levels
scripts/
  prepare_data.py # real, documented datasets and GPT-2 tokenization
  run_matrix.py   # depth x learning rate stability sweep
configs/         # suggested experiment settings
.github/workflows/ci.yml
  # unit tests + end-to-end CPU smoke + chart artifacts
```

## Primary literature

- Xiong et al., [On Layer Normalization in the Transformer Architecture](https://arxiv.org/abs/2002.04745), ICML 2020.
- Wang et al., [DeepNet: Scaling Transformers to 1,000 Layers](https://arxiv.org/abs/2203.00555), IEEE TPAMI 2024.
- Shleifer et al., [NormFormer: Improved Transformer Pretraining with Extra Normalization](https://arxiv.org/abs/2110.09456), EMNLP 2022.

**Interpretation:** Pre-LN protects an unnormalized identity residual path, which can make deep optimization easier. Post-LN directly constrains each block output's scale but may have less benign early optimization dynamics. Which yields better validation performance depends on depth, initialization, schedule, dataset, and training budget—not norm placement alone.
