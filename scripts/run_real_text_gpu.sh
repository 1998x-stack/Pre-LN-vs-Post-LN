#!/usr/bin/env bash
# Reproducible benchmark on a machine with CUDA available.
# This script requires a network-enabled training machine to download WikiText-2.
set -euo pipefail
python - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit('CUDA unavailable. Do not report this as a GPU experiment.')
print('GPU:',torch.cuda.get_device_name(0),'torch:',torch.__version__)
PY
python -m pip install -e '.[datasets,dev]'
python scripts/prepare_data.py --dataset wikitext2 --out data/wikitext2
python scripts/run_matrix.py \
  --stage complex --device cuda \
  --depths 8 12 16 --lrs 0.0001 0.0003 0.001 \
  --seeds 7 19 31 --width 128 --heads 4 \
  --context 128 --vocab-size 50257 --batch 8 \
  --steps 600 --warmup 60 --log-every 100 \
  --train-tokens data/wikitext2/train.npy \
  --val-tokens data/wikitext2/val.npy \
  --out runs/gpu_wikitext2_matrix
