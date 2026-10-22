#!/usr/bin/env python3
"""Perform real evaluation-time layer ablations on trained checkpoints.

Ablation = skip a whole residual block. For Post-LN this also skips both
within-block normalizations; the quantity is functional block importance,
NOT a causal proof that attention weights/updates are unused.
"""
import argparse
import json
import math
import sys
from dataclasses import fields
from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from loguru import logger
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from normlab.train import Config, CONDITIONS, prepare, evaluate, make_model, set_seed


def ablate_one(model, layer, c, val, device):
    block = model.blocks[layer]
    # Block modules differ in signature (MLP trace= / Transformer mask,trace=).
    # Replace forward, avoiding memory-heavy re-instantiation for each layer.
    forward = block.forward
    try:
        block.forward = lambda x, *args, **kwargs: x
        result, _ = evaluate(model, c, val, device)
    finally:
        block.forward = forward
    return result


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--run', required=True, help='Directory with config.json + per-arm checkpoints')
    p.add_argument('--out', default=None)
    p.add_argument('--device', default='cpu')
    a = p.parse_args(argv)
    root = Path(a.run)
    out = Path(a.out) if a.out else root / 'ablation'
    out.mkdir(parents=True, exist_ok=True)
    saved = json.loads((root / 'config.json').read_text())
    params = {k: v for k,v in saved.items() if k in {f.name for f in fields(Config)}}
    c = Config(**params)
    c.validate()
    _, val = prepare(c)
    device = torch.device(a.device)
    rows = []
    for summary_file in sorted(root.glob('*_seed*/summary.json')):
        folder = summary_file.parent
        checkpoint = folder / 'model_state.pt'
        if not checkpoint.exists():
            logger.warning('skip missing checkpoint {}', checkpoint)
            continue
        summary = json.loads(summary_file.read_text())
        condition, seed = summary['condition'], summary['seed']
        set_seed(seed)
        model = make_model(c, CONDITIONS[condition][0]).to(device)
        state = torch.load(checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(state)
        model.eval()
        original, _ = evaluate(model, c, val, device)
        for i in range(c.layers):
            mutated = ablate_one(model, i, c, val, device)
            rows.append({'condition': condition, 'seed': seed, 'layer': i+1,
                         'original_val_loss': original, 'ablated_val_loss': mutated,
                         'loss_delta': mutated-original, 'stage': c.stage})
    if not rows:
        raise RuntimeError('no checkpoints found; train using --save-checkpoints first')
    df = pd.DataFrame(rows)
    df.to_csv(out / 'layer_ablation.csv', index=False)
    sns.set_theme(style='whitegrid')
    fig, ax = plt.subplots(figsize=(11, 5.5), layout='constrained')
    sns.lineplot(data=df, x='layer', y='loss_delta', hue='condition',
                 estimator='mean', errorbar=('ci', 95), marker='o', ax=ax)
    ax.axhline(0, ls='--', color='gray', lw=1)
    ax.set(title='Measured layer-ablation effect on validation loss',
           ylabel='Validation CE with block skipped minus original CE',
           xlabel='Ablated layer (1-based)')
    fig.savefig(out / 'layer_ablation.png', dpi=170, bbox_inches='tight')
    plt.close(fig)
    logger.info('saved {} measured ablations to {}', len(df), out)
    return df

if __name__ == '__main__':
    main()
