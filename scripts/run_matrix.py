#!/usr/bin/env python3
"""Repeat experiments over depth/LR and construct empirical stability maps.

Example: python scripts/run_matrix.py --depths 4 16 32 --lrs 0.001 0.003 0.01 --steps 250
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--depths', nargs='+', type=int, default=[4, 16, 32])
    p.add_argument('--lrs', nargs='+', type=float, default=[.001, .003, .01])
    p.add_argument('--seeds', nargs='+', type=int, default=[7, 19, 31])
    p.add_argument('--variants', nargs='+', default=['pre', 'post', 'post_warm', 'pre_warm'])
    p.add_argument('--steps', type=int, default=250)
    p.add_argument('--warmup', type=int, default=50)
    p.add_argument('--out', default='runs/matrix')
    args = p.parse_args()
    root = Path(args.out)
    root.mkdir(parents=True, exist_ok=True)
    rows = []
    for depth in args.depths:
        for lr in args.lrs:
            folder = root / f'depth{depth}_lr{lr:g}'
            cmd = [sys.executable, '-m', 'normlab.cli', '--stage', 'simple', '--out', str(folder),
                   '--layers', str(depth), '--steps', str(args.steps), '--warmup', str(args.warmup),
                   '--lr', str(lr), '--variants', *args.variants, '--seeds', *map(str, args.seeds)]
            subprocess.run(cmd, check=True)
            for result in json.loads((folder / 'summary.json').read_text()):
                rows.append({'depth': depth, 'lr': lr, **result,
                             'failed': int(result['status'] != 'completed')})
    table = pd.DataFrame(rows)
    table.to_csv(root / 'stability.csv', index=False)
    for condition, entries in table.groupby('condition'):
        matrix = entries.pivot_table(index='depth', columns='lr', values='failed', aggfunc='mean')
        plt.figure(figsize=(7, 4))
        sns.heatmap(matrix, annot=True, fmt='.2f', cmap='Reds', vmin=0, vmax=1)
        plt.title(f'{condition} empirical failure fraction (NaN/Inf only)')
        plt.savefig(root / f'stability_{condition}.png', dpi=160, bbox_inches='tight')
        plt.close()


if __name__ == '__main__':
    main()
