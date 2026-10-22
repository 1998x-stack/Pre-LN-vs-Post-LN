#!/usr/bin/env python3
"""Controlled stability phase diagrams with paired seeds and normalized quality metrics.

Examples:
  python scripts/run_matrix.py --depths 4 16 32 --lrs .003 .01 .03 \\
    --seeds 7 19 31 --steps 100 --width 32 --warmup 20 --out runs/cpu_matrix

Plots are strictly empirical; a numerical-failure map may be all zeros when no
NaN/Inf failures are observed. Nonlearning and loss spikes are shown separately.
"""
from __future__ import annotations
import argparse
import json
import math
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from loguru import logger

# Make script importable from a checkout even when package wasn't installed.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from normlab.train import Config, CONDITIONS, run_experiment


def collect(rows, depth, lr, directory, min_improvement):
    for run in json.loads((directory / 'summary.json').read_text()):
        failed = int(run['status'] != 'completed')
        improvement = run.get('val_loss_improvement')
        did_not_learn = int(failed or improvement is None or not math.isfinite(improvement)
                            or improvement < min_improvement)
        rows.append({'depth': depth, 'lr': lr, **run, 'failed': failed,
                     'did_not_learn': did_not_learn,
                     'spikes_per_100_steps': 100 * run.get('loss_spike_count', 0)
                     / max(1, run['steps_completed'])})


def summarize(df):
    return (df.groupby(['condition', 'depth', 'lr'], as_index=False)
            .agg(n=('seed', 'count'), failure_rate=('failed', 'mean'),
                 nonlearning_rate=('did_not_learn', 'mean'),
                 final_loss_mean=('final_val_loss', 'mean'),
                 final_loss_std=('final_val_loss', 'std'),
                 improvement_mean=('val_loss_improvement', 'mean'),
                 spike_rate=('spikes_per_100_steps', 'mean'),
                 max_grad_mean=('max_grad_global_preclip', 'mean')))


def plots(table, out):
    sns.set_theme(style='white', context='talk', font_scale=.74)
    order = ['pre', 'post', 'post_warm', 'pre_warm']
    measures = [
        ('failure_rate', 'Numerical failure fraction (NaN/Inf)', 'Reds', 0, 1, '.2f'),
        ('nonlearning_rate', 'Did-not-learn fraction (thresholded)', 'OrRd', 0, 1, '.2f'),
        ('improvement_mean', 'Validation loss improvement', 'YlGnBu', None, None, '.3f'),
        ('final_loss_mean', 'Final validation cross-entropy', 'viridis_r', None, None, '.3f'),
        ('spike_rate', 'Loss spikes per 100 steps', 'magma', 0, None, '.1f'),
    ]
    for key, title, cmap, low, high, fmt in measures:
        fig, axes = plt.subplots(1, 4, figsize=(20, 4.8), layout='constrained', sharey=True)
        for ax, variant in zip(axes, order):
            one = table[table.condition == variant]
            matrix = one.pivot(index='depth', columns='lr', values=key).sort_index(ascending=False)
            sns.heatmap(matrix, ax=ax, annot=True, fmt=fmt, cmap=cmap, vmin=low,
                        vmax=high, cbar=(variant == order[-1]), linewidths=.5,
                        linecolor='white', annot_kws={'size': 10})
            ax.set_title(variant)
            ax.set_xlabel('Peak learning rate')
            ax.set_ylabel('Transformer / MLP depth' if variant == 'pre' else '')
        fig.suptitle(f'{title} | observed, not extrapolated', fontsize=16)
        fig.savefig(out / f'phase_{key}.png', dpi=150, bbox_inches='tight')
        plt.close(fig)

    # Scientific interpretation: paired seeds and learning curves, not simply failures.
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), layout='constrained')
    sns.lineplot(data=table, x='depth', y='final_loss_mean', hue='condition', style='lr',
                 markers=True, dashes=False, errorbar=None, ax=axes[0])
    axes[0].set_title('Final validation loss across depth / LR')
    sns.lineplot(data=table, x='depth', y='max_grad_mean', hue='condition', style='lr',
                 markers=True, dashes=False, errorbar=None, ax=axes[1])
    axes[1].set_title('Maximum observed pre-clip gradient norm')
    axes[1].set_yscale('log')
    fig.savefig(out / 'depth_gradient_comparison.png', dpi=160, bbox_inches='tight')
    plt.close(fig)

    fig = make_subplots(rows=1, cols=2, subplot_titles=('Final loss', 'Loss improvement'))
    for condition in order:
        one = table[table.condition == condition]
        for lr in sorted(one.lr.unique()):
            subset = one[one.lr == lr].sort_values('depth')
            name = f'{condition} | lr={lr:g}'
            for j, metric in enumerate(('final_loss_mean', 'improvement_mean')):
                fig.add_trace(go.Scatter(x=subset.depth, y=subset[metric], mode='lines+markers',
                                         name=name, legendgroup=name, showlegend=j == 0),
                              row=1, col=j+1)
    fig.update_layout(title='Measured stability / learning sensitivity',
                      height=520, hovermode='x unified')
    fig.write_html(str(out / 'matrix_interactive.html'), include_plotlyjs='cdn')


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--stage', choices=('simple', 'medium', 'complex'), default='simple')
    p.add_argument('--depths', nargs='+', type=int, default=[4, 16, 32])
    p.add_argument('--lrs', nargs='+', type=float, default=[.003, .01, .03])
    p.add_argument('--seeds', nargs='+', type=int, default=[7, 19, 31])
    p.add_argument('--variants', nargs='+', choices=list(CONDITIONS), default=list(CONDITIONS))
    p.add_argument('--steps', type=int, default=100)
    p.add_argument('--warmup', type=int, default=20)
    p.add_argument('--width', type=int, default=32)
    p.add_argument('--heads', type=int, default=4)
    p.add_argument('--batch', type=int, default=64)
    p.add_argument('--log-every', type=int, default=25)
    p.add_argument('--min-improvement', type=float, default=.02,
                   help='Threshold applied to baseline minus final validation CE; not numerical failure')
    p.add_argument('--context', type=int, default=128)
    p.add_argument('--vocab-size', type=int, default=50257)
    p.add_argument('--train-tokens')
    p.add_argument('--val-tokens')
    p.add_argument('--device', default='auto')
    p.add_argument('--out', default='runs/stability_matrix')
    p.add_argument('--no-resume', action='store_true')
    args = p.parse_args(argv)
    root = Path(args.out)
    root.mkdir(parents=True, exist_ok=True)
    (root / 'matrix_config.json').write_text(json.dumps(vars(args), indent=2))
    rows = []
    for depth in args.depths:
        for lr in args.lrs:
            folder = root / f'depth{depth}_lr{lr:g}'
            if not ((folder / 'summary.json').exists() and not args.no_resume):
                config = Config(stage=args.stage, out=str(folder),
                                variants=tuple(args.variants), seeds=tuple(args.seeds),
                                layers=depth, width=args.width, heads=args.heads,
                                steps=args.steps, batch=args.batch, lr=lr, warmup=args.warmup,
                                log_every=args.log_every, context=args.context,
                                vocab_size=args.vocab_size, train_tokens=args.train_tokens,
                                val_tokens=args.val_tokens, device=args.device)
                logger.info('running stage={} depth={} lr={} seeds={}',
                            args.stage, depth, lr, args.seeds)
                run_experiment(config)
            else:
                logger.info('loading already observed depth={} lr={}', depth, lr)
            collect(rows, depth, lr, folder, args.min_improvement)
            pd.DataFrame(rows).to_csv(root / 'stability_runs.csv', index=False)
    df = pd.DataFrame(rows)
    table = summarize(df)
    table.to_csv(root / 'stability_summary.csv', index=False)
    plots(table, root)
    (root / 'RESULTS.md').write_text(
        '# Stability experiment results (measured)\n\n'
        f'- Stage: `{args.stage}`; device: `{args.device}` (see individual environment.json)\n'
        f'- Depths: `{args.depths}`; peak LR: `{args.lrs}`; seeds: `{args.seeds}`\n'
        f'- Steps/run: `{args.steps}`; warmup steps: `{args.warmup}`\n'
        f'- Total measured runs: **{len(df)}**\n'
        f'- Nonlearning criterion: `baseline_val_loss - final_val_loss < {args.min_improvement}`. '
        'This is not a numerical failure and is sensitive to the training budget.\n'
        '- Numerical failure: nonfinite loss / gradient / validation as detected by the runner.\n'
        '- Loss spike: training loss > max(2.0, 3x preceding 20-step median) after at least 10 prior steps.\n\n'
        '## Aggregate per-configuration measurements\n\n'
        + table.to_markdown(index=False, floatfmt='.4f') + '\n\n'
        '## Limitations\n\n'
        'These are measured outcomes only for the specified dataset, seeds and training budget. '
        'An all-zero numerical-failure map does NOT show universal stability. '
        'Warmup groups differ from non-warmup groups in early cumulative learning-rate exposure.\n',
        encoding='utf-8')
    logger.info('completed {} measured runs; table/plots at {}', len(df), root)
    return df, table


if __name__ == '__main__':
    main()
