"""Publication-oriented static charts (Seaborn) and interactive Plotly dashboard."""
from __future__ import annotations
import json
import math
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from loguru import logger

PALETTE = {'pre': '#2563eb', 'post': '#dc2626', 'post_warm': '#059669', 'pre_warm': '#9333ea'}


def load_run_tables(folder: Path):
    frames = []
    profiles = {}
    for summary_file in sorted(folder.glob('*_seed*/summary.json')):
        d = summary_file.parent
        summary = json.loads(summary_file.read_text(encoding='utf-8'))
        condition, seed = summary['condition'], summary['seed']
        records = [json.loads(line) for line in (d / 'metrics.jsonl').read_text().splitlines() if line.strip()]
        if records:
            frame = pd.DataFrame(records)
            frame['condition'], frame['seed'] = condition, seed
            frames.append(frame)
        profiles[(condition, seed)] = json.loads((d / 'profiles.json').read_text(encoding='utf-8'))
    if not frames:
        raise ValueError(f'no result metrics found in {folder}')
    return pd.concat(frames, ignore_index=True), profiles


def save_dashboard(df: pd.DataFrame, profiles: dict, output: Path):
    sns.set_theme(style='whitegrid', context='paper', font_scale=1.15)
    fig, axes = plt.subplots(2, 3, figsize=(17, 9), layout='constrained')
    curve_specs = [('train_loss', 'Train loss', 'linear'), ('val_loss', 'Validation loss', 'linear'),
                   ('grad_global_preclip', 'Gradient norm (pre-clip)', 'log'), ('lr', 'Learning rate', 'linear')]
    for ax, (field, title, scale) in zip(axes.flat[:4], curve_specs):
        usable = df[df[field].notna()] if field in df.columns else pd.DataFrame()
        if len(usable):
            sns.lineplot(data=usable, x='step', y=field, hue='condition',
                         hue_order=[x for x in PALETTE if x in usable.condition.unique()],
                         palette=PALETTE, estimator='mean', errorbar=('ci', 95), ax=ax)
        ax.set_title(title)
        ax.set_xlabel('Optimizer step')
        ax.set_yscale(scale)
    for ax, metric, name in zip(axes.flat[4:], ('output_rms', 'residual_delta_ratio'),
                                ('Block output RMS', 'Pre-normalization residual delta / input')):
        rows = []
        for (condition, seed), entries in profiles.items():
            if not entries:
                continue
            for i, item in enumerate(entries[-1]['activations']):
                if metric in item:
                    rows.append({'condition': condition, 'seed': seed, 'layer': i + 1,
                                 'value': item[metric]})
        # For Transformer, show attention delta/input as the comparable quantity.
        if not rows and metric == 'residual_delta_ratio':
            for (condition, seed), entries in profiles.items():
                if entries:
                    for i, item in enumerate(entries[-1]['activations']):
                        rows.append({'condition': condition, 'seed': seed, 'layer': i + 1,
                                     'value': item['attn_delta_ratio']})
            name = 'Attention delta / residual (before norm)'
        if rows:
            sns.lineplot(data=pd.DataFrame(rows), x='layer', y='value', hue='condition',
                         palette=PALETTE, estimator='mean', errorbar=('ci', 95), ax=ax)
        ax.set_title(name)
        ax.set_xlabel('Layer index (1-based)')
    fig.suptitle('Observed normalization placement experiments (mean and 95% bootstrap CI if seeds > 1)', fontsize=14)
    fig.savefig(output / 'dashboard.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


def save_heatmaps(profiles: dict, output: Path):
    if not profiles:
        return
    items = list(profiles.items())
    mats = []
    for (_, _), entries in items:
        if entries:
            arr = np.array([[max(x['combined'], 1e-12) for x in r['gradient_by_layer']]
                            for r in entries], dtype=float).T
            mats.append(np.log10(arr))
        else:
            mats.append(np.zeros((1, 1)))
    values = np.concatenate([x.flatten() for x in mats])
    lo, hi = np.quantile(values, [.02, .98])
    if hi <= lo:
        hi = lo + 1
    fig, axes = plt.subplots(1, len(items), figsize=(max(6, 4.5 * len(items)), 4.5),
                             sharey=True, squeeze=False, layout='constrained')
    for ax, ((condition, seed), entries), arr in zip(axes[0], items, mats):
        sns.heatmap(arr, cmap='viridis', vmin=lo, vmax=hi, cbar=False, ax=ax)
        ax.set_title(f'{condition} | seed={seed}')
        ax.set_ylabel('Layer index (zero-based)')
        steps = [e['step'] for e in entries]
        if steps:
            indices = np.unique(np.linspace(0, len(steps) - 1, min(6, len(steps)), dtype=int))
            ax.set_xticks(indices + .5, [str(steps[i]) for i in indices])
        ax.set_xlabel('Training step')
    fig.suptitle('Per-layer sublayer gradient norms (log10; common color scale)')
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    fig.colorbar(ScalarMappable(norm=Normalize(lo, hi), cmap='viridis'),
                 ax=axes.flatten().tolist(), label='log10(grad L2 norm)', shrink=.8)
    fig.savefig(output / 'layer_gradients.png', dpi=145, bbox_inches='tight')
    plt.close(fig)


def save_interactive(df: pd.DataFrame, output: Path):
    fig = make_subplots(rows=2, cols=2, subplot_titles=[
        'Train loss', 'Validation loss', 'Pre-clip gradient norm', 'Learning rate'])
    for condition in [x for x in PALETTE if x in df['condition'].unique()]:
        subset = df[df.condition == condition]
        for seed in sorted(subset.seed.unique()):
            run = subset[subset.seed == seed].sort_values('step')
            for k, metric in enumerate(('train_loss', 'val_loss', 'grad_global_preclip', 'lr')):
                m = run[run[metric].notna()]
                fig.add_trace(go.Scatter(x=m.step, y=m[metric], mode='lines',
                            name=f'{condition} seed={seed}', legendgroup=f'{condition}-{seed}',
                            showlegend=k == 0, line={'color': PALETTE[condition], 'width': 1.7}),
                            row=k // 2 + 1, col=k % 2 + 1)
    fig.update_yaxes(type='log', row=2, col=1)
    fig.update_layout(height=760, title='Pre-LN vs Post-LN: measured runs',
                      legend_title='Condition / seed', hovermode='x unified')
    fig.write_html(str(output / 'interactive_dashboard.html'), include_plotlyjs='cdn', full_html=True)


def render_report(folder: str | Path):
    folder = Path(folder)
    df, profiles = load_run_tables(folder)
    df.to_csv(folder / 'metrics_all.csv', index=False)
    save_dashboard(df, profiles, folder)
    save_heatmaps(profiles, folder)
    save_interactive(df, folder)
    logger.info('visualizations saved to {}', folder)
