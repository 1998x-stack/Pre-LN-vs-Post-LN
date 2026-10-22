"""Experiment runner with paired seeds, Loguru, and immutable run manifests."""
from __future__ import annotations
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
import numpy as np
import torch
from loguru import logger
from torch.nn import functional as F
from .data import RECALL_VOCAB, classification_data, recall_batch, load_tokens, lm_batch
from .models import ResidualMLP, CausalTransformer, parameter_gradient_norms

CONDITIONS = {'pre': ('pre', False), 'post': ('post', False),
              'post_warm': ('post', True), 'pre_warm': ('pre', True)}


@dataclass
class Config:
    stage: str = 'simple'
    out: str = 'runs/simple'
    variants: tuple[str, ...] = ('pre', 'post', 'post_warm', 'pre_warm')
    seeds: tuple[int, ...] = (7, 19, 31)
    layers: int = 16
    width: int = 64
    heads: int = 4
    steps: int = 500
    batch: int = 64
    lr: float = .003
    warmup: int = 50
    log_every: int = 10
    eval_batches: int = 4
    grad_clip: float = 0.0
    pairs: int = 6
    context: int = 128
    vocab_size: int = 50257
    train_tokens: str | None = None
    val_tokens: str | None = None
    device: str = 'auto'
    save_checkpoints: bool = False

    def validate(self):
        if self.stage not in {'simple', 'medium', 'complex'}:
            raise ValueError('invalid stage')
        if any(x not in CONDITIONS for x in self.variants):
            raise ValueError('unknown variant')
        if not self.variants or not self.seeds:
            raise ValueError('at least one variant and seed required')
        for k in ('layers', 'width', 'heads', 'steps', 'batch', 'log_every', 'eval_batches'):
            if getattr(self, k) <= 0:
                raise ValueError(f'{k} must be positive')
        if self.width % self.heads or self.width < 6:
            raise ValueError('width must be >=6 and divisible by heads')
        if self.lr <= 0 or self.warmup < 0 or self.warmup > self.steps:
            raise ValueError('invalid lr/warmup')
        if self.stage == 'medium' and not 1 <= self.pairs <= 32:
            raise ValueError('medium pairs must be between 1 and 32')
        if self.stage == 'complex' and (not self.train_tokens or not self.val_tokens):
            raise ValueError('complex stage requires .npy train/validation tokens')


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_model(c: Config, kind: str):
    if c.stage == 'simple':
        return ResidualMLP(c.width, c.layers, kind)
    vocab = RECALL_VOCAB if c.stage == 'medium' else c.vocab_size
    seq = 2 * c.pairs + 3 if c.stage == 'medium' else c.context
    return CausalTransformer(vocab, c.width, c.layers, c.heads, seq, kind)


def get_loss(model, x, y, stage: str, trace=False):
    logits = model(x, trace=trace)
    if stage == 'medium':
        logits = logits[:, -1]
    if stage == 'complex':
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
        return loss, None
    loss = F.cross_entropy(logits, y)
    accuracy = float((logits.argmax(-1) == y).float().mean().item())
    return loss, accuracy


def prepare(c: Config):
    if c.stage == 'simple':
        return classification_data(111, c.width, 8192), classification_data(222, c.width, 1024)
    if c.stage == 'medium':
        return None, recall_batch(torch.Generator().manual_seed(222), 512, c.pairs)
    return load_tokens(c.train_tokens, c.vocab_size, c.context), load_tokens(c.val_tokens, c.vocab_size, c.context)


def batches(c: Config, source, gen):
    if c.stage == 'simple':
        x, y = source
        idx = torch.randint(len(x), (c.batch,), generator=gen)
        return x[idx], y[idx]
    if c.stage == 'medium':
        return recall_batch(gen, c.batch, c.pairs)
    return lm_batch(source, gen, c.batch, c.context)


@torch.no_grad()
def evaluate(model, c: Config, val, device: torch.device):
    model.eval()
    if c.stage != 'complex':
        x, y = val
        # Entire fixed validation set; chunking avoids large GPU memory use.
        loss_sum, correct, count = 0., 0., 0
        for i in range(0, len(x), c.batch):
            xx, yy = x[i:i + c.batch].to(device), y[i:i + c.batch].to(device)
            loss, accuracy = get_loss(model, xx, yy, c.stage)
            n = len(xx)
            loss_sum += loss.item() * n
            correct += accuracy * n
            count += n
        return loss_sum / count, correct / count
    # Fixed random windows, independent of training RNG; reproducible proxy eval.
    gen = torch.Generator().manual_seed(2024)
    sums = 0.
    for _ in range(c.eval_batches):
        x, y = lm_batch(val, gen, c.batch, c.context)
        loss, _ = get_loss(model, x.to(device), y.to(device), 'complex')
        sums += float(loss)
    return sums / c.eval_batches, None


def run_one(c: Config, label: str, seed: int, train, val, device: torch.device, out: Path):
    kind, warm = CONDITIONS[label]
    set_seed(seed)
    model = make_model(c, kind).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=c.lr, betas=(.9, .95), weight_decay=.01)
    rng = torch.Generator().manual_seed(10000 + seed)
    metrics, profiles = [], []
    status = 'completed'
    started = time.monotonic()
    baseline_val, baseline_acc = evaluate(model, c, val, device)
    max_grad, loss_spikes, rolling_losses = 0.0, 0, []
    logger.info('run start stage={} condition={} seed={} params={} device={}',
                c.stage, label, seed, sum(p.numel() for p in model.parameters()), device)
    for step in range(1, c.steps + 1):
        lr = c.lr * min(1., step / c.warmup) if warm and c.warmup else c.lr
        for group in optimizer.param_groups:
            group['lr'] = lr
        x, y = batches(c, train, rng)
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        sampled = step == 1 or step % c.log_every == 0 or step == c.steps
        model.train()
        loss, accuracy = get_loss(model, x, y, c.stage, trace=sampled)
        if not torch.isfinite(loss):
            status = f'non_finite_loss_step_{step}'
            logger.error('{}', status)
            break
        loss.backward()
        if sampled:
            grad, layer_grad = parameter_gradient_norms(model)
        else:
            grad = math.sqrt(sum(p.grad.detach().float().square().sum().item()
                                 for p in model.parameters() if p.grad is not None))
            layer_grad = None
        if not math.isfinite(grad):
            status = f'non_finite_grad_step_{step}'
            logger.error('{}', status)
            break
        max_grad = max(max_grad, grad)
        # A transient loss spike is a diagnostic, not by itself a run failure.
        # Compare against the PREVIOUS 20 losses, never the current value.
        previous_median = float(np.median(rolling_losses[-20:])) if len(rolling_losses) >= 10 else None
        is_spike = bool(previous_median is not None and float(loss) > max(2.0, 3.0 * previous_median))
        loss_spikes += int(is_spike)
        rolling_losses.append(float(loss))
        if c.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), c.grad_clip)
        if sampled:
            profiles.append({'step': step, 'gradient_by_layer': layer_grad,
                             'activations': [b.trace for b in model.blocks]})
        optimizer.step()
        row = {'step': step, 'lr': lr, 'train_loss': float(loss.item()),
               'train_accuracy': accuracy, 'grad_global_preclip': grad, 'loss_spike': is_spike}
        if sampled:
            row['val_loss'], row['val_accuracy'] = evaluate(model, c, val, device)
            logger.info('condition={} seed={} step={}/{} train={:.4f} val={:.4f} grad={:.4g} lr={:.3g}',
                        label, seed, step, c.steps, row['train_loss'], row['val_loss'], grad, lr)
        metrics.append(row)
    final_val = next((r['val_loss'] for r in reversed(metrics) if 'val_loss' in r), None)
    final_acc = next((r['val_accuracy'] for r in reversed(metrics) if 'val_accuracy' in r), None)
    if final_val is not None and not math.isfinite(final_val):
        status = 'non_finite_validation'
    summary = {'stage': c.stage, 'condition': label, 'seed': seed, 'status': status,
               'baseline_val_loss': baseline_val, 'baseline_val_accuracy': baseline_acc,
               'val_loss_improvement': baseline_val - final_val if final_val is not None else None,
               'final_val_accuracy': final_acc, 'max_grad_global_preclip': max_grad,
               'loss_spike_count': loss_spikes, 'peak_lr': c.lr, 'warmup_steps': c.warmup if warm else 0,
               'elapsed_seconds': round(time.monotonic() - started, 2),
               'parameters': sum(p.numel() for p in model.parameters()),
               'steps_completed': len(metrics),
               'best_val_loss': min((r['val_loss'] for r in metrics if 'val_loss' in r), default=None),
               'final_val_loss': final_val}
    dest = out / f'{label}_seed{seed}'
    dest.mkdir(parents=True, exist_ok=True)
    if c.save_checkpoints:
        torch.save(model.cpu().state_dict(), dest / 'model_state.pt')
    (dest / 'metrics.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in metrics), encoding='utf-8')
    (dest / 'profiles.json').write_text(json.dumps(profiles, indent=2), encoding='utf-8')
    (dest / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    logger.info('run complete condition={} seed={} status={} best_val_loss={}',
                label, seed, status, summary['best_val_loss'])
    return summary


def run_experiment(c: Config) -> dict:
    c.validate()
    target = Path(c.out)
    target.mkdir(parents=True, exist_ok=True)
    logger.remove()
    logger.add(lambda msg: print(msg, end=''), level='INFO', colorize=False)
    logger.add(target / 'experiment.log', rotation='10 MB', retention=3, enqueue=False)
    device = torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if c.device == 'auto' else c.device)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    (target / 'config.json').write_text(json.dumps(asdict(c), indent=2), encoding='utf-8')
    (target / 'environment.json').write_text(json.dumps({
        'torch': torch.__version__, 'numpy': np.__version__, 'device': str(device),
        'cuda_available': torch.cuda.is_available(), 'torch_deterministic': False,
    }, indent=2), encoding='utf-8')
    train, val = prepare(c)
    summaries = []
    for seed in c.seeds:
        for label in c.variants:
            summaries.append(run_one(c, label, seed, train, val, device, target))
            (target / 'summary.json').write_text(json.dumps(summaries, indent=2), encoding='utf-8')
    from .visualize import render_report
    render_report(target)
    return {'out': str(target), 'runs': summaries}
