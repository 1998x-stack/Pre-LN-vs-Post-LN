"""Matched Pre-LN and Post-LN research networks (LayerNorm only)."""
from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F


def tensor_rms(x: torch.Tensor) -> float:
    return float(x.detach().float().square().mean().sqrt().item())


def relative_norm(delta: torch.Tensor, residual: torch.Tensor) -> float:
    return float((delta.detach().float().norm() / (residual.detach().float().norm() + 1e-12)).item())


class ResidualMLPBlock(nn.Module):
    def __init__(self, dim: int, variant: str):
        super().__init__()
        self.variant = variant
        self.norm = nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim))
        self.trace: dict | None = None

    def forward(self, x: torch.Tensor, trace: bool = False) -> torch.Tensor:
        delta = self.ff(self.norm(x) if self.variant == 'pre' else x)
        y = x + delta
        y = self.norm(y) if self.variant == 'post' else y
        if trace:
            self.trace = {'input_rms': tensor_rms(x), 'output_rms': tensor_rms(y),
                          'residual_delta_ratio': relative_norm(delta, x)}
        return y


class ResidualMLP(nn.Module):
    def __init__(self, dim: int, layers: int, variant: str):
        super().__init__()
        self.blocks = nn.ModuleList([ResidualMLPBlock(dim, variant) for _ in range(layers)])
        # A shared final normalization policy prevents the output head from being a confound.
        self.final_norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, 2)

    def forward(self, x: torch.Tensor, trace: bool = False) -> torch.Tensor:
        for block in self.blocks:
            x = block(x, trace=trace)
        return self.head(self.final_norm(x))


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, heads: int, variant: str):
        super().__init__()
        self.variant = variant
        self.norm_attn = nn.LayerNorm(dim)
        self.norm_ff = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=0.0, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))
        self.trace: dict | None = None

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor, trace: bool = False) -> torch.Tensor:
        if self.variant == 'pre':
            z = self.norm_attn(x)
            attn_delta = self.attn(z, z, z, attn_mask=attn_mask, need_weights=False)[0]
            h = x + attn_delta
            ff_delta = self.ff(self.norm_ff(h))
            y = h + ff_delta
        else:
            attn_delta = self.attn(x, x, x, attn_mask=attn_mask, need_weights=False)[0]
            h = self.norm_attn(x + attn_delta)
            ff_delta = self.ff(h)
            y = self.norm_ff(h + ff_delta)
        if trace:
            self.trace = {'input_rms': tensor_rms(x), 'output_rms': tensor_rms(y),
                          'attn_delta_ratio': relative_norm(attn_delta, x),
                          'ff_delta_ratio': relative_norm(ff_delta, h)}
        return y


class CausalTransformer(nn.Module):
    def __init__(self, vocab_size: int, dim: int, layers: int, heads: int,
                 context: int, variant: str):
        super().__init__()
        self.context = context
        self.tok = nn.Embedding(vocab_size, dim)
        self.pos = nn.Embedding(context, dim)
        self.blocks = nn.ModuleList([TransformerBlock(dim, heads, variant) for _ in range(layers)])
        self.final_norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)

    def forward(self, ids: torch.Tensor, trace: bool = False) -> torch.Tensor:
        _, t = ids.shape
        if t > self.context:
            raise ValueError(f'length {t} exceeds model context {self.context}')
        positions = torch.arange(t, device=ids.device)
        x = self.tok(ids) + self.pos(positions).unsqueeze(0)
        mask = torch.ones((t, t), device=ids.device, dtype=torch.bool).triu(1)
        for block in self.blocks:
            x = block(x, mask, trace=trace)
        return self.head(self.final_norm(x))


def parameter_gradient_norms(model: nn.Module) -> tuple[float, list[dict]]:
    """Unclipped gradients; per-layer attention/FFN grads exclude norm/head params."""
    total = sum((p.grad.detach().float().square().sum().item()
                 for p in model.parameters() if p.grad is not None), 0.0)
    layers = []
    for block in model.blocks:
        norms = {}
        for prefix in (('ff', 'ff.'), ('attn', 'attn.')):
            square = sum((p.grad.detach().float().square().sum().item()
                          for name, p in block.named_parameters()
                          if name.startswith(prefix[1]) and p.grad is not None), 0.0)
            if any(name.startswith(prefix[1]) for name, _ in block.named_parameters()):
                norms[prefix[0]] = square ** 0.5
        norms['combined'] = sum(v * v for v in norms.values()) ** .5
        layers.append(norms)
    return total ** .5, layers
