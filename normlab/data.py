"""Deterministic task construction and evaluation batches."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import torch

N_KEYS = 32
N_VALUES = 32
RECALL_VOCAB = N_KEYS + N_VALUES + 2  # BOS=0, keys 1..32, values 33..64, QUERY=65


def classification_data(seed: int, dim: int, count: int) -> tuple[torch.Tensor, torch.Tensor]:
    if dim < 6:
        raise ValueError('classifier dim must be at least 6')
    gen = torch.Generator().manual_seed(seed)
    x = torch.randn(count, dim, generator=gen)
    s = x[:, 0] * x[:, 1] + .7 * x[:, 2] - .55 * x[:, 3] * x[:, 4] + .2 * torch.sin(3 * x[:, 5])
    return x, (s > 0).long()


def recall_batch(gen: torch.Generator, count: int, pairs: int) -> tuple[torch.Tensor, torch.Tensor]:
    if not 1 <= pairs <= N_KEYS:
        raise ValueError('1 <= pairs <= 32 required')
    keys = torch.stack([torch.randperm(N_KEYS, generator=gen)[:pairs] for _ in range(count)]) + 1
    values = torch.randint(N_KEYS + 1, N_KEYS + N_VALUES + 1, (count, pairs), generator=gen)
    query_i = torch.randint(pairs, (count,), generator=gen)
    target = values[torch.arange(count), query_i]
    tokens = torch.zeros((count, 2 * pairs + 3), dtype=torch.long)
    tokens[:, 1:-2:2] = keys
    tokens[:, 2:-2:2] = values
    tokens[:, -2] = N_KEYS + N_VALUES + 1  # QUERY
    tokens[:, -1] = keys[torch.arange(count), query_i]
    return tokens, target


def load_tokens(path: str | Path, vocab_size: int, context: int) -> np.ndarray:
    tokens = np.load(str(path), mmap_mode='r', allow_pickle=False)
    if tokens.ndim != 1 or tokens.dtype.kind not in 'iu':
        raise ValueError('token file must be a 1-D integer .npy array')
    if len(tokens) < context + 2:
        raise ValueError('corpus too short for context length')
    if tokens.min() < 0 or tokens.max() >= vocab_size:
        raise ValueError('token ID outside vocab_size')
    return tokens


def lm_batch(tokens: np.ndarray, gen: torch.Generator, batch: int, context: int):
    offsets = torch.randint(0, len(tokens) - context, (batch,), generator=gen).tolist()
    buf = np.stack([np.asarray(tokens[i:i + context + 1], dtype=np.int64) for i in offsets])
    x = torch.from_numpy(buf[:, :-1].copy())
    y = torch.from_numpy(buf[:, 1:].copy())
    return x, y
