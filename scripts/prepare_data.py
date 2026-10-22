#!/usr/bin/env python3
"""Download a documented Hugging Face split and tokenize to 1D .npy arrays.

pip install -e '.[datasets]'
python scripts/prepare_data.py --dataset wikitext2 --out data/wikitext2
python scripts/prepare_data.py --dataset tinystories --limit-train 30000 --limit-val 2000 --out data/tinystories

Do not mix WikiText and TinyStories in the same comparison; held-out splits are maintained.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from loguru import logger


SOURCES = {
    'wikitext2': ('Salesforce/wikitext', 'wikitext-2-raw-v1'),
    'tinystories': ('roneneldan/TinyStories', None),
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', choices=SOURCES, default='wikitext2')
    p.add_argument('--out', default='data/wikitext2')
    p.add_argument('--limit-train', type=int, default=0, help='0=all examples')
    p.add_argument('--limit-val', type=int, default=0, help='0=all examples')
    p.add_argument('--revision', default=None, help='Optional immutable Hugging Face revision SHA')
    args = p.parse_args()
    if args.limit_train < 0 or args.limit_val < 0:
        p.error('limits must be non-negative')
    try:
        from datasets import load_dataset
        import tiktoken
    except ImportError as err:
        p.error(f'Install optional dataset dependencies: pip install -e ".[datasets]" ({err})')
    destination = Path(args.out)
    destination.mkdir(parents=True, exist_ok=True)
    model, subset = SOURCES[args.dataset]
    tokenizer = tiktoken.get_encoding('gpt2')
    details = {}
    for split, filename, limit in [('train', 'train.npy', args.limit_train),
                                    ('validation', 'val.npy', args.limit_val)]:
        opts = {'split': split, 'streaming': True}
        if args.revision:
            opts['revision'] = args.revision
        logger.info('loading {} config={} split={}', model, subset, split)
        ds = load_dataset(model, subset, **opts) if subset else load_dataset(model, **opts)
        # Samples are concatenated with GPT-2's EOT token to preserve boundaries.
        tokens = []
        n = 0
        for sample in ds:
            if limit and n >= limit:
                break
            text = (sample.get('text') or '').strip()
            if not text:
                continue
            tokens.extend(tokenizer.encode_ordinary(text))
            tokens.append(tokenizer.eot_token)
            n += 1
        if len(tokens) < 2:
            raise RuntimeError(f'not enough tokens for split {split}')
        out = destination / filename
        np.save(out, np.asarray(tokens, dtype=np.uint16))  # GPT-2 vocabulary < 65536
        digest = hashlib.sha256(out.read_bytes()).hexdigest()
        details[split] = {'samples': n, 'tokens': len(tokens), 'sha256': digest,
                          'path': filename}
        logger.info('{}: {} samples, {} tokens -> {}', split, n, len(tokens), out)
    manifest = {'dataset': model, 'config': subset, 'requested_revision': args.revision,
                'tokenizer': 'tiktoken:gpt2', 'vocab_size': 50257, 'splits': details}
    (destination / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    logger.info('data manifest: {}', destination / 'manifest.json')


if __name__ == '__main__':
    main()
