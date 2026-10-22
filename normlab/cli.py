"""CLI: python -m normlab.cli --stage simple ..."""
import argparse
from .train import Config, CONDITIONS, run_experiment


def main():
    p = argparse.ArgumentParser(description='Controlled experiments comparing normalization placement')
    p.add_argument('--stage', choices=('simple', 'medium', 'complex'), default='simple')
    p.add_argument('--out', default='runs/simple')
    p.add_argument('--variants', nargs='+', choices=list(CONDITIONS), default=list(CONDITIONS))
    p.add_argument('--seeds', nargs='+', type=int, default=[7, 19, 31])
    p.add_argument('--layers', type=int, default=16)
    p.add_argument('--width', type=int, default=64)
    p.add_argument('--heads', type=int, default=4)
    p.add_argument('--steps', type=int, default=500)
    p.add_argument('--batch', type=int, default=64)
    p.add_argument('--lr', type=float, default=.003)
    p.add_argument('--warmup', type=int, default=50)
    p.add_argument('--log-every', type=int, default=10)
    p.add_argument('--eval-batches', type=int, default=4)
    p.add_argument('--grad-clip', type=float, default=0.)
    p.add_argument('--pairs', type=int, default=6)
    p.add_argument('--context', type=int, default=128)
    p.add_argument('--vocab-size', type=int, default=50257)
    p.add_argument('--train-tokens')
    p.add_argument('--val-tokens')
    p.add_argument('--device', default='auto')
    p.add_argument('--save-checkpoints', action='store_true', help='Save model weights for later layer ablation')
    a = vars(p.parse_args())
    a['variants'], a['seeds'] = tuple(a['variants']), tuple(a['seeds'])
    try:
        run_experiment(Config(**a))
    except ValueError as e:
        p.error(str(e))


if __name__ == '__main__':
    main()
