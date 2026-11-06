"""Train all ten NeRF models, with an optional sparse-view comparison."""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
OBJECTS = ('mug', 'apple', 'chair', 'table', 'bowl', 'bottle', 'banana', 'box', 'plate', 'cabinet')


def run(args, slug, output, views=None, gif_frames=48):
    cmd = [sys.executable, str(ROOT / 'nerf.py'), '--input', str(ROOT / 'data' / slug),
           '--output', str(output), '--steps', str(args.steps), '--rays', str(args.rays),
           '--seed', str(args.seed), '--gif-frames', str(gif_frames),
           '--log-every', str(args.log_every)]
    if views is not None:
        cmd += ['--train-views', str(views), '--no-model']
    print(' '.join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--steps', type=int, default=10000)
    parser.add_argument('--rays', type=int, default=2048)
    parser.add_argument('--seed', type=int, default=2503)
    parser.add_argument('--log-every', type=int, default=2000)
    parser.add_argument('--sparse-views', type=int, help='also train sparse-view models for all ten objects')
    parser.add_argument('--skip-existing', action='store_true')
    args = parser.parse_args()
    for slug in OBJECTS:
        output = ROOT / 'results' / slug
        if not args.skip_existing or not (output / 'orbit.gif').is_file() or not (output / 'metrics.json').is_file():
            run(args, slug, output)
        if args.sparse_views:
            sparse = ROOT / 'results' / f'ablation_views_{args.sparse_views}' / slug
            if not args.skip_existing or not (sparse / 'metrics.json').is_file():
                run(args, slug, sparse, args.sparse_views, gif_frames=0)


if __name__ == '__main__':
    main()
