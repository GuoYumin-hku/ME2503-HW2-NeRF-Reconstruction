"""Check the local HW2 handoff without using original GLBs or a GPU."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent
SLUGS = ('mug', 'apple', 'chair', 'table', 'bowl', 'bottle', 'banana', 'box', 'plate', 'cabinet')


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def verify():
    dataset = read_json(ROOT / 'data/verification.json')
    assert dataset['status'] == 'passed' and dataset['object_count'] == 10 and dataset['image_count'] == 360
    manifest = read_json(ROOT / 'data/manifest.json')
    assert {row['slug'] for row in manifest['objects']} == set(SLUGS)
    selected = read_json(ROOT / 'matches/selected/summary.json')
    assert {row['object'] for row in selected['objects']} == set(SLUGS)
    checked = 0
    for slug in SLUGS:
        data = ROOT / 'data' / slug
        meta = read_json(data / 'transforms.json')
        train = [f for f in meta['frames'] if f['split'] == 'train']
        test = [f for f in meta['frames'] if f['split'] == 'test']
        assert len(train) == 32 and len(test) == 4, slug
        assert len({f['file_path'] for f in meta['frames']}) == 36, slug
        for frame in meta['frames']:
            assert (data / frame['file_path']).is_file(), (slug, frame['file_path'])
            assert (data / frame['mask_path']).is_file(), (slug, frame['mask_path'])
        matches = read_json(ROOT / 'matches' / slug / 'summary.json')
        assert matches['object'] == slug and matches['tested_pair_count'] == 24, slug
        assert matches['transforms_sha256'] == hashlib.sha256((data / 'transforms.json').read_bytes()).hexdigest(), slug
        for file_path, expected in matches['input_sha256'].items():
            assert hashlib.sha256((data / file_path).read_bytes()).hexdigest() == expected, (slug, file_path)
        chosen = read_json(ROOT / 'matches/selected' / slug / 'summary.json')
        pair = read_json(ROOT / 'matches/selected' / slug / 'pair.json')
        assert pair['pair'] == chosen['pair'] and chosen['object'] == slug, slug
        assert (ROOT / 'matches/selected' / slug / 'point_matches.png').is_file(), slug
        checked += bool(chosen['quality_checked'])
        result = ROOT / 'results' / slug
        metrics = read_json(result / 'metrics.json')
        assert len(metrics['train_frames']) == 32 and len(metrics['test_frames']) == 4, slug
        assert metrics['config']['steps'] == 10000 and metrics['config']['seed'] == 2503, slug
        assert math.isfinite(metrics['average_psnr_db']) and math.isfinite(metrics['average_foreground_psnr_db']), slug
        sparse_path = ROOT / 'results/ablation_views_8' / slug / 'metrics.json'
        if sparse_path.is_file():
            sparse = read_json(sparse_path)
            assert len(sparse['train_frames']) == 8 and len(sparse['test_frames']) == 4, slug
            assert sparse['config']['steps'] == metrics['config']['steps'] and sparse['config']['seed'] == metrics['config']['seed'], slug
            assert [x['file_path'] for x in sparse['test_frames']] == [x['file_path'] for x in metrics['test_frames']], slug
            assert math.isfinite(sparse['average_psnr_db']) and math.isfinite(sparse['average_foreground_psnr_db']), slug
        assert (result / 'model.pt').is_file(), slug
        with Image.open(result / 'orbit.gif') as gif:
            assert gif.n_frames == 48 and gif.size == (128, 128), slug
            gif.seek(0); first = gif.convert('RGB').tobytes()
            gif.seek(24); assert gif.convert('RGB').tobytes() != first, slug
        for i in range(4):
            for suffix in ('prediction', 'reference'):
                path = result / 'heldout' / f'{i:02d}_{suffix}.png'
                with Image.open(path) as image:
                    assert image.size == (128, 128), (slug, path)
        print(f'{slug}: data=36, match QC={chosen["quality_checked"]}, '
              f'PSNR={metrics["average_psnr_db"]:.2f} dB, GIF=48 frames')
    assert checked == 8, f'expected 8 quality-checked matches, got {checked}'
    print('All ten local HW2 deliveries verified.')


if __name__ == '__main__':
    verify()
