"""Select a presentable correspondence pair per object, preserving raw trials.

The known calibrated cameras are used here only for a post-hoc geometric
quality check. All attempts, including failed ones, remain under matches/.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from match_features import draw_matches, extract, image_path, write_json

TRIALS = (
    ('sift_080', ROOT / 'matches'),
    ('sift_090', ROOT / 'matches/experiments/sift_090'),
    ('orb_090', ROOT / 'matches/experiments/orb_090'),
    ('sift_090_more_pairs', ROOT / 'matches/experiments/sift_090_100'),
    ('orb_090_more_pairs', ROOT / 'matches/experiments/orb_090_100'),
)


def matches_current_data(data_dir, trial_dir, slug):
    path = trial_dir / slug / 'summary.json'
    if not path.is_file():
        return False
    summary = json.loads(path.read_text(encoding='utf-8'))
    if hashlib.sha256((data_dir / 'transforms.json').read_bytes()).hexdigest() != summary['transforms_sha256']:
        return False
    return all(hashlib.sha256((data_dir / name).read_bytes()).hexdigest() == digest
               for name, digest in summary['input_sha256'].items())


def quality(pair):
    inliers = pair['ransac_inliers']
    diagnostic = pair.get('calibrated_diagnostic', {})
    consistent = diagnostic.get('within_threshold_ransac_inliers', 0)
    median = diagnostic.get('median_sampson_pixels_ransac_inliers')
    return (inliers >= 8 and consistent >= .75 * inliers
            and median is not None and median <= 1.5)


def run():
    destination = ROOT / 'matches/selected'
    destination.mkdir(parents=True, exist_ok=True)
    manifest = []
    for data_dir in sorted((ROOT / 'data').iterdir()):
        if not (data_dir / 'transforms.json').is_file():
            continue
        candidates = []
        for label, source in TRIALS:
            if not matches_current_data(data_dir, source, data_dir.name):
                continue
            for path in (source / data_dir.name).glob('pair_*.json'):
                pair = json.loads(path.read_text(encoding='utf-8'))
                candidates.append((quality(pair), pair['ransac_inliers'],
                                   pair['ratio_mutual_count'], label, path, pair))
        if not candidates:
            raise ValueError(f'no current match results: {data_dir.name}')
        _, _, _, label, path, pair = max(candidates, key=lambda x: (x[0], x[1], x[2], x[3]))
        out_dir = destination / data_dir.name
        out_dir.mkdir(exist_ok=True)
        write_json(out_dir / 'pair.json', pair)
        detector = (cv2.SIFT_create(nfeatures=4000, contrastThreshold=.01, edgeThreshold=12)
                    if pair['detector'] == 'sift' else
                    cv2.ORB_create(nfeatures=3000, edgeThreshold=8, patchSize=21, fastThreshold=7))
        meta = json.loads((data_dir / 'transforms.json').read_text(encoding='utf-8'))
        frames = {frame['file_path']: frame for frame in meta['frames']}
        inputs = []
        for name in (pair['image_1'], pair['image_2']):
            frame = frames[name]
            inputs.append(extract(image_path(data_dir, frame), detector, data_dir / frame['mask_path']))
        draw_matches(*inputs, pair, out_dir / 'point_matches.png')
        row = {'object': data_dir.name, 'quality_checked': quality(pair), 'trial': label,
               'source_pair_json': str(path.relative_to(ROOT)).replace('\\', '/'),
               'pair': pair['pair'], 'detector': pair['detector'],
               'mutual_matches': pair['ratio_mutual_count'], 'fundamental_inliers': pair['ransac_inliers'],
               'calibrated_consistent_inliers': pair['calibrated_diagnostic']['within_threshold_ransac_inliers'],
               'calibrated_median_error_px': pair['calibrated_diagnostic']['median_sampson_pixels_ransac_inliers'],
               'status': pair['status']}
        write_json(out_dir / 'summary.json', row)
        manifest.append(row)
        print(f'{data_dir.name}: {label} {row["fundamental_inliers"]}/{row["mutual_matches"]}, '
              f'known-pose consistent {row["calibrated_consistent_inliers"]}; quality={row["quality_checked"]}')
    write_json(destination / 'summary.json', {'rule': 'minimum 8 F-RANSAC inliers, at least 75% within 1.5 px of calibrated epipolar geometry, median <= 1.5 px; then choose highest inlier count',
                                              'known_pose_use': 'post-hoc selection and diagnostic only',
                                              'objects': manifest})


if __name__ == '__main__':
    run()
