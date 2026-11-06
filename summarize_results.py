"""Write aggregate metrics and a reference/prediction visual sheet."""
from __future__ import annotations

import csv
import json
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent
SLUGS = ('mug', 'apple', 'chair', 'table', 'bowl', 'bottle', 'banana', 'box', 'plate', 'cabinet')


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    rows = []
    for slug in SLUGS:
        full = load(ROOT / 'results' / slug / 'metrics.json')
        row = {'object': slug, 'train_views': len(full['train_frames']),
               'test_views': len(full['test_frames']),
               'mean_psnr_db': full['average_psnr_db'],
               'mean_foreground_psnr_db': full['average_foreground_psnr_db'],
               'train_seconds': full['elapsed_train_seconds']}
        sparse_path = ROOT / 'results/ablation_views_8' / slug / 'metrics.json'
        if sparse_path.is_file():
            sparse = load(sparse_path)
            row.update(sparse_views=len(sparse['train_frames']),
                       sparse_psnr_db=sparse['average_psnr_db'],
                       sparse_foreground_psnr_db=sparse['average_foreground_psnr_db'],
                       full_minus_sparse_psnr_db=full['average_psnr_db'] - sparse['average_psnr_db'],
                       full_minus_sparse_foreground_psnr_db=full['average_foreground_psnr_db'] - sparse['average_foreground_psnr_db'])
        rows.append(row)
    out = ROOT / 'results'
    (out / 'summary.json').write_text(json.dumps({'objects': rows}, indent=2), encoding='utf-8')
    with (out / 'summary.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(key for row in rows for key in row)))
        writer.writeheader(); writer.writerows(rows)
    panel = Image.new('RGB', (848, 5 * 174 + 36), 'white')
    draw = ImageDraw.Draw(panel)
    draw.text((10, 8), 'HW2: held-out reference / NeRF reconstruction (each pair is the same test camera)', fill='black')
    for i, row in enumerate(rows):
        col, line = i % 2, i // 2
        x, y = col * 424, 36 + line * 174
        draw.text((x + 6, y), f'{row["object"]}  PSNR {row["mean_psnr_db"]:.1f} dB', fill='black')
        for k, kind in enumerate(('reference', 'prediction')):
            with Image.open(out / row['object'] / 'heldout' / f'00_{kind}.png') as image:
                panel.paste(image.convert('RGB'), (x + 6 + k * 136, y + 20))
            draw.text((x + 6 + k * 136, y + 151), kind, fill='black')
    panel.save(out / 'overview.png')
    if all('sparse_psnr_db' in row for row in rows):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as np
        fig, axes = plt.subplots(1, 2, figsize=(11, 5.3), sharey=True)
        y = np.arange(len(rows))
        for ax, (full_key, sparse_key, title) in zip(axes, (
            ('mean_psnr_db', 'sparse_psnr_db', 'Full image'),
            ('mean_foreground_psnr_db', 'sparse_foreground_psnr_db', 'Object foreground'))):
            ax.barh(y - .18, [row[full_key] for row in rows], height=.35, label='32 training views', color='#275d8f')
            ax.barh(y + .18, [row[sparse_key] for row in rows], height=.35, label='8 training views', color='#ba6941')
            ax.set_yticks(y, [row['object'] for row in rows])
            ax.invert_yaxis(); ax.set_xlabel('Mean held-out PSNR (dB)')
            ax.set_title(title); ax.grid(axis='x', alpha=.25)
        axes[1].legend(loc='lower right')
        fig.suptitle('View-count ablation: same object, optimizer and held-out cameras')
        fig.tight_layout()
        fig.savefig(out / 'view_count_ablation.png', dpi=200)
        fig.savefig(out / 'view_count_ablation.pdf')
        plt.close(fig)
    print(f'Wrote {out / "summary.csv"} and {out / "overview.png"}')


if __name__ == '__main__':
    main()
