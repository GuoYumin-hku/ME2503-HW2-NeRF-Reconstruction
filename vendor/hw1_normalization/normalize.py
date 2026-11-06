"""Create a reversible parent transform without rewriting mesh/buffer/material data."""
from __future__ import annotations

import hashlib
import json
import os
import struct
import tempfile
from pathlib import Path

import numpy as np

from .geometry import JSON_CHUNK, analyze_glb, read_glb


def normalize_glb(source: Path, destination: Path, analysis: dict) -> dict:
    result = {'status': 'skipped', 'output_path': None, 'sha256': None,
              'transform': None, 'validation': None, 'errors': []}
    source, destination = Path(source), Path(destination)
    temporary = None
    try:
        if source.resolve() == destination.resolve() or (destination.exists() and os.path.samefile(source, destination)):
            raise ValueError('source and destination must be distinct files')
        if analysis.get('status') != 'passed' or not analysis.get('normalizable'):
            result['reason'] = 'input analysis does not allow static-scene normalization'
            return result
        # Fresh inspection prevents stale caller metadata from bypassing safety checks.
        current = analyze_glb(source)
        if current['status'] != 'passed' or not current['normalizable']:
            result['reason'] = 'fresh source analysis does not allow normalization'
            result['validation'] = current
            return result
        document, chunks = read_glb(source)
        scene_index = document['scene']
        center = np.asarray(current['metrics']['center'], dtype=float)
        scale = 1.0 / current['metrics']['max_dim']
        translation = -center * scale
        if not np.isfinite(scale) or not np.all(np.isfinite(translation)):
            raise ValueError('normalization transform is non-finite')
        nodes = document['nodes']
        scene = document['scenes'][scene_index]
        parent_index = len(nodes)
        nodes.append({'name': 'HW1 unit-box normalization', 'children': list(scene['nodes']),
                      'translation': translation.tolist(), 'scale': [scale, scale, scale]})
        scene['nodes'] = [parent_index]
        transform = {'method': 'default_scene_parent_trs', 'scene_index': scene_index,
                     'parent_node': parent_index, 'source_center': center.tolist(),
                     'source_max_dim': current['metrics']['max_dim'], 'scale': scale,
                     'translation': translation.tolist(),
                     'inverse': {'scale': 1 / scale, 'translation': center.tolist()}}
        encoded = json.dumps(document, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode('utf-8')
        encoded += b' ' * (-len(encoded) % 4)
        length = 12 + 8 + len(encoded) + sum(8 + len(data) for _, data in chunks[1:])
        if length > 0xFFFFFFFF:
            raise ValueError('normalized GLB exceeds uint32 container size')
        destination.parent.mkdir(parents=True, exist_ok=True)
        fd, name = tempfile.mkstemp(prefix='.normalize-', suffix='.glb', dir=destination.parent)
        temporary = Path(name)
        with os.fdopen(fd, 'wb') as stream:
            stream.write(struct.pack('<III', 0x46546C67, 2, length))
            stream.write(struct.pack('<II', len(encoded), JSON_CHUNK))
            stream.write(encoded)
            for kind, data in chunks[1:]:
                stream.write(struct.pack('<II', len(data), kind))
                stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        validation = analyze_glb(temporary)
        bounds = validation.get('metrics', {})
        bounds_ok = bool(validation['status'] == 'passed'
                         and np.isclose(bounds.get('max_dim', 0), 1, rtol=1e-7, atol=1e-7)
                         and np.allclose(bounds.get('center', [1,1,1]), [0,0,0], rtol=0, atol=1e-7))
        validation['bounds_verified'] = bounds_ok
        result['validation'] = validation
        if not bounds_ok:
            raise ValueError('derived GLB failed independent unit-box/center verification')
        # Replace only a fully validated file. Unknown chunks and BIN bytes are unchanged.
        os.replace(temporary, destination)
        temporary = None
        digest = hashlib.sha256()
        with destination.open('rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(block)
        result.update(status='applied', output_path=str(destination.resolve()), sha256=digest.hexdigest(), transform=transform)
    except (OSError, ValueError, KeyError, TypeError, OverflowError) as exc:
        result['status'] = 'failed'
        result['errors'].append(str(exc))
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return result
