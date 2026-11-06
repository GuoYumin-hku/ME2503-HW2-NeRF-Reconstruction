"""Conservative glTF 2.0 geometry inspection; no external URI is fetched.

This is a geometry check, not a replacement for the independent Khronos
Validator. Bounds and instance counts refer only to the selected default scene.
Dynamic assets may report static-reference metrics but never become normalizable.
"""
from __future__ import annotations

import base64
import binascii
import json
import struct
from pathlib import Path

import numpy as np

JSON_CHUNK = 0x4E4F534A
BIN_CHUNK = 0x004E4942
COMPONENTS = {5120: np.dtype('i1'), 5121: np.dtype('u1'), 5122: np.dtype('<i2'),
              5123: np.dtype('<u2'), 5125: np.dtype('<u4'), 5126: np.dtype('<f4')}
WIDTHS = {'SCALAR': 1, 'VEC2': 2, 'VEC3': 3, 'VEC4': 4, 'MAT2': 4, 'MAT3': 9, 'MAT4': 16}
COMPRESSED = {'KHR_draco_mesh_compression', 'EXT_meshopt_compression', 'KHR_meshopt_compression'}
SAFE_EXTENSIONS = {
    'KHR_mesh_quantization', 'KHR_lights_punctual', 'KHR_texture_transform',
    'KHR_texture_basisu', 'EXT_texture_webp', 'EXT_texture_avif',
    'KHR_materials_unlit', 'KHR_materials_pbrSpecularGlossiness',
    'KHR_materials_clearcoat', 'KHR_materials_transmission', 'KHR_materials_volume',
    'KHR_materials_ior', 'KHR_materials_specular', 'KHR_materials_sheen',
    'KHR_materials_emissive_strength', 'KHR_materials_iridescence',
    'KHR_materials_anisotropy', 'KHR_materials_variants', 'KHR_materials_dispersion',
    'KHR_xmp_json_ld', 'MSFT_texture_dds', 'MSFT_packing_normalRoughnessMetallic',
    'MSFT_packing_occlusionRoughnessMetallic', 'EXT_texture_filter_anisotropic',
}


class InvalidGLB(ValueError):
    pass


class UnsupportedGLB(ValueError):
    pass


def _integer(value, label, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise InvalidGLB(f'{label} must be an integer >= {minimum}')
    return value


def _item(items, index, label):
    _integer(index, label)
    if not isinstance(items, list) or index >= len(items):
        raise InvalidGLB(f'{label} index {index} is out of range')
    item = items[index]
    if not isinstance(item, dict):
        raise InvalidGLB(f'{label} must reference an object')
    return item


def _reject_constant(value):
    raise InvalidGLB(f'non-finite JSON number: {value}')


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise InvalidGLB(f'duplicate JSON key: {key}')
        result[key] = value
    return result


def read_glb(path: Path):
    """Return parsed JSON and original chunk views after strict container checks."""
    raw = Path(path).read_bytes()
    if len(raw) < 20:
        raise InvalidGLB('GLB is shorter than the header and JSON chunk header')
    magic, version, declared_length = struct.unpack_from('<III', raw)
    if magic != 0x46546C67 or version != 2:
        raise InvalidGLB('expected glTF 2 GLB magic/version')
    if declared_length != len(raw) or declared_length % 4:
        raise InvalidGLB('GLB declared length differs from file length or is unaligned')
    chunks = []
    offset = 12
    while offset < len(raw):
        if offset + 8 > len(raw):
            raise InvalidGLB('truncated chunk header')
        length, kind = struct.unpack_from('<II', raw, offset)
        if length % 4 or offset + 8 + length > len(raw):
            raise InvalidGLB('chunk is unaligned or exceeds GLB range')
        if kind == JSON_CHUNK and chunks:
            raise InvalidGLB('JSON chunk must be first and unique')
        if kind == BIN_CHUNK and (len(chunks) != 1 or any(k == BIN_CHUNK for k, _ in chunks)):
            raise InvalidGLB('BIN chunk must be second and unique')
        chunks.append((kind, memoryview(raw)[offset + 8:offset + 8 + length]))
        offset += 8 + length
    if not chunks or chunks[0][0] != JSON_CHUNK:
        raise InvalidGLB('first GLB chunk is not JSON')
    try:
        document = json.loads(bytes(chunks[0][1]).decode('utf-8'),
                              parse_constant=_reject_constant, object_pairs_hook=_unique_object)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise InvalidGLB(f'invalid JSON chunk: {exc}') from exc
    if not isinstance(document, dict):
        raise InvalidGLB('GLB JSON root must be an object')
    if document.get('asset', {}).get('version') != '2.0':
        raise InvalidGLB('asset.version must be 2.0')
    return document, chunks


def _extensions(document):
    names = set(document.get('extensionsUsed', [])) | set(document.get('extensionsRequired', []))
    stack = [document]
    while stack:
        value = stack.pop()
        if isinstance(value, dict):
            ext = value.get('extensions', {})
            if isinstance(ext, dict):
                names.update(ext)
            stack.extend(value.values())
        elif isinstance(value, list):
            stack.extend(value)
    return names


class AccessorReader:
    def __init__(self, document, chunks):
        self.document = document
        self.cache = {}
        self.buffers = []
        binary = next((data for kind, data in chunks if kind == BIN_CHUNK), None)
        for index, buffer in enumerate(document.get('buffers', [])):
            length = _integer(buffer.get('byteLength'), f'buffers[{index}].byteLength', 1)
            uri = buffer.get('uri')
            if uri is not None:
                if not isinstance(uri, str) or not uri.startswith('data:'):
                    raise UnsupportedGLB('external buffer URI is not read')
                try:
                    header, content = uri.split(',', 1)
                    if not header.endswith(';base64'):
                        raise UnsupportedGLB('non-base64 embedded buffer URI')
                    data = memoryview(base64.b64decode(content, validate=True))
                except (ValueError, binascii.Error) as exc:
                    raise InvalidGLB('invalid base64 buffer URI') from exc
                if len(data) != length:
                    raise InvalidGLB('embedded URI buffer byteLength mismatch')
            else:
                if index != 0 or binary is None:
                    raise InvalidGLB('only buffer 0 can reference the GLB BIN chunk')
                data = binary
                if not length <= len(data) <= length + 3:
                    raise InvalidGLB('BIN chunk length does not match buffer byteLength')
                if any(data[length:]):
                    raise InvalidGLB('BIN chunk padding must be zero')
            self.buffers.append(data[:length])
        for index in range(len(document.get('bufferViews', []))):
            self.view(index)

    def view(self, index):
        view = _item(self.document.get('bufferViews', []), index, 'bufferView')
        buffer_index = _integer(view.get('buffer'), 'bufferView.buffer')
        if buffer_index >= len(self.buffers):
            raise InvalidGLB('bufferView.buffer is out of range')
        start = _integer(view.get('byteOffset', 0), 'bufferView.byteOffset')
        length = _integer(view.get('byteLength'), 'bufferView.byteLength', 1)
        if start + length > len(self.buffers[buffer_index]):
            raise InvalidGLB('bufferView exceeds buffer range')
        if 'byteStride' in view:
            stride = _integer(view['byteStride'], 'bufferView.byteStride', 4)
            if stride > 252 or stride % 4:
                raise InvalidGLB('bufferView.byteStride must be a multiple of 4 in [4,252]')
        return view, self.buffers[buffer_index][start:start + length]

    def _array(self, view_index, byte_offset, count, dtype, width, label, packed=False):
        view, data = self.view(view_index)
        byte_offset = _integer(byte_offset, f'{label}.byteOffset')
        item_size = dtype.itemsize * width
        stride = item_size if packed else view.get('byteStride', item_size)
        if packed and 'byteStride' in view:
            raise InvalidGLB(f'{label} bufferView cannot have byteStride')
        if stride < item_size or stride % dtype.itemsize:
            raise InvalidGLB(f'{label} stride is too small or unaligned')
        if byte_offset % dtype.itemsize or (view.get('byteOffset', 0) + byte_offset) % dtype.itemsize:
            raise InvalidGLB(f'{label} component offset is unaligned')
        required = byte_offset + (count - 1) * stride + item_size if count else byte_offset
        if required > len(data):
            raise InvalidGLB(f'{label} exceeds bufferView range')
        return np.ndarray((count, width), dtype=dtype, buffer=data, offset=byte_offset,
                          strides=(stride, dtype.itemsize))

    def read(self, index):
        if index in self.cache:
            return self.cache[index]
        accessor = _item(self.document.get('accessors', []), index, 'accessor')
        count = _integer(accessor.get('count'), 'accessor.count', 1)
        component = accessor.get('componentType')
        dtype = COMPONENTS.get(component)
        width = WIDTHS.get(accessor.get('type'))
        if dtype is None or width is None:
            raise InvalidGLB('unsupported or invalid accessor component/type')
        if accessor['type'].startswith('MAT'):
            raise UnsupportedGLB('matrix accessors are outside geometry decoding scope')
        if 'bufferView' in accessor:
            values = self._array(accessor['bufferView'], accessor.get('byteOffset', 0),
                                 count, dtype, width, 'accessor')
        else:
            if accessor.get('byteOffset', 0):
                raise InvalidGLB('accessor without bufferView has nonzero byteOffset')
            values = np.zeros((count, width), dtype=dtype)
        sparse = accessor.get('sparse')
        if sparse is not None:
            sparse_count = _integer(sparse.get('count'), 'sparse.count', 1)
            if sparse_count > count:
                raise InvalidGLB('sparse count exceeds accessor count')
            indices = sparse['indices']
            sparse_type = indices.get('componentType')
            if sparse_type not in (5121, 5123, 5125):
                raise InvalidGLB('sparse indices require an unsigned integer component type')
            replacements = self._array(indices['bufferView'], indices.get('byteOffset', 0), sparse_count,
                                       COMPONENTS[sparse_type], 1, 'sparse.indices', packed=True).ravel()
            if np.any(replacements >= count) or np.any(replacements[1:] <= replacements[:-1]):
                raise InvalidGLB('sparse indices must be strictly increasing and in range')
            source = sparse['values']
            sparse_values = self._array(source['bufferView'], source.get('byteOffset', 0), sparse_count,
                                       dtype, width, 'sparse.values', packed=True)
            values = values.copy()
            values[replacements] = sparse_values
        if accessor.get('normalized', False):
            if component not in (5120, 5121, 5122, 5123):
                raise InvalidGLB('normalized accessor must use 8/16-bit integers')
            info = np.iinfo(dtype)
            values = values.astype(np.float64) / info.max
            if info.min < 0:
                values = np.maximum(values, -1.0)
        if not np.all(np.isfinite(values)):
            raise InvalidGLB('non-finite accessor values')
        self.cache[index] = values
        return values


def _vector(value, length, label):
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (length,) or not np.all(np.isfinite(result)):
        raise InvalidGLB(f'{label} must contain {length} finite numbers')
    return result


def node_matrix(node):
    if 'matrix' in node:
        if any(key in node for key in ('translation', 'rotation', 'scale')):
            raise InvalidGLB('node contains matrix and TRS simultaneously')
        matrix = _vector(node['matrix'], 16, 'node.matrix').reshape((4, 4), order='F')
        if not np.allclose(matrix[3], [0, 0, 0, 1], rtol=0, atol=1e-12):
            raise InvalidGLB('node.matrix is not affine')
        return matrix
    translation = _vector(node.get('translation', [0, 0, 0]), 3, 'node.translation')
    scale = _vector(node.get('scale', [1, 1, 1]), 3, 'node.scale')
    quaternion = _vector(node.get('rotation', [0, 0, 0, 1]), 4, 'node.rotation')
    if not np.isclose(np.linalg.norm(quaternion), 1, atol=1e-5, rtol=0):
        raise InvalidGLB('node quaternion must have unit length')
    x, y, z, w = quaternion / np.linalg.norm(quaternion)
    rotation = np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                         [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                         [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])
    result = np.eye(4)
    result[:3, :3] = rotation * scale
    result[:3, 3] = translation
    return result


def _triangles(indices, mode):
    size = len(indices)
    if mode == 4:
        if size % 3:
            raise InvalidGLB('TRIANGLES index/vertex count is not a multiple of 3')
        return indices.reshape((-1, 3))
    if mode in (5, 6):
        if size < 3:
            raise InvalidGLB('triangle strip/fan has fewer than 3 vertices')
        if mode == 6:
            return np.column_stack((np.full(size-2, indices[0]), indices[1:-1], indices[2:]))
        result = np.column_stack((indices[:-2], indices[1:-1], indices[2:]))
        result[1::2, :2] = result[1::2, 1::-1]
        return result
    if mode == 1 and size % 2:
        raise InvalidGLB('LINES index/vertex count is not a multiple of 2')
    if mode in (2, 3) and size < 2:
        raise InvalidGLB('line loop/strip has fewer than 2 vertices')
    if mode not in (0, 1, 2, 3):
        raise InvalidGLB('primitive.mode is invalid')
    return np.empty((0, 3), dtype=np.int64)


def _scene_instances(document):
    nodes = document.get('nodes', [])
    parents = {}
    matrices = []
    for index, node in enumerate(nodes):
        if not isinstance(node, dict):
            raise InvalidGLB('node must be an object')
        matrices.append(node_matrix(node))
        for child in node.get('children', []):
            _item(nodes, child, 'node child')
            if child in parents:
                raise InvalidGLB('node has multiple parents or duplicate children')
            parents[child] = index
    # Validate cycles even in nodes not referenced by the default scene.
    finished = set()
    for index in range(len(nodes)):
        current, route = index, set()
        while current not in finished:
            if current in route:
                raise InvalidGLB('cycle in node hierarchy')
            route.add(current)
            if current not in parents:
                break
            current = parents[current]
        finished.update(route)
    scenes = document.get('scenes', [])
    if not scenes:
        raise UnsupportedGLB('no scene is available; no default-scene geometry inferred')
    scene_index = document.get('scene', 0)
    scene = _item(scenes, scene_index, 'scene')
    roots = scene.get('nodes', [])
    seen_roots = set()
    for root in roots:
        _item(nodes, root, 'scene root')
        if root in parents or root in seen_roots:
            raise InvalidGLB('scene root is a child node or appears more than once')
        seen_roots.add(root)
    pending = [(index, np.eye(4)) for index in reversed(roots)]
    while pending:
        index, parent_matrix = pending.pop()
        matrix = parent_matrix @ matrices[index]
        if not np.all(np.isfinite(matrix)):
            raise InvalidGLB('non-finite accumulated node transform')
        node = nodes[index]
        if 'mesh' in node:
            _item(document.get('meshes', []), node['mesh'], 'node.mesh')
            yield node['mesh'], matrix
        pending.extend((child, matrix) for child in reversed(node.get('children', [])))


def analyze_glb(path: Path) -> dict:
    result = {'status': 'failed', 'errors': [], 'warnings': [], 'metrics': {},
              'normalizable': False, 'features': {},
              'container': {'status': 'not_checked', 'errors': [], 'warnings': []}}
    try:
        document, chunks = read_glb(path)
        result['container']['status'] = 'passed'
    except (OSError, ValueError, KeyError, TypeError, AttributeError, OverflowError, RecursionError) as exc:
        result['errors'].append(str(exc))
        result['container'].update(status='failed', errors=[str(exc)])
        return result
    try:
        extensions = _extensions(document)
        required = set(document.get('extensionsRequired', []))
        features = result['features'] = {
            'required_extensions': sorted(required), 'extensions': sorted(extensions),
            'animations': bool(document.get('animations')), 'skins': bool(document.get('skins')),
            'morph_targets': any(p.get('targets') for m in document.get('meshes', []) for p in m.get('primitives', [])),
            'textures': len(document.get('textures', [])),
            'external_images': any('uri' in img and not img['uri'].startswith('data:') for img in document.get('images', [])),
            'scene_index': document.get('scene', 0), 'explicit_default_scene': 'scene' in document,
            'scope': 'default_scene', 'validator_coverage': 'geometry_only',
        }
        if extensions & COMPRESSED:
            raise UnsupportedGLB('compressed geometry requires an unavailable decoder: ' + ', '.join(sorted(extensions & COMPRESSED)))
        if 'EXT_mesh_gpu_instancing' in extensions:
            raise UnsupportedGLB('EXT_mesh_gpu_instancing requires instance attribute transforms')
        unsupported = required - SAFE_EXTENSIONS
        if unsupported:
            raise UnsupportedGLB('unknown required extensions: ' + ', '.join(sorted(unsupported)))
        dynamic = features['animations'] or features['skins'] or features['morph_targets']
        if any('skin' in node or 'weights' in node for node in document.get('nodes', [])):
            dynamic = True
        if any('weights' in mesh for mesh in document.get('meshes', [])):
            dynamic = True
        features['static_reference_only'] = bool(dynamic)
        reader = AccessorReader(document, chunks)
        primitive_cache = {}
        metrics = {'vertices': 0, 'triangles': 0, 'unique_triangles': 0, 'unique_vertices': 0,
                   'degenerate_triangles': 0, 'mesh_instances': 0, 'primitive_instances': 0,
                   'unique_primitives': 0, 'nonfinite_values': 0, 'primitive_modes': {}}
        low, high = np.full(3, np.inf), np.full(3, -np.inf)
        for mesh_index, matrix in _scene_instances(document):
            metrics['mesh_instances'] += 1
            mesh = document['meshes'][mesh_index]
            for primitive_index, primitive in enumerate(mesh.get('primitives', [])):
                key = (mesh_index, primitive_index)
                if key not in primitive_cache:
                    attributes = primitive.get('attributes', {})
                    if 'POSITION' not in attributes:
                        raise InvalidGLB('primitive has no POSITION attribute')
                    position_spec = _item(document.get('accessors', []), attributes['POSITION'], 'POSITION accessor')
                    if position_spec.get('type') != 'VEC3':
                        raise InvalidGLB('POSITION accessor must have type VEC3')
                    positions = reader.read(attributes['POSITION']).astype(np.float64, copy=False)
                    for name, accessor_index in attributes.items():
                        values = reader.read(accessor_index)
                        if len(values) != len(positions):
                            raise InvalidGLB(f'{name} accessor count differs from POSITION count')
                    if 'indices' in primitive:
                        spec = _item(document.get('accessors', []), primitive['indices'], 'indices accessor')
                        if spec.get('componentType') not in (5121, 5123, 5125) or spec.get('type') != 'SCALAR' or spec.get('normalized', False):
                            raise InvalidGLB('indices accessor must be unsigned integer SCALAR without normalization')
                        indices = reader.read(primitive['indices']).ravel()
                        if np.any(indices >= len(positions)):
                            raise InvalidGLB('primitive index exceeds POSITION accessor count')
                    else:
                        indices = np.arange(len(positions), dtype=np.int64)
                    mode = _integer(primitive.get('mode', 4), 'primitive.mode')
                    triangles = _triangles(indices, mode)
                    primitive_cache[key] = (positions, indices, triangles, mode)
                    metrics['unique_vertices'] += len(positions)
                    metrics['unique_triangles'] += len(triangles)
                positions, indices, triangles, mode = primitive_cache[key]
                with np.errstate(over='ignore', invalid='ignore'):
                    world = positions @ matrix[:3, :3].T + matrix[:3, 3]
                if not np.all(np.isfinite(world)):
                    raise InvalidGLB('non-finite world-space POSITION')
                visible = world[indices]
                low = np.minimum(low, visible.min(axis=0))
                high = np.maximum(high, visible.max(axis=0))
                metrics['vertices'] += len(positions)
                metrics['triangles'] += len(triangles)
                metrics['primitive_instances'] += 1
                metrics['primitive_modes'][str(mode)] = metrics['primitive_modes'].get(str(mode), 0) + 1
                if len(triangles):
                    # Scale edges before cross products to avoid unit-dependent thresholds/overflow.
                    a = world[triangles[:, 1]] - world[triangles[:, 0]]
                    b = world[triangles[:, 2]] - world[triangles[:, 0]]
                    scale = np.maximum(np.abs(a).max(axis=1), np.abs(b).max(axis=1))
                    scale[scale == 0] = 1
                    cross = np.cross(a / scale[:, None], b / scale[:, None])
                    degenerate = np.linalg.norm(cross, axis=1) <= 1e-12
                    metrics['degenerate_triangles'] += int(np.count_nonzero(degenerate))
        metrics['unique_primitives'] = len(primitive_cache)
        if not metrics['primitive_instances']:
            raise InvalidGLB('selected scene contains no geometry primitives')
        extents = high - low
        center = low / 2 + high / 2
        if not np.all(np.isfinite(extents)) or not np.all(np.isfinite(center)):
            raise InvalidGLB('non-finite scene bounds')
        metrics.update(bbox_min=low.tolist(), bbox_max=high.tolist(), extents=extents.tolist(),
                       center=center.tolist(), max_dim=float(extents.max()))
        result['metrics'] = metrics
        selected_roots = set(document['scenes'][features['scene_index']].get('nodes', []))
        features['shared_scene_roots'] = any(
            selected_roots.intersection(scene.get('nodes', []))
            for index, scene in enumerate(document['scenes']) if index != features['scene_index'])
        features['planar'] = bool(np.count_nonzero(extents > 0) == 2)
        features['points'] = '0' in metrics['primitive_modes']
        features['lines'] = any(str(mode) in metrics['primitive_modes'] for mode in (1, 2, 3))
        result['status'] = 'unsupported' if dynamic else 'passed'
        if dynamic:
            result['warnings'].append('animation/skin/morph is unsupported; metrics are static base-pose reference only')
        if metrics['degenerate_triangles']:
            result['warnings'].append(f"{metrics['degenerate_triangles']} degenerate/near-collinear triangle instances (relative cross-product threshold 1e-12)")
        if not features['explicit_default_scene']:
            features['scope'] = 'first_scene_reference'
            result['warnings'].append('default scene is absent; scene 0 used for reference only')
        if features['external_images']:
            result['warnings'].append('external image URIs preserved; relocation normalization disabled')
        if features['shared_scene_roots']:
            result['warnings'].append('default roots are shared by other scenes; parent insertion would invalidate those scene roots')
        unknown_optional = extensions - SAFE_EXTENSIONS
        if unknown_optional:
            result['warnings'].append('unknown optional extensions; normalization disabled: ' + ', '.join(sorted(unknown_optional)))
        result['normalizable'] = bool(result['status'] == 'passed' and metrics['max_dim'] > 0
                                     and features['explicit_default_scene'] and not features['external_images']
                                     and not features['shared_scene_roots'] and not unknown_optional)
    except UnsupportedGLB as exc:
        result.update(status='unsupported', normalizable=False)
        result['warnings'].append(str(exc))
    except (OSError, ValueError, KeyError, TypeError, IndexError, AttributeError, OverflowError, RecursionError) as exc:
        result.update(status='failed', normalizable=False)
        result['errors'].append(str(exc))
    return result
