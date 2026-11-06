"""Compact feature-grid NeRF, optimized solely against posed RGB images.

Camera matrices follow Blender/OpenGL convention (+X right, +Y up, -Z forward).
The field is a trilinear 3-D feature grid with a small directional RGB MLP;
density is a separate trainable grid. Both are optimized by volume rendering.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import torch
from torch import nn
import torch.nn.functional as F


def make_rays(height, width, focal, c2w):
    """Ray origins and unit directions through pixel centers in world space."""
    y, x = torch.meshgrid(torch.arange(height, device=c2w.device, dtype=c2w.dtype),
                          torch.arange(width, device=c2w.device, dtype=c2w.dtype), indexing='ij')
    local = torch.stack(((x + .5 - width / 2) / focal,
                         -(y + .5 - height / 2) / focal, -torch.ones_like(x)), -1)
    directions = F.normalize(local @ c2w[:3, :3].T, dim=-1)
    return c2w[:3, 3].expand_as(directions), directions


def volume_composite(rgb, sigma, deltas, background=1.0):
    """Emission-absorption integral against a constant RGB background."""
    alpha = 1.0 - torch.exp(-sigma * deltas)
    trans = torch.cumprod(torch.cat((torch.ones_like(alpha[..., :1]),
                                   1.0 - alpha + 1e-10), dim=-1), dim=-1)[..., :-1]
    weights = trans * alpha
    opacity = weights.sum(dim=-1)
    color = (weights[..., None] * rgb).sum(dim=-2) + (1.0 - opacity)[..., None] * background
    return color, opacity, weights


def generate_split(frames, split):
    if split not in ('train', 'test'):
        raise ValueError(f'unknown split: {split}')
    result = [f for f in frames if f.get('split') == split]
    if not result:
        raise ValueError(f'no {split} images found')
    return result


def diverse_view_indices(positions, count):
    """Greedy farthest-camera selection for a sparse-view ablation."""
    positions = np.asarray(positions, dtype=np.float64)
    if not 1 <= count <= len(positions):
        raise ValueError('invalid view count')
    directions = positions / np.maximum(np.linalg.norm(positions, axis=1, keepdims=True), 1e-12)
    selected = [0]
    best_distance = np.linalg.norm(directions - directions[0], axis=1)
    for _ in range(count - 1):
        best_distance[selected] = -1
        index = int(np.argmax(best_distance))
        selected.append(index)
        best_distance = np.minimum(best_distance, np.linalg.norm(directions - directions[index], axis=1))
    return sorted(selected)


class FeatureNeRF(nn.Module):
    def __init__(self, resolution=64, channels=8, bound=.65):
        super().__init__()
        self.resolution, self.channels, self.bound = resolution, channels, bound
        self.density = nn.Parameter(torch.full((1, 1, resolution, resolution, resolution), -2.0))
        self.features = nn.Parameter(torch.randn((1, channels, resolution, resolution, resolution)) * .1)
        self.color = nn.Sequential(nn.Linear(channels + 3, 64), nn.ReLU(), nn.Linear(64, 3), nn.Sigmoid())

    def forward(self, points, directions):
        shape = points.shape[:-1]
        xyz = points.reshape(1, 1, 1, -1, 3) / self.bound
        # Tensor storage is D,H,W -> grid_sample coordinate order x,y,z.
        density = F.grid_sample(self.density, xyz, align_corners=True, padding_mode='zeros').reshape(*shape)
        feature = F.grid_sample(self.features, xyz, align_corners=True, padding_mode='zeros')
        feature = feature.reshape(self.channels, -1).T.reshape(*shape, self.channels)
        inside = (points.abs() <= self.bound).all(dim=-1)
        sigma = F.softplus(density) * inside
        rgb = self.color(torch.cat((feature, directions), dim=-1))
        return rgb, sigma


def render_rays(field, origins, directions, near, far, n_samples, randomized=False):
    bins = torch.linspace(near, far, n_samples + 1, device=origins.device, dtype=origins.dtype)
    fractions = (torch.rand((len(origins), n_samples), device=origins.device)
                 if randomized else torch.full((len(origins), n_samples), .5, device=origins.device))
    z = bins[:-1] + (bins[1:] - bins[:-1]) * fractions
    pts = origins[:, None, :] + directions[:, None, :] * z[..., None]
    dirs = directions[:, None, :].expand_as(pts)
    rgb, sigma = field(pts, dirs)
    return volume_composite(rgb, sigma, (bins[1:] - bins[:-1]).expand_as(z), field.background)[:2]


def load_dataset(directory, device, train_views=None):
    directory = Path(directory)
    meta = json.loads((directory / 'transforms.json').read_text(encoding='utf-8-sig'))
    width, height = int(meta['width']), int(meta['height'])
    focal = .5 * width / math.tan(.5 * float(meta['camera_angle_x']))
    sets = {}
    for split in ('train', 'test'):
        frames = generate_split(meta['frames'], split)
        if split == 'train' and train_views is not None:
            if not 1 <= train_views <= len(frames):
                raise ValueError('train-views must be between 1 and total training views')
            idx = diverse_view_indices([np.asarray(f['transform_matrix'])[:3, 3] for f in frames], train_views)
            frames = [frames[i] for i in idx]
        origins, dirs, rgbs, images, masks, poses = [], [], [], [], [], []
        for frame in frames:
            path = directory / frame['file_path']
            if not path.is_file() and path.suffix == '':
                path = path.with_suffix('.png')
            with Image.open(path) as image:
                image = image.convert('RGBA')
                if image.size != (width, height):
                    raise ValueError(f'image size mismatch: {path} {image.size}')
                bg_rgb = tuple(int(round(x * 255)) for x in meta.get('background_rgb', [1., 1., 1.]))
                background = Image.new('RGBA', image.size, (*bg_rgb, 255))
                background.alpha_composite(image)
                pixels = np.asarray(background.convert('RGB'), dtype=np.float32) / 255.
            mask_path = directory / frame['mask_path']
            with Image.open(mask_path) as mask_image:
                mask = np.asarray(mask_image.convert('L'), dtype=np.float32) / 255.
            if mask.shape != (height, width):
                raise ValueError(f'mask size mismatch: {mask_path}')
            pose = torch.tensor(frame['transform_matrix'], dtype=torch.float32, device=device)
            o, d = make_rays(height, width, focal, pose)
            origins.append(o.reshape(-1, 3)); dirs.append(d.reshape(-1, 3))
            rgbs.append(torch.tensor(pixels, device=device).reshape(-1, 3))
            images.append(pixels); masks.append(mask); poses.append(pose)
        sets[split] = {'origins': torch.stack(origins), 'dirs': torch.stack(dirs),
                       'rgb': torch.stack(rgbs), 'images': images,
                       'masks': torch.tensor(np.stack(masks), device=device),
                       'poses': poses, 'frames': frames}
    return meta, focal, sets


def image_render(field, pose, height, width, focal, args):
    origins, dirs = make_rays(height, width, focal, pose)
    pixels, opacity = [], []
    with torch.no_grad():
        for o, d in zip(origins.reshape(-1, 3).split(args.chunk), dirs.reshape(-1, 3).split(args.chunk)):
            rgb, alpha = render_rays(field, o, d, args.near, args.far, args.samples)
            pixels.append(rgb); opacity.append(alpha)
    return (torch.cat(pixels).reshape(height, width, 3).clamp(0, 1).cpu().numpy(),
            torch.cat(opacity).reshape(height, width).cpu().numpy())


def make_orbit_pose(angle, elevation, radius, device):
    """Look at the origin with world Z up, in Blender's camera convention."""
    position = torch.tensor([radius * math.cos(elevation) * math.cos(angle),
                             radius * math.cos(elevation) * math.sin(angle),
                             radius * math.sin(elevation)], device=device, dtype=torch.float32)
    back = F.normalize(position, dim=0)
    right = F.normalize(torch.linalg.cross(torch.tensor([0., 0., 1.], device=device), back), dim=0)
    up = torch.linalg.cross(back, right)
    pose = torch.eye(4, device=device)
    pose[:3, :3] = torch.stack((right, up, back), dim=1)
    pose[:3, 3] = position
    return pose


def save_image(path, array):
    Image.fromarray(np.round(np.clip(array, 0, 1) * 255).astype(np.uint8)).save(path)


def run(args):
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    meta, focal, data = load_dataset(args.input, device, args.train_views)
    h, w = int(meta['height']), int(meta['width'])
    train = data['train']
    radius = float(torch.linalg.vector_norm(train['poses'][0][:3, 3]))
    if args.near is None:
        args.near = max(.01, radius - 1.0)
    if args.far is None:
        args.far = radius + 1.0
    if not 0 < args.near < args.far:
        raise ValueError('ray range must satisfy 0 < near < far')
    field = FeatureNeRF(args.grid, args.channels, args.bound).to(device)
    field.register_buffer('background', torch.tensor(meta.get('background_rgb', [1., 1., 1.]), device=device))
    optim = torch.optim.Adam([{'params': [field.features], 'lr': args.grid_lr},
                              {'params': [field.density], 'lr': args.density_lr},
                              {'params': field.color.parameters(), 'lr': args.mlp_lr}])
    rays_o = train['origins'].reshape(-1, 3); rays_d = train['dirs'].reshape(-1, 3)
    colors = train['rgb'].reshape(-1, 3)
    # Ground-truth alpha masks allow balanced ray sampling and silhouette supervision.
    masks = train['masks'].reshape(-1)
    fg = torch.nonzero(masks > .5, as_tuple=True)[0]
    bg = torch.nonzero(masks <= .5, as_tuple=True)[0]
    if not len(fg) or not len(bg):
        raise ValueError('expected both object foreground and background')
    losses, start = [], time.time()
    for step in range(args.steps):
        n_fg = args.rays // 2
        index = torch.cat((fg[torch.randint(len(fg), (n_fg,), device=device)],
                           bg[torch.randint(len(bg), (args.rays - n_fg,), device=device)]))
        pred, opacity = render_rays(field, rays_o[index], rays_d[index], args.near, args.far, args.samples, True)
        rgb_loss = F.mse_loss(pred, colors[index])
        mask_loss = F.mse_loss(opacity, masks[index])
        loss = rgb_loss + args.mask_weight * mask_loss
        optim.zero_grad(set_to_none=True); loss.backward(); optim.step()
        # Smooth exponential schedule, same multiplier for each parameter group.
        factor = .1 ** ((step + 1) / args.steps)
        for group, initial in zip(optim.param_groups, (args.grid_lr, args.density_lr, args.mlp_lr)):
            group['lr'] = initial * factor
        if step == 0 or (step + 1) % args.log_every == 0 or step + 1 == args.steps:
            value = float(loss.detach())
            losses.append({'step': step + 1, 'loss': value, 'rgb_mse': float(rgb_loss.detach()),
                           'mask_mse': float(mask_loss.detach()), 'elapsed_seconds': round(time.time() - start, 2)})
            print(f'{out.name} step {step+1}/{args.steps}: loss={value:.6f}, {time.time()-start:.1f}s', flush=True)
    seconds = time.time() - start
    if not args.no_model:
        torch.save({'state_dict': field.state_dict(), 'resolution': args.grid, 'channels': args.channels,
                    'bound': args.bound, 'focal': focal, 'height': h, 'width': w,
                    'near': args.near, 'far': args.far, 'samples': args.samples,
                    'seed': args.seed}, out / 'model.pt')
    eval_dir = out / 'heldout'; eval_dir.mkdir(exist_ok=True)
    metrics = []
    for i, (image, mask, pose, frame) in enumerate(zip(data['test']['images'], data['test']['masks'], data['test']['poses'], data['test']['frames'])):
        prediction, opacity = image_render(field, pose, h, w, focal, args)
        error = (prediction - image) ** 2
        foreground = mask.detach().cpu().numpy() > .5
        mse = float(error.mean()); fg_mse = float(error[foreground].mean()) if foreground.any() else None
        save_image(eval_dir / f'{i:02d}_prediction.png', prediction)
        save_image(eval_dir / f'{i:02d}_reference.png', image)
        metrics.append({'file_path': frame['file_path'], 'mse': mse,
                        'psnr_db': -10 * math.log10(max(mse, 1e-12)),
                        'foreground_mse': fg_mse,
                        'foreground_psnr_db': (-10 * math.log10(max(fg_mse, 1e-12)) if fg_mse is not None else None),
                        'foreground_fraction': float(foreground.mean()),
                        'mean_opacity': float(opacity.mean())})
    if args.gif_frames:
        imgs = []
        radius = float(torch.linalg.vector_norm(data['train']['poses'][0][:3, 3]))
        for i in range(args.gif_frames):
            pose = make_orbit_pose(2 * math.pi * i / args.gif_frames, math.radians(args.orbit_elevation), radius, device)
            prediction, _ = image_render(field, pose, h, w, focal, args)
            imgs.append(Image.fromarray(np.round(prediction * 255).astype(np.uint8)))
        imgs[0].save(out / 'orbit.gif', save_all=True, append_images=imgs[1:], duration=args.gif_duration,
                     loop=0, optimize=False, disposal=2)
    summary = {'method': 'feature-grid NeRF: learned density and feature grids plus directional RGB MLP',
               'source': str(Path(args.input)), 'train_frames': [f['file_path'] for f in train['frames']],
               'test_frames': metrics, 'average_psnr_db': float(np.mean([m['psnr_db'] for m in metrics])),
               'average_foreground_psnr_db': float(np.mean([m['foreground_psnr_db'] for m in metrics if m['foreground_psnr_db'] is not None])),
               'elapsed_train_seconds': seconds, 'peak_cuda_bytes': int(torch.cuda.max_memory_allocated()) if device.type == 'cuda' else None,
               'device': str(device), 'device_name': torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU',
               'torch_version': torch.__version__, 'cuda_version': torch.version.cuda,
               'config': vars(args), 'loss_history': losses,
               'note': 'Known synthetic camera poses. Held-out images are not used for optimization.'}
    (out / 'metrics.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    print(f'{out.name}: held-out PSNR {summary["average_psnr_db"]:.2f} dB; foreground {summary["average_foreground_psnr_db"]:.2f} dB', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True); parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cuda'); parser.add_argument('--seed', type=int, default=2503)
    parser.add_argument('--steps', type=int, default=1500); parser.add_argument('--rays', type=int, default=2048)
    parser.add_argument('--grid', type=int, default=64); parser.add_argument('--channels', type=int, default=8)
    parser.add_argument('--bound', type=float, default=.65); parser.add_argument('--samples', type=int, default=64)
    parser.add_argument('--near', type=float); parser.add_argument('--far', type=float)
    parser.add_argument('--grid-lr', type=float, default=.08); parser.add_argument('--density-lr', type=float, default=.12)
    parser.add_argument('--mlp-lr', type=float, default=.005); parser.add_argument('--train-views', type=int)
    parser.add_argument('--mask-weight', type=float, default=.02)
    parser.add_argument('--no-model', action='store_true', help='omit checkpoint for metric-only ablations')
    parser.add_argument('--chunk', type=int, default=2048); parser.add_argument('--gif-frames', type=int, default=48)
    parser.add_argument('--gif-duration', type=int, default=80); parser.add_argument('--orbit-elevation', type=float, default=20.)
    parser.add_argument('--log-every', type=int, default=100)
    args = parser.parse_args()
    if args.steps < 1 or args.samples < 1 or args.rays < 2:
        parser.error('steps/samples/rays must be positive')
    run(args)


if __name__ == '__main__':
    main()
