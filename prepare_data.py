"""Reproduce ten HW1-cleaned assets, render calibrated images, and verify them.

Source GLBs are read only. Local normalized GLBs stay in .local_assets and are
not part of the submission. Included, unchanged HW1 normalization modules
reproduce the historical bytes; image datasets need no original GLB files.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
CURATION = Path("F:/dataset/ME2503/hw1_work/curation")
BLENDER = Path("F:/dataset/ME2503/hw1_work/tools/blender-4.5.14-windows-x64/blender.exe")
CATEGORIES = ["mug", "apple", "chair", "table", "bowl", "bottle", "banana", "box", "plate", "cabinet"]
BACKGROUND_RGB = (128, 128, 128)
LICENSE_URLS = {
    "by": "https://creativecommons.org/licenses/by/4.0/",
    "by-nc": "https://creativecommons.org/licenses/by-nc/4.0/",
    "cc0": "https://creativecommons.org/publicdomain/zero/1.0/",
}


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def selected_records(curation):
    audit_path = PROJECT / "course_project/research/course_asset_audit.json"
    snapshot_path = ROOT / "data/hw1_selection.json"
    if not audit_path.is_file() or not (curation / "objects_index.jsonl").is_file():
        snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
        return [(item["category"], item["index_record"], item["tier_record"]) for item in snapshot["selections"]]
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    examples = {row["category"]: row for row in audit["examples"]}
    uids = {examples[category]["uid"] for category in CATEGORIES}
    tiers, rows = {}, {}
    for filename, target in (("tiers.jsonl", tiers), ("objects_index.jsonl", rows)):
        with (curation / filename).open(encoding="utf-8-sig") as stream:
            for line in stream:
                row = json.loads(line)
                if row["uid"] in uids:
                    target[row["uid"]] = row
    if set(tiers) != uids or set(rows) != uids:
        raise ValueError("All ten selected objects must occur in both HW1 evidence files")
    selected = [(category, rows[examples[category]["uid"]], tiers[examples[category]["uid"]]) for category in CATEGORIES]
    selections = []
    for category, row, tier in selected:
        compact = {key: row[key] for key in ("uid", "shard", "path", "sha256", "geometry", "normalization")}
        compact["metadata"] = {key: row["metadata"].get(key) for key in ("name", "license", "viewerUrl", "user")}
        compact["metadata"]["user"] = {key: row["metadata"].get("user", {}).get(key, "") for key in ("displayName", "profileUrl", "username")}
        selections.append({"category": category, "index_record": compact, "tier_record": tier})
    write_json(snapshot_path, {"description": "Unchanged relevant fields from preserved HW1 index and tier records", "selections": selections})
    return selected


def prepare(curation, source_root=None):
    sys.path.insert(0, str(ROOT))
    from vendor.hw1_normalization.normalize import normalize_glb
    records = []
    for category, row, tier in selected_records(curation):
        if tier["tier"] != "core":
            raise ValueError(f"{row['uid']} is not an HW1 core object")
        source = Path(row["path"])
        if source_root is not None:
            source = source_root / row["shard"] / f"{row['uid']}.glb"
        if digest(source) != row["sha256"]:
            raise ValueError(f"HW1 source SHA-256 mismatch: {source}")
        normalization = row["normalization"]
        if normalization["status"] != "applied":
            raise ValueError("Selected asset lacks an applied HW1 normalization")
        destination = ROOT / ".local_assets" / f"{row['uid']}.glb"
        if not destination.exists():
            result = normalize_glb(source, destination, row["geometry"])
            if result["status"] != "applied" or result["transform"] != normalization["transform"]:
                raise ValueError(f"Historical normalization was not reproduced: {result}")
        if digest(destination) != normalization["sha256"]:
            raise ValueError(f"Normalized SHA-256 differs from HW1: {destination}")
        if digest(source) != row["sha256"]:
            raise ValueError("Source changed during normalization")
        metadata = row["metadata"]
        license_code = metadata.get("license")
        if license_code not in LICENSE_URLS:
            raise ValueError(f"Unsupported license: {license_code}")
        records.append({
            "slug": category, "uid": row["uid"], "name": metadata["name"],
            "category": category, "hw1_tier": tier["tier"],
            "source_path": str(source), "source_sha256": row["sha256"],
            "normalized_path": destination.relative_to(ROOT).as_posix(),
            "normalized_sha256": normalization["sha256"],
            "normalization": normalization["transform"],
            "normalization_verification": "exact SHA-256 match to recorded HW1 normalized GLB",
            "triangles": row["geometry"]["metrics"]["triangles"],
            "textures": row["geometry"]["features"]["textures"],
            "license": license_code, "license_url": LICENSE_URLS[license_code],
            "author": metadata.get("user", {}).get("displayName", ""),
            "author_url": metadata.get("user", {}).get("profileUrl", ""),
            "source_url": metadata.get("viewerUrl", ""),
            "modifications": "HW1 center/unit-box normalization; synthetic camera views with fixed illumination; neutral gray compositing",
        })
        print(f"Verified HW1 normalized asset: {category} / {row['uid']}", flush=True)
    manifest = {
        "schema_version": 1,
        "selection": "Ten distinct categories from existing course_project audit; cupboard excluded because it is similar to cabinet.",
        "hw1_evidence": {str(path.name): {"path": str(path), "sha256": digest(path)} for path in
                         (curation / "tiers.jsonl", curation / "objects_index.jsonl") if path.is_file()},
        "selection_snapshot": {"path": "data/hw1_selection.json", "sha256": digest(ROOT / "data/hw1_selection.json")},
        "source_assets_modified": False,
        "source_mesh_usage": "Only to synthesize RGB supervision. Mesh vertices/depth/normals are not exported to the NeRF trainer.",
        "redistribution": "Source and normalized GLBs are excluded from submission. Rendered derivatives retain author/license information; table is CC BY-NC and is for noncommercial coursework.",
        "objects": records,
    }
    write_json(ROOT / "data/manifest.json", manifest)
    credits = ["# Dataset image credits", "", "These images are synthetic views derived from the following Objaverse / Sketchfab assets. The original mesh data is excluded from the submission. Source links and attribution must accompany redistributed images. All images have been rendered with new camera views and fixed illumination, normalized to a unit box, and composited onto neutral gray RGB(128,128,128).", "", "| Object | Original title | Author | License | Original source |", "|---|---|---|---|---|"]
    for row in records:
        credits.append(f"| {row['slug']} | {row['name']} | [{row['author']}]({row['author_url']}) | [{row['license']}]({row['license_url']}) | [Sketchfab]({row['source_url']}) |")
    credits.extend(["", "The table images are CC BY-NC 4.0: their permitted reuse is noncommercial. Other image sets are CC BY 4.0. This attribution is based on the preserved HW1 source metadata; it is not a new license for the original assets.", ""])
    (ROOT / "data/ATTRIBUTION.md").write_text("\n".join(credits), encoding="utf-8")


def composite_and_verify():
    import numpy as np
    from PIL import Image
    manifest = json.loads((ROOT / "data/manifest.json").read_text(encoding="utf-8"))
    checks = []
    for record in manifest["objects"]:
        directory = ROOT / "data" / record["slug"]
        path = directory / "transforms.json"
        transform = json.loads(path.read_text(encoding="utf-8"))
        previous_background = np.asarray(transform.get("background_rgb", [1., 1., 1.])) * 255
        target_background = np.asarray(BACKGROUND_RGB, dtype=float)
        assert len(transform["frames"]) == 36
        assert sum(f["split"] == "train" for f in transform["frames"]) == 32
        assert sum(f["split"] == "test" for f in transform["frames"]) == 4
        seen = set()
        frame_checks = []
        for frame in transform["frames"]:
            matrix = np.asarray(frame["transform_matrix"], dtype=float)
            assert matrix.shape == (4, 4) and np.isfinite(matrix).all()
            assert np.allclose(matrix[3], [0, 0, 0, 1])
            rotation = matrix[:3, :3]
            assert np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5)
            assert np.linalg.det(rotation) > 0.999
            position = matrix[:3, 3]
            forward = -rotation[:, 2]
            assert np.dot(forward, -position) / np.linalg.norm(position) > 0.999
            pose = tuple(np.round(matrix, 6).flatten())
            assert pose not in seen, "Train/test camera duplication"
            seen.add(pose)
            image_path = directory / frame["file_path"]
            with Image.open(image_path) as image:
                assert image.size == (transform["width"], transform["height"])
                if image.mode == "RGBA":
                    alpha = image.getchannel("A")
                    mask_path = directory / frame["mask_path"]
                    mask_path.parent.mkdir(parents=True, exist_ok=True)
                    alpha.save(mask_path)
                    composite = Image.new("RGB", image.size, BACKGROUND_RGB)
                    composite.paste(image, mask=alpha)
                    composite.save(image_path)
                elif not np.allclose(previous_background, target_background):
                    # Alpha compositing is linear in the chosen background:
                    # C_new = C_old + (B_new - B_old) * (1 - alpha).
                    with Image.open(directory / frame["mask_path"]) as alpha_image:
                        alpha = np.asarray(alpha_image, dtype=float)[..., None] / 255
                    rgb = np.asarray(image, dtype=float)
                    updated = np.clip(np.rint(rgb + (target_background - previous_background) * (1 - alpha)), 0, 255).astype(np.uint8)
                    Image.fromarray(updated).save(image_path)
            with Image.open(directory / frame["mask_path"]) as mask:
                foreground = np.asarray(mask) > 127
                coverage = float(foreground.mean())
                assert 0.005 < coverage < 0.95, f"Suspicious foreground coverage {coverage}: {image_path}"
                edge_pixels = int(foreground[0].sum() + foreground[-1].sum() + foreground[:, 0].sum() + foreground[:, -1].sum())
                assert edge_pixels == 0, f"Foreground clipped by image border: {image_path}"
            frame_checks.append({"path": frame["file_path"], "sha256": digest(image_path), "foreground_fraction": coverage, "foreground_border_pixels": edge_pixels})
        checks.append({"slug": record["slug"], "train_frames": 32, "test_frames": 4, "unique_camera_poses": len(seen), "frames": frame_checks})
        transform["background_rgb"] = [value / 255 for value in BACKGROUND_RGB]
        transform["background"] = "neutral gray RGB(128,128,128), with separate alpha masks"
        write_json(path, transform)
        print(f"Verified 36 images and distinct poses: {record['slug']}", flush=True)
    write_json(ROOT / "data/verification.json", {"status": "passed", "object_count": len(checks), "image_count": 36 * len(checks), "objects": checks})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--curation", type=Path, default=CURATION)
    parser.add_argument("--blender", type=Path, default=BLENDER)
    parser.add_argument("--resolution", type=int, default=128)
    parser.add_argument("--source-root", type=Path, help="Alternative original GLB root containing shard/UID.glb")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--render-only", action="store_true")
    parser.add_argument("--force-render", action="store_true")
    args = parser.parse_args()
    if args.verify_only:
        composite_and_verify()
        return
    if not args.render_only:
        prepare(args.curation, args.source_root)
    if args.prepare_only:
        return
    log_path = ROOT / "data/render.log"
    command = [str(args.blender), "--background", "--factory-startup", "--python", str(ROOT / "render_views_blender.py"), "--", "--manifest", str(ROOT / "data/manifest.json"), "--resolution", str(args.resolution)]
    if args.force_render:
        command.append("--force")
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    with log_path.open("w", encoding="utf-8") as output:
        subprocess.run(command, check=True, stdout=output, stderr=subprocess.STDOUT, cwd=ROOT, env=environment)
    composite_and_verify()


if __name__ == "__main__":
    main()
