"""Reproducible SIFT correspondences and two-view geometry for HW2.

This stage uses training RGB images only. Estimated matrices are obtained from
image correspondences; known Blender cameras are used only for a separate
geometric diagnostic, never as evidence that SfM recovered the input poses.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def image_path(root: Path, frame: dict) -> Path:
    path = root / frame["file_path"]
    if not path.suffix:
        path = path.with_suffix(".png")
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def intrinsics(meta: dict, frame: dict, width: int, height: int) -> np.ndarray | None:
    values = dict(meta)
    values.update(frame)
    if "fl_x" in values:
        fx = float(values["fl_x"])
        fy = float(values.get("fl_y", fx))
    elif "camera_angle_x" in values:
        fx = width / (2 * math.tan(float(values["camera_angle_x"]) / 2))
        fy = fx
    elif "focal_length_px" in values:
        fx = fy = float(values["focal_length_px"])
    elif "K" in values:
        return np.asarray(values["K"], dtype=np.float64)
    else:
        return None
    return np.array([[fx, 0, float(values.get("cx", width / 2))],
                     [0, fy, float(values.get("cy", height / 2))],
                     [0, 0, 1]], dtype=np.float64)


def skew(v: np.ndarray) -> np.ndarray:
    x, y, z = v
    return np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]], dtype=np.float64)


def calibrated_geometry(frame1: dict, frame2: dict, k1: np.ndarray,
                        k2: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return F, R_21 and t_21, converting OpenGL c2w to OpenCV axes."""
    axis_change = np.diag([1., -1., -1., 1.])
    camera1 = np.asarray(frame1["transform_matrix"], dtype=float) @ axis_change
    camera2 = np.asarray(frame2["transform_matrix"], dtype=float) @ axis_change
    relative = np.linalg.inv(camera2) @ camera1
    rotation, translation = relative[:3, :3], relative[:3, 3]
    fundamental = np.linalg.inv(k2).T @ skew(translation) @ rotation @ np.linalg.inv(k1)
    norm = np.linalg.norm(fundamental)
    if norm < 1e-12:
        raise ValueError("zero camera baseline")
    return fundamental / norm, rotation, translation


def sampson_errors(fundamental: np.ndarray, xy1: np.ndarray,
                   xy2: np.ndarray) -> np.ndarray:
    """Square-root Sampson error in pixels for x2.T @ F @ x1 = 0."""
    a = np.column_stack([xy1, np.ones(len(xy1))])
    b = np.column_stack([xy2, np.ones(len(xy2))])
    fa = a @ fundamental.T
    ftb = b @ fundamental
    numerator = np.abs(np.sum(b * fa, axis=1))
    denominator = np.sqrt(fa[:, 0] ** 2 + fa[:, 1] ** 2 + ftb[:, 0] ** 2 + ftb[:, 1] ** 2)
    return numerator / np.maximum(denominator, 1e-12)


def candidate_pairs(frames: list[dict], maximum: int) -> list[tuple[int, int, float]]:
    """Pick nearby viewpoints deterministically; only train frames are supplied."""
    candidates = {}
    for i, first in enumerate(frames):
        p = np.asarray(first["transform_matrix"], dtype=float)[:3, 3]
        neighbours = []
        for j, second in enumerate(frames):
            if i == j:
                continue
            q = np.asarray(second["transform_matrix"], dtype=float)[:3, 3]
            cosine = np.dot(p, q) / max(np.linalg.norm(p) * np.linalg.norm(q), 1e-12)
            angle = math.degrees(math.acos(float(np.clip(cosine, -1, 1))))
            if 2 <= angle <= 50:
                neighbours.append((angle, j))
        for angle, j in sorted(neighbours)[:2]:
            key = tuple(sorted((i, j)))
            candidates[key] = angle
    ordered = [(i, j, angle) for (i, j), angle in sorted(candidates.items())]
    if len(ordered) > maximum:
        indices = np.linspace(0, len(ordered) - 1, maximum).round().astype(int)
        ordered = [ordered[index] for index in indices]
    return ordered


def extract(path: Path, sift, mask_path: Path | None = None) -> dict:
    with Image.open(path) as original:
        rgba = np.array(original.convert("RGBA"))
    alpha = rgba[:, :, 3:4].astype(np.float32) / 255
    rgb = np.round(rgba[:, :, :3] * alpha + 255 * (1 - alpha)).astype(np.uint8)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    # Exclude background and alpha-boundary features; all coordinates remain in
    # the original image pixel coordinate system, without any image resizing.
    if mask_path is not None:
        with Image.open(mask_path) as source_mask:
            alpha_mask = np.array(source_mask.convert("L"))
        if alpha_mask.shape != gray.shape:
            raise ValueError(f"Mask/image size mismatch: {mask_path}")
    else:
        alpha_mask = rgba[:, :, 3]
    mask = (alpha_mask >= 128).astype(np.uint8) * 255
    mask = cv2.erode(mask, np.ones((3, 3), np.uint8))
    points, descriptors = sift.detectAndCompute(gray, mask)
    return {"rgb": rgb, "points": points, "descriptors": descriptors,
            "width": rgb.shape[1], "height": rgb.shape[0]}


def mutual_ratio_matches(descriptors1, descriptors2, ratio: float):
    if descriptors1 is None or descriptors2 is None or min(len(descriptors1), len(descriptors2)) < 2:
        return []
    norm = cv2.NORM_HAMMING if descriptors1.dtype == np.uint8 else cv2.NORM_L2
    matcher = cv2.BFMatcher(norm)
    forward, backward = matcher.knnMatch(descriptors1, descriptors2, k=2), matcher.knnMatch(descriptors2, descriptors1, k=2)
    reverse = {p[0].queryIdx: p[0].trainIdx for p in backward
               if len(p) == 2 and p[0].distance < ratio * p[1].distance}
    result = []
    for pair in forward:
        if len(pair) != 2:
            continue
        match, second = pair
        if match.distance < ratio * second.distance and reverse.get(match.trainIdx) == match.queryIdx:
            result.append((match.queryIdx, match.trainIdx, float(match.distance),
                           float(match.distance / max(second.distance, 1e-12))))
    return sorted(result, key=lambda item: (item[2], item[0], item[1]))


def match_pair(first: dict, second: dict, frame1: dict, frame2: dict,
               meta: dict, ratio: float, threshold: float, seed: int) -> dict:
    matches = mutual_ratio_matches(first["descriptors"], second["descriptors"], ratio)
    xy1 = np.asarray([first["points"][m[0]].pt for m in matches], dtype=np.float64).reshape(-1, 2)
    xy2 = np.asarray([second["points"][m[1]].pt for m in matches], dtype=np.float64).reshape(-1, 2)
    result = {"image_1": frame1["file_path"], "image_2": frame2["file_path"],
              "split": "train", "keypoints_1": len(first["points"]), "keypoints_2": len(second["points"]),
              "ratio_mutual_count": len(matches), "ransac_inliers": 0,
              "status": "insufficient_correspondences", "fundamental_matrix": None,
              "estimated_essential_matrix": None, "essential_inliers": 0,
              "ratio_threshold": ratio, "ransac_threshold_pixels": threshold,
              "ransac_confidence": 0.999, "ransac_max_iterations": 10000,
              "coordinate_convention": "original-image pixels, x right and y down, x2.T F x1 = 0",
              "correspondences": []}
    fundamental = None
    inliers = np.zeros(len(matches), dtype=bool)
    estimated_errors = np.full(len(matches), np.nan)
    if len(matches) >= 8:
        cv2.setRNGSeed(seed)
        fundamental, mask = cv2.findFundamentalMat(xy1, xy2, cv2.FM_RANSAC, threshold, 0.999, 10000)
        if fundamental is not None and fundamental.shape == (3, 3) and np.isfinite(fundamental).all() and mask is not None:
            inliers = mask.ravel().astype(bool)
            estimated_errors = sampson_errors(fundamental, xy1, xy2)
            result["fundamental_matrix"] = fundamental.tolist()
            result["ransac_inliers"] = int(inliers.sum())
            result["status"] = "estimated" if inliers.sum() >= 8 else "insufficient_inliers"
        else:
            result["status"] = "fundamental_estimation_failed"
    result["inlier_ratio"] = float(inliers.mean()) if len(matches) else 0.0
    k1 = intrinsics(meta, frame1, first["width"], first["height"])
    k2 = intrinsics(meta, frame2, second["width"], second["height"])
    truth_errors = np.full(len(matches), np.nan)
    essential_mask = np.zeros(len(matches), dtype=bool)
    if k1 is not None and k2 is not None:
        truth_f, truth_r, truth_t = calibrated_geometry(frame1, frame2, k1, k2)
        truth_errors = sampson_errors(truth_f, xy1, xy2)
        result["calibrated_diagnostic"] = {
            "note": "Known Blender camera diagnostic only; does not establish estimated SfM poses.",
            "K_1": k1.tolist(), "K_2": k2.tolist(), "fundamental_matrix": truth_f.tolist(),
            "median_sampson_pixels_all_matches": float(np.median(truth_errors)) if len(matches) else None,
            "median_sampson_pixels_ransac_inliers": float(np.median(truth_errors[inliers])) if inliers.any() else None,
            "within_threshold_all_matches": int((truth_errors <= threshold).sum()),
            "within_threshold_ransac_inliers": int(((truth_errors <= threshold) & inliers).sum())}
        if len(matches) >= 8 and np.allclose(k1, k2):
            cv2.setRNGSeed(seed)
            essential, mask_e = cv2.findEssentialMat(xy1, xy2, k1, method=cv2.RANSAC,
                                                    prob=0.999, threshold=threshold, maxIters=10000)
            if essential is not None and essential.shape == (3, 3) and mask_e is not None:
                essential_mask = mask_e.ravel().astype(bool)
                result["estimated_essential_matrix"] = essential.tolist()
                result["essential_inliers"] = int(essential_mask.sum())
                count, estimated_r, estimated_t, pose_mask = cv2.recoverPose(essential, xy1, xy2, k1, mask=mask_e.copy())
                angle = math.degrees(math.acos(float(np.clip((np.trace(estimated_r @ truth_r.T) - 1) / 2, -1, 1))))
                cosine = float(np.dot(estimated_t.ravel(), truth_t) / max(np.linalg.norm(truth_t), 1e-12))
                result["relative_pose_diagnostic"] = {
                    "rotation": estimated_r.tolist(), "translation_unit": estimated_t.ravel().tolist(),
                    "cheirality_inliers": int(count), "rotation_error_degrees": angle,
                    "translation_direction_error_degrees": math.degrees(math.acos(float(np.clip(cosine, -1, 1)))),
                    "note": "Two-view estimate only, susceptible to planar/low-parallax degeneracy; not bundle-adjusted SfM."}
    for index, match in enumerate(matches):
        result["correspondences"].append({
            "keypoint_1_index": match[0], "keypoint_2_index": match[1],
            "xy_1": xy1[index].tolist(), "xy_2": xy2[index].tolist(),
            "descriptor_distance": match[2], "forward_lowe_ratio": match[3],
            "fundamental_ransac_inlier": bool(inliers[index]),
            "essential_ransac_inlier": bool(essential_mask[index]),
            "estimated_sampson_pixels": float(estimated_errors[index]) if np.isfinite(estimated_errors[index]) else None,
            "calibrated_sampson_pixels": float(truth_errors[index]) if np.isfinite(truth_errors[index]) else None})
    return result


def draw_matches(first: dict, second: dict, result: dict, path: Path) -> None:
    scale = max(1, int(math.ceil(400 / max(first["width"], second["width"]))))
    left = Image.fromarray(first["rgb"]).resize((first["width"] * scale, first["height"] * scale))
    right = Image.fromarray(second["rgb"]).resize((second["width"] * scale, second["height"] * scale))
    canvas = Image.new("RGB", (left.width + right.width, max(left.height, right.height) + 72), "white")
    canvas.paste(left, (0, 72)); canvas.paste(right, (left.width, 72))
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 8), f"{result.get('detector', 'sift').upper()} + mutual Lowe {result['ratio_threshold']} + F-RANSAC", fill="black")
    draw.text((10, 27), f"{result['ratio_mutual_count']} matches; {result['ransac_inliers']} F inliers; {result['status']}", fill="black")
    draw.text((10, 46), "Green: estimated F inlier. Red: rejected. At most 80 correspondences shown.", fill="black")
    for match in result["correspondences"][:80]:
        a = (match["xy_1"][0] * scale, match["xy_1"][1] * scale + 72)
        b = (match["xy_2"][0] * scale + left.width, match["xy_2"][1] * scale + 72)
        color = (0, 145, 80) if match["fundamental_ransac_inlier"] else (205, 50, 50)
        draw.line([a, b], fill=color, width=1)
        for x, y in (a, b):
            draw.ellipse((x - 2, y - 2, x + 2, y + 2), outline=color, width=2)
    canvas.save(path)


def run_object(root: Path, output: Path, args) -> dict:
    meta = json.loads((root / "transforms.json").read_text(encoding="utf-8"))
    frames = [f for f in meta["frames"] if f.get("split") == "train"]
    if len(frames) < 2:
        raise ValueError(f"{root.name}: need at least two explicitly marked training frames")
    output.mkdir(parents=True, exist_ok=True)
    if args.detector == "sift":
        sift = cv2.SIFT_create(nfeatures=4000, contrastThreshold=0.01, edgeThreshold=12)
        detector_parameters = {"nfeatures": 4000, "contrastThreshold": 0.01, "edgeThreshold": 12}
    elif args.detector == "akaze":
        sift = cv2.AKAZE_create(threshold=0.0005)
        detector_parameters = {"threshold": 0.0005}
    else:
        sift = cv2.ORB_create(nfeatures=3000, edgeThreshold=8, patchSize=21, fastThreshold=7)
        detector_parameters = {"nfeatures": 3000, "edgeThreshold": 8, "patchSize": 21, "fastThreshold": 7}
    cache, summaries = {}, []
    best, best_pair = None, None
    pairs = candidate_pairs(frames, args.max_pairs)
    for pair_index, (i, j, angle) in enumerate(pairs):
        for index in (i, j):
            if index not in cache:
                mask_path = root / frames[index]["mask_path"] if frames[index].get("mask_path") else None
                cache[index] = extract(image_path(root, frames[index]), sift, mask_path)
        result = match_pair(cache[i], cache[j], frames[i], frames[j], meta,
                            args.ratio, args.threshold, args.seed + pair_index)
        name = f"pair_{i:03d}_{j:03d}"
        result.update(pair=name, camera_baseline_angle_degrees=angle, detector=args.detector,
                      descriptor_distance_metric="L2" if args.detector == "sift" else "Hamming")
        write_json(output / f"{name}.json", result)
        brief = {k: v for k, v in result.items() if k != "correspondences"}
        brief["correspondences_file"] = f"{name}.json"
        summaries.append(brief)
        if best is None or (result["ransac_inliers"], result["ratio_mutual_count"]) > (best["ransac_inliers"], best["ratio_mutual_count"]):
            best, best_pair = result, (i, j)
    if best is not None:
        draw_matches(cache[best_pair[0]], cache[best_pair[1]], best, output / "best_matches.png")
        write_json(output / "best_matches.json", best)
    for index, feature in cache.items():
        points = feature["points"]
        np.savez_compressed(output / f"features_{index:03d}.npz",
                            xy=np.asarray([p.pt for p in points], dtype=np.float32).reshape(-1, 2),
                            size=np.asarray([p.size for p in points]), angle=np.asarray([p.angle for p in points]),
                            response=np.asarray([p.response for p in points]),
                            descriptors=feature["descriptors"] if feature["descriptors"] is not None else np.empty((0, 128), dtype=np.float32))
    summary = {"object": root.name, "status": "estimated" if best and best["status"] == "estimated" else "matching_failed",
               "method": f"OpenCV {args.detector.upper()}, bidirectional mutual Lowe ratio, fundamental/essential RANSAC",
               "opencv_version": cv2.__version__, "seed": args.seed, "train_frame_count": len(frames),
               "tested_pair_count": len(pairs), "successful_pair_count": sum(p["status"] == "estimated" for p in summaries),
               "best_pair": best["pair"] if best else None,
               "best_inlier_count": best["ransac_inliers"] if best else 0,
               "best_match_count": best["ratio_mutual_count"] if best else 0,
               "feature_image_map": {f"features_{i:03d}.npz": frames[i]["file_path"] for i in cache},
               "input_sha256": {frames[i]["file_path"]: hashlib.sha256(image_path(root, frames[i]).read_bytes()).hexdigest() for i in cache},
               "transforms_sha256": hashlib.sha256((root / "transforms.json").read_bytes()).hexdigest(),
               "best_selection": "largest F-RANSAC inlier count, then largest mutual ratio match count",
               "caveat": "Pairwise feature matching and pose diagnostics are not an end-to-end SfM reconstruction. NeRF uses calibrated synthetic cameras.",
               "parameters": {"ratio": args.ratio, "threshold_pixels": args.threshold,
                              "detector": args.detector, "detector_parameters": detector_parameters,
                              "alpha_mask_threshold": 128,
                              "alpha_mask_erosion_kernel": [3, 3]},
               "pairs": summaries}
    write_json(output / "summary.json", summary)
    return summary


def main() -> None:
    base = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=base / "data")
    parser.add_argument("--output", type=Path, default=base / "matches")
    parser.add_argument("--objects", nargs="*", help="Optional object slugs; default: all transforms.json folders")
    parser.add_argument("--max-pairs", type=int, default=24)
    parser.add_argument("--ratio", type=float, default=0.80)
    parser.add_argument("--threshold", type=float, default=1.5)
    parser.add_argument("--seed", type=int, default=2503)
    parser.add_argument("--detector", choices=("sift", "akaze", "orb"), default="sift")
    args = parser.parse_args()
    if args.max_pairs <= 0 or not 0 < args.ratio < 1 or args.threshold <= 0:
        parser.error("max-pairs and threshold must be positive; ratio must be in (0, 1)")
    roots = [args.data / slug for slug in args.objects] if args.objects else sorted(p.parent for p in args.data.glob("*/transforms.json"))
    if not roots:
        parser.error("No input transforms.json found")
    cv2.setNumThreads(4)
    summaries = []
    for root in roots:
        summary = run_object(root, args.output / root.name, args)
        summaries.append({k: v for k, v in summary.items() if k not in ("pairs", "feature_image_map")})
        print(f"{root.name}: {summary['status']}, best {summary['best_inlier_count']}/{summary['best_match_count']} inliers, {summary['successful_pair_count']}/{summary['tested_pair_count']} pairs", flush=True)
    write_json(args.output / "summary.json", {"objects": summaries})
    with (args.output / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["object", "status", "best_pair", "best_inlier_count", "best_match_count", "successful_pair_count", "tested_pair_count"])
        writer.writeheader()
        for row in summaries:
            writer.writerow({key: row[key] for key in writer.fieldnames})


if __name__ == "__main__":
    main()
