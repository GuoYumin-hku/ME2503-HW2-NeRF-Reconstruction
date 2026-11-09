# HW2: 3D Reconstruction and Rendering

This repository contains the completed synthetic multi-view reconstruction pipeline for ten HW1 assets, including ten result folders with metrics, checkpoints, held-out comparisons, and 48-frame orbit GIFs. Read [ASSIGNMENT.md](ASSIGNMENT.md) for the course brief, [REPORT.md](REPORT.md) for the method, results, analysis, limitations, and team metadata TODOs, and [DATA_PREPARATION.md](DATA_PREPARATION.md) before regenerating data; it documents source assets, normalization, camera setup, and attribution.

## Contents

- `prepare_data.py`, `render_views_blender.py`: reproduce normalized assets and render/verify 32 training plus four held-out views per object.
- `match_features.py`: SIFT correspondence and two-view geometry diagnostics for all ten objects. Results are in `matches/summary.csv`, per-object JSON, and match visualizations.
- `nerf.py`: train a compact feature-grid radiance field, evaluate held-out views, save a checkpoint and orbit GIF.
- `data/<object>/`: generated images, masks, and camera metadata.
- `results/<object>/`: per-run metrics, held-out reference/prediction images, model checkpoint, and orbit GIF.
- `data/ATTRIBUTION.md`: required source image credits and license notes.

## Environment and data

The original source GLBs are not included and must be obtained separately under their stated licenses. Run data preparation with the existing HW1 environment and Blender setup described in `DATA_PREPARATION.md`. On this Windows machine, use the bundled Python 3.13 environment for matching and NeRF. If `hw2` is the repository root, use these commands; if running from its parent directory, add the `hw2/` prefix to script and data paths.

```powershell

# From the parent directory (E:\HKU-PhD-courses\2503):
& 'F:/dataset/ME2503/hw1_work/env/Scripts/python.exe' -B hw2/prepare_data.py
& 'E:/ComfyUI_windows_portable/python_embeded/python.exe' -B hw2/match_features.py

# If hw2 is the current repository root, remove the `hw2/` prefixes.
```

Preparation verifies source hashes, normalization, image coverage, masks, and camera poses. It writes `data/verification.json`; the expected dataset is 10 objects × 36 images = 360 images.

### Clean Python 3.13 environment (portable setup)

For a fresh virtual environment, install the pinned packages below. Install the CUDA 12.8 PyTorch wheels from the official PyTorch index:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install torch==2.9.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install numpy==2.2.6 opencv-python-headless==4.12.0.88 Pillow==12.0.0 imageio==2.37.2 matplotlib==3.10.7
```

## Train NeRF

Run one object at a time. The mug command below reproduces the production run configuration (10,000 steps):

```powershell
& 'E:/ComfyUI_windows_portable/python_embeded/python.exe' -B hw2/nerf.py --input hw2/data/mug --output hw2/results/mug --steps 10000 --rays 2048 --grid 64 --channels 8 --samples 64 --seed 2503
```

For another object, replace `mug` in both paths with its category slug. If `hw2` is the current repository root, remove the `hw2/` prefixes. The script writes `metrics.json`, held-out predictions/references, `model.pt`, and `orbit.gif`. Each metric file records the exact config, hardware/software, runtime, and held-out full-image and foreground PSNR.

## Full reproduction and handoff checks

Use the HW1 Python environment only for asset preparation and Blender rendering. The embedded Python 3.13 environment has the CUDA-enabled PyTorch and OpenCV packages for matching and NeRF. From the parent directory, train all objects, regenerate the aggregate CSV and visual overview, and verify the local handoff:

```powershell
& 'E:/ComfyUI_windows_portable/python_embeded/python.exe' -B hw2/run_all.py --steps 10000 --rays 2048 --seed 2503
& 'E:/ComfyUI_windows_portable/python_embeded/python.exe' -B hw2/summarize_results.py
& 'E:/ComfyUI_windows_portable/python_embeded/python.exe' -B hw2/verify_submission.py
```

For the baseline feature matches and post-hoc selected-pair QC, run:

```powershell
& 'E:/ComfyUI_windows_portable/python_embeded/python.exe' -B hw2/match_features.py
& 'E:/ComfyUI_windows_portable/python_embeded/python.exe' -B hw2/match_features.py --output hw2/matches/experiments/sift_090 --ratio 0.90
& 'E:/ComfyUI_windows_portable/python_embeded/python.exe' -B hw2/match_features.py --output hw2/matches/experiments/orb_090 --detector orb --ratio 0.90
& 'E:/ComfyUI_windows_portable/python_embeded/python.exe' -B hw2/match_features.py --output hw2/matches/experiments/akaze_090 --detector akaze --ratio 0.90
& 'E:/ComfyUI_windows_portable/python_embeded/python.exe' -B hw2/match_features.py --output hw2/matches/experiments/sift_090_100 --ratio 0.90 --max-pairs 100 --objects mug banana table
& 'E:/ComfyUI_windows_portable/python_embeded/python.exe' -B hw2/match_features.py --output hw2/matches/experiments/orb_090_100 --detector orb --ratio 0.90 --max-pairs 100 --objects mug banana table
& 'E:/ComfyUI_windows_portable/python_embeded/python.exe' -B hw2/select_matches.py
```

`run_all.py` supports `--skip-existing`; use `--sparse-views 8 --skip-existing` to add the eight-view ablation to existing full runs. The `--max-pairs 100` retries actually evaluate 37 distinct camera pairs for each specified object. `verify_submission.py` checks the ten image sets, match artifacts, metrics, checkpoints, held-out images, and 48-frame GIFs without original GLBs or a GPU. If `hw2` is the repository root, remove the `hw2/` prefixes from paths and command arguments.

## Team submission fields

- Group number: **TODO**
- Member names and contributions: **TODO**
- GitHub repository URL: [ME2503-HW2-NeRF-Reconstruction](https://github.com/GuoYumin-hku/ME2503-HW2-NeRF-Reconstruction)

Before submission, fill the group number and member names/contributions in `REPORT.md`; include all ten result folders and the prompts in [PROMPTS.md](PROMPTS.md). Exclude `hw2/.local_assets/`, a temporary local cache; do not redistribute source GLBs. Keep `data/ATTRIBUTION.md` with any generated images.
