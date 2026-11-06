# Data preparation

The ten objects come from the existing `course_project/research/course_asset_audit.json` examples: mug, apple, chair, table, bowl, bottle, banana, box, plate and cabinet. Each UID is required to be in HW1's `core` tier. The original source hash is checked before and after processing. The exact HW1 normalized GLB is reproduced using `hw1.normalize.normalize_glb`; its SHA-256 and normalization transform must match HW1's saved record. No source files are modified.

The original course-project audit also contains a cupboard, excluded here because it is similar to the cabinet. Some chosen objects have no texture; retaining them provides natural difficult cases for point matching. Selection is by the existing category examples, without consulting reconstruction scores.

Run from the project directory using the existing HW1 Python environment:

```powershell
& 'F:/dataset/ME2503/hw1_work/env/Scripts/python.exe' -B hw2/prepare_data.py
```

The local defaults can be changed with `--curation`, `--blender`, and `--resolution`. Use `--source-root /path/to/glbs` on another machine, where files are arranged as `shard/UID.glb`. The bundled `data/hw1_selection.json` contains the ten selected records, and `vendor/hw1_normalization/` contains exact copies of the two required HW1 implementation files with hashes; this avoids depending on the rest of the original course folder. Original GLBs must be obtained separately with their licenses respected. Use `--prepare-only` to verify/reproduce normalized models, `--render-only` to use the existing manifest, or `--verify-only` to check images and cameras. `--force-render` rerenders existing images. The scripts resume complete image sets; a partial object is rendered again.

Each `data/<category>/transforms.json` has 32 training views (eight azimuths at elevations −20°, 10°, 35°, 60°) and four held-out views at azimuths halfway between training cameras and different elevations. Cameras are perspective pinhole cameras, with a 50 mm focal length and 36 mm horizontal sensor. Their matrices are OpenGL/Blender camera-to-world matrices: +X right, +Y up, −Z forward. The scene uses Blender's Z-up convention after glTF import. Lens distortion is absent. Fixed illumination is shared across views.

RGB PNGs are composited onto neutral gray RGB(128,128,128); original alpha masks are saved separately. `background_rgb` stores this color in the [0,1] range. Gray was chosen after inspecting the white mug against a white background; the same background is used for all objects and splits. Source geometry is used only by Blender to create image supervision. The NeRF optimizer must consume RGB and calibrated cameras, not the source mesh, depth, or normals. Test images are held out of optimization.

`data/verification.json` records file hashes, coverage checks, image counts, orthonormal camera rotations, camera viewing directions, and train/test pose uniqueness. `data/manifest.json` records source lineage, normalization hashes, authors and licenses. `data/ATTRIBUTION.md` must accompany redistributed rendered images; the table example is CC BY-NC and intended for noncommercial coursework. The `.local_assets` directory contains temporary local normalized models and must be excluded from the submission/repository.
