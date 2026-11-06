# HW2 implementation plan

Goal: produce ten real, calibrated image-to-NeRF reconstructions, point-matching evidence and GIFs from HW1 cleaned assets.

- Recover selected assets against HW1's recorded hashes and render calibrated synthetic RGB multiviews with disjoint held-out views. Record provenance, licensing, splits and camera convention.
- For each object, run SIFT, ratio matching and RANSAC; retain coordinates, masks, diagnostic figures and failure counts. Treat known synthetic poses as input, not SfM output.
- Train a compact neural radiance field from training pixels alone, ray-march and alpha-composite to render novel views, save checkpoints and held-out metrics. Test camera directions and compositing; visually inspect representative outputs.
- Run an explicit controlled experiment for at least one factor. Present results, caveats, reproducible commands, prompt record and exact software settings. Keep team fields and repository link as explicit pending information.
- Check all ten objects, matches, metrics, frames and GIFs; do not describe unrun work as results.
