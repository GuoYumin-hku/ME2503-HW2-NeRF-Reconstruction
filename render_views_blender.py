"""Run inside Blender. Render 32 training and 4 held-out perspective views."""
import argparse
import json
import math
from pathlib import Path
import sys
import time

import bpy


def remove_scene():
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    for collection in (bpy.data.meshes, bpy.data.materials, bpy.data.images, bpy.data.cameras, bpy.data.lights):
        for item in list(collection):
            if item.users == 0:
                collection.remove(item)


def orbit():
    cameras = []
    for ring, elevation in enumerate((-20, 10, 35, 60)):
        for index in range(8):
            cameras.append(("train", ring * 8 + index, index * 45, elevation))
    for index, (azimuth, elevation) in enumerate(((22.5, 0), (112.5, 22.5), (202.5, 47.5), (292.5, 20))):
        cameras.append(("test", index, azimuth, elevation))
    return cameras


def render_object(root, record, resolution, force):
    output = root / "data" / record["slug"]
    transforms_path = output / "transforms.json"
    radius = 2.8 if record["slug"] in ("box", "table") else 1.9
    if transforms_path.is_file() and not force:
        old = json.loads(transforms_path.read_text(encoding="utf-8"))
        old_frames = old.get("frames", [])
        old_radius = math.sqrt(sum(old_frames[0]["transform_matrix"][axis][3] ** 2 for axis in range(3))) if old_frames else 0
        if old.get("width") == resolution and abs(old_radius - radius) < 0.001 and len(old_frames) == 36 and all((output / frame["file_path"]).is_file() for frame in old_frames):
            print(f"HW2 reuse completed views: {record['slug']}", flush=True)
            return
    remove_scene()
    source = root / record["normalized_path"]
    bpy.ops.import_scene.gltf(filepath=str(source), import_pack_images=True)
    scene = bpy.context.scene
    scene.frame_set(0)
    for obj in list(scene.objects):
        if obj.type in ("CAMERA", "LIGHT"):
            bpy.data.objects.remove(obj, do_unlink=True)
    # The imported GLB already contains the exact HW1 unit-box transform.
    # Blender converts the glTF Y-up frame to Blender Z-up on import.
    bpy.context.view_layer.update()
    vertices = [obj.matrix_world @ vertex.co for obj in scene.objects if obj.type == "MESH" for vertex in obj.data.vertices]
    if not vertices:
        raise RuntimeError("Imported asset contains no mesh vertices")
    low = [min(v[axis] for v in vertices) for axis in range(3)]
    high = [max(v[axis] for v in vertices) for axis in range(3)]
    if abs(max(high[i] - low[i] for i in range(3)) - 1) > 0.001 or max(abs((low[i] + high[i]) / 2) for i in range(3)) > 0.001:
        raise RuntimeError(f"Imported normalized bounds invalid: {low} {high}")

    scene.render.engine = "CYCLES"
    scene.cycles.samples = 16
    scene.cycles.use_denoising = True
    scene.cycles.max_bounces = 4
    # Cycles/OptiX preserves PBR materials and works headlessly on this machine.
    backend = "CPU"
    preferences = bpy.context.preferences.addons['cycles'].preferences
    for candidate in ("OPTIX", "CUDA"):
        try:
            preferences.compute_device_type = candidate
            preferences.refresh_devices()
            devices = [device for device in preferences.devices if device.type == candidate]
            if devices:
                for device in preferences.devices:
                    device.use = device.type == candidate
                scene.cycles.device = "GPU"
                backend = candidate
                break
        except (TypeError, ValueError, RuntimeError):
            pass
    scene.render.resolution_x = scene.render.resolution_y = resolution
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.film_transparent = True
    scene.render.use_compositing = False
    scene.render.use_sequencer = False
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "Medium High Contrast"
    scene.view_settings.exposure = 0
    scene.view_settings.gamma = 1
    world = bpy.data.worlds.new("HW2 fixed world")
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.8, 0.8, 0.8, 1)
    world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.7
    scene.world = world
    for index, (location, energy, size) in enumerate((((2.5, -3.5, 4), 300, 4), ((-3, 1.5, 2), 180, 4), ((0, 0, -3), 75, 3))):
        light = bpy.data.lights.new(f"HW2 area {index}", type="AREA")
        light.energy, light.size = energy, size
        obj = bpy.data.objects.new(light.name, light)
        scene.collection.objects.link(obj)
        obj.location = location
        obj.rotation_euler = (-obj.location).to_track_quat("-Z", "Y").to_euler()
    camera = bpy.data.cameras.new("HW2 perspective")
    camera.type = "PERSP"
    camera.lens = 50
    camera.sensor_width = 36
    camera.sensor_fit = "HORIZONTAL"
    camera.clip_start, camera.clip_end = 0.05, 20
    camera_obj = bpy.data.objects.new(camera.name, camera)
    scene.collection.objects.link(camera_obj)
    scene.camera = camera_obj
    output.joinpath("images").mkdir(parents=True, exist_ok=True)
    frames = []
    started = time.monotonic()
    for split, index, azimuth_deg, elevation_deg in orbit():
        azimuth, elevation = math.radians(azimuth_deg), math.radians(elevation_deg)
        camera_obj.location = (radius * math.cos(elevation) * math.cos(azimuth), radius * math.cos(elevation) * math.sin(azimuth), radius * math.sin(elevation))
        camera_obj.rotation_euler = (-camera_obj.location).to_track_quat("-Z", "Y").to_euler()
        bpy.context.view_layer.update()
        filename = f"{split}_{index:03d}.png"
        scene.render.filepath = str(output / "images" / filename)
        scene.cycles.seed = 2503 + len(frames)
        bpy.ops.render.render(write_still=True)
        frames.append({"file_path": f"images/{filename}", "mask_path": f"masks/{filename}", "split": split,
                       "transform_matrix": [list(row) for row in camera_obj.matrix_world],
                       "azimuth_degrees": azimuth_deg, "elevation_degrees": elevation_deg})
    result = {"camera_angle_x": 2 * math.atan(camera.sensor_width / (2 * camera.lens)),
              "width": resolution, "height": resolution, "fl_x": resolution * camera.lens / camera.sensor_width,
              "fl_y": resolution * camera.lens / camera.sensor_width, "cx": resolution / 2, "cy": resolution / 2,
              "camera_model": "perspective pinhole", "coordinate_convention": "OpenGL/Blender camera-to-world: +X right, +Y up, -Z forward; scene Blender Z-up",
              "background": "neutral gray RGB(128,128,128), with separate alpha masks after prepare_data.py verification",
              "background_rgb": [128 / 255] * 3,
              "uid": record["uid"], "source_normalized_sha256": record["normalized_sha256"],
              "camera_radius": radius,
              "render": {"blender_version": bpy.app.version_string, "engine": scene.render.engine, "backend": backend,
                         "samples": scene.cycles.samples, "denoising": True, "seed": 2503,
                         "light_policy": "fixed world and three area lights shared across all views", "duration_seconds": time.monotonic() - started},
              "imported_bounds": {"min": low, "max": high}, "frames": frames}
    transforms_path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"HW2 completed {record['slug']}: {len(frames)} views in {result['render']['duration_seconds']:.1f}s", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--resolution", type=int, default=128)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:])
    root = args.manifest.resolve().parent.parent
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    for record in manifest["objects"]:
        render_object(root, record, args.resolution, args.force)


if __name__ == "__main__":
    main()
