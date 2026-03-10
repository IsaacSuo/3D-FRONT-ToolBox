#!/usr/bin/env python3
"""
Standalone 3D-FRONT lighting test renderer (Cycles, Blender).

Goal:
- Import rooms exported by json2obj.py
- Detect lamp furniture from model_info.json (super-category == Lighting)
- Add synthetic physically-plausible lights per lamp category
- Render many validation views (room-level + per-lamp close-up)

Run:
  blender -b -P scripts/test_front_lighting.py -- \
    --front-room-root ./front_scenes \
    --model-info ./model_info.json \
    --output ./lighting_test_output
"""

import os
import re
import sys
import json
import math
import argparse
import random
from typing import Dict, List, Tuple

import bpy
from mathutils import Vector


UUID_RE = re.compile(r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}")


def blender_args():
    argv = sys.argv
    if "--" in argv:
        return argv[argv.index("--") + 1 :]
    return []


def parse_args():
    p = argparse.ArgumentParser("3D-FRONT lighting sanity render")
    p.add_argument("--front-room-root", required=True, help="json2obj output root")
    p.add_argument("--model-info", default="./model_info.json", help="3D-FUTURE model_info.json")
    p.add_argument("--output", default="./lighting_test_output", help="output root")
    p.add_argument("--max-rooms", type=int, default=0, help="0 means all rooms")
    p.add_argument("--global-views", type=int, default=16, help="views per room")
    p.add_argument("--lamp-views", type=int, default=6, help="views per lamp")
    p.add_argument("--res-x", type=int, default=1600)
    p.add_argument("--res-y", type=int, default=1200)
    p.add_argument("--samples", type=int, default=512)
    p.add_argument("--preview-images", action="store_true", help="also render light-independent preview images")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="GPU", choices=["GPU", "CPU"])
    p.add_argument("--gpu-backend", default="CUDA", choices=["CUDA", "OPTIX", "HIP", "METAL", "ONEAPI"])
    p.add_argument("--world-strength", type=float, default=0.02, help="tiny ambient fill only")
    return p.parse_args(blender_args())


def load_model_info(path: str) -> Dict[str, dict]:
    data = json.load(open(path, "r", encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected list model_info.json, got: {type(data)}")
    out = {}
    for item in data:
        mid = str(item.get("model_id", "")).strip()
        if mid:
            out[mid] = item
    return out


def discover_room_dirs(front_room_root: str) -> List[str]:
    root = os.path.abspath(front_room_root)
    rooms = []
    for cur_root, _, names in os.walk(root):
        if "meshes.obj" in names:
            rooms.append(cur_root)
    rooms.sort()
    return rooms


def clear_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)


def setup_cycles(args):
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.render.resolution_x = args.res_x
    scene.render.resolution_y = args.res_y
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.image_settings.color_depth = "8"
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "None"
    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma = 1.0

    c = scene.cycles
    c.samples = args.samples
    c.use_adaptive_sampling = False
    c.use_denoising = False
    c.max_bounces = 16
    c.diffuse_bounces = 8
    c.glossy_bounces = 8
    c.transparent_max_bounces = 16
    c.transmission_bounces = 8
    c.sample_clamp_indirect = 10.0
    c.sample_clamp_direct = 0.0
    if hasattr(c, "caustics_reflective"):
        c.caustics_reflective = True
    if hasattr(c, "caustics_refractive"):
        c.caustics_refractive = True

    c.device = args.device
    if args.device == "GPU":
        prefs = bpy.context.preferences.addons["cycles"].preferences
        try:
            prefs.compute_device_type = args.gpu_backend
        except Exception:
            prefs.compute_device_type = "CUDA"
        prefs.get_devices()
        for d in prefs.devices:
            d.use = True


def setup_world(strength: float):
    scene = bpy.context.scene
    if not scene.world:
        scene.world = bpy.data.worlds.new("World")
    world = scene.world
    world.use_nodes = True
    nodes = world.node_tree.nodes
    links = world.node_tree.links
    nodes.clear()
    bg = nodes.new(type="ShaderNodeBackground")
    out = nodes.new(type="ShaderNodeOutputWorld")
    bg.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
    bg.inputs["Strength"].default_value = float(strength)
    links.new(bg.outputs["Background"], out.inputs["Surface"])


def import_obj(filepath: str) -> List[bpy.types.Object]:
    scene = bpy.context.scene
    before = set(o.name for o in scene.objects)
    try:
        bpy.ops.wm.obj_import(filepath=filepath)
    except Exception:
        bpy.ops.import_scene.obj(filepath=filepath)
    after = set(o.name for o in scene.objects)
    return [scene.objects[n] for n in (after - before) if scene.objects[n].type == "MESH"]


def obj_world_bounds(objs: List[bpy.types.Object]) -> Tuple[Vector, Vector]:
    mins = Vector((1e18, 1e18, 1e18))
    maxs = Vector((-1e18, -1e18, -1e18))
    for obj in objs:
        for c in obj.bound_box:
            wc = obj.matrix_world @ Vector(c)
            mins.x = min(mins.x, wc.x)
            mins.y = min(mins.y, wc.y)
            mins.z = min(mins.z, wc.z)
            maxs.x = max(maxs.x, wc.x)
            maxs.y = max(maxs.y, wc.y)
            maxs.z = max(maxs.z, wc.z)
    return mins, maxs


def parse_model_id_from_name(path: str) -> str:
    m = UUID_RE.search(os.path.basename(path))
    return m.group(0).lower() if m else ""


def kelvin_to_rgb(temp_k: float) -> Tuple[float, float, float]:
    t = max(1000.0, min(40000.0, float(temp_k))) / 100.0
    if t <= 66.0:
        r = 255.0
        g = 99.4708025861 * math.log(t) - 161.1195681661
        b = 0.0 if t <= 19.0 else (138.5177312231 * math.log(t - 10.0) - 305.0447927307)
    else:
        r = 329.698727446 * ((t - 60.0) ** -0.1332047592)
        g = 288.1221695283 * ((t - 60.0) ** -0.0755148492)
        b = 255.0
    clamp = lambda v: max(0.0, min(1.0, v / 255.0))
    return clamp(r), clamp(g), clamp(b)


def lamp_profile(category: str):
    c = (category or "").lower()
    # Return: light_type, power_w, color_temp, radius_m
    if "chandelier" in c:
        return ("AREA", 180.0, 2900, 0.20)
    if "pendant" in c:
        return ("AREA", 90.0, 3000, 0.12)
    if "ceiling" in c:
        return ("AREA", 120.0, 3500, 0.16)
    if "wall" in c:
        return ("POINT", 35.0, 3000, 0.08)
    if "floor" in c:
        return ("POINT", 45.0, 3000, 0.09)
    if "table" in c:
        return ("POINT", 30.0, 2700, 0.07)
    return ("POINT", 50.0, 3200, 0.08)


def add_light_for_lamp(instance_id: int, lamp_objs: List[bpy.types.Object], category: str):
    bmin, bmax = obj_world_bounds(lamp_objs)
    center = Vector(((bmin.x + bmax.x) * 0.5, (bmin.y + bmax.y) * 0.5, (bmin.z + bmax.z) * 0.5))

    ltype, power, cct, radius = lamp_profile(category)
    color = kelvin_to_rgb(cct)

    if ltype == "AREA":
        data = bpy.data.lights.new(name=f"L_{instance_id:04d}", type="AREA")
        data.energy = power
        data.color = color
        data.shape = "DISK"
        data.size = max(radius * 2.0, 0.05)
        # Ceiling-like lamps: light source slightly below lamp bbox top.
        loc = Vector((center.x, center.y, max(bmin.z + 0.02, bmax.z - radius)))
    else:
        data = bpy.data.lights.new(name=f"L_{instance_id:04d}", type="POINT")
        data.energy = power
        data.color = color
        data.shadow_soft_size = max(radius, 0.03)
        # Point lights in the upper half of the lamp volume.
        loc = Vector((center.x, center.y, (center.z + bmax.z) * 0.5))

    lobj = bpy.data.objects.new(data.name, data)
    bpy.context.scene.collection.objects.link(lobj)
    lobj.location = loc

    if ltype == "AREA":
        # Emit mainly downward.
        lobj.rotation_euler = (math.pi, 0.0, 0.0)

    return {
        "light_object": lobj.name,
        "category": category,
        "light_type": ltype,
        "power_w": power,
        "color_temp_k": cct,
        "radius_m": radius,
        "lamp_center": [center.x, center.y, center.z],
        "light_pos": [loc.x, loc.y, loc.z],
    }


def look_at(cam_obj: bpy.types.Object, target: Vector):
    direction = target - cam_obj.location
    cam_obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def ensure_camera():
    scene = bpy.context.scene
    cam_data = bpy.data.cameras.new(name="EvalCamera")
    cam = bpy.data.objects.new("EvalCamera", cam_data)
    scene.collection.objects.link(cam)
    scene.camera = cam
    cam_data.lens = 35
    return cam


def ring_points(center: Vector, radius: float, n: int, z: float):
    pts = []
    n = max(1, int(n))
    for i in range(n):
        a = 2.0 * math.pi * float(i) / float(n)
        pts.append(Vector((center.x + radius * math.cos(a), center.y + radius * math.sin(a), z)))
    return pts


def render_still(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    bpy.context.scene.render.filepath = path
    bpy.ops.render.render(write_still=True)

def get_or_create_preview_material():
    name = "__PreviewEmissionWhite__"
    mat = bpy.data.materials.get(name)
    if mat:
        return mat
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()
    out = nodes.new(type="ShaderNodeOutputMaterial")
    em = nodes.new(type="ShaderNodeEmission")
    em.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
    em.inputs["Strength"].default_value = 1.0
    links.new(em.outputs["Emission"], out.inputs["Surface"])
    return mat

def render_preview_still(path: str, preview_mat):
    if preview_mat is None or preview_mat.name not in bpy.data.materials:
        preview_mat = get_or_create_preview_material()
    scene = bpy.context.scene
    layer = scene.view_layers[0]
    old_override = layer.material_override
    old_engine = scene.render.engine
    old_samples = scene.cycles.samples
    try:
        layer.material_override = preview_mat
        scene.render.engine = "CYCLES"
        scene.cycles.samples = min(32, max(1, old_samples))
        render_still(path)
    finally:
        layer.material_override = old_override
        scene.render.engine = old_engine
        scene.cycles.samples = old_samples


def import_room_and_detect_lamps(room_dir: str, model_info_map: Dict[str, dict]):
    all_meshes = []
    lamp_instances = []
    debug = {
        "room_dir": room_dir,
        "num_obj_files": 0,
        "num_import_failed": 0,
        "num_meshes_obj": 0,
        "num_mid_missing": 0,
        "num_mid_not_in_model_info": 0,
        "num_not_lighting": 0,
        "num_lighting_detected": 0,
        "examples_not_in_model_info": [],
        "examples_not_lighting": [],
    }

    room_objs = sorted([p for p in os.listdir(room_dir) if p.lower().endswith(".obj")])
    debug["num_obj_files"] = len(room_objs)
    lamp_idx = 0
    for name in room_objs:
        p = os.path.join(room_dir, name)
        imported = import_obj(p)
        if not imported:
            debug["num_import_failed"] += 1
            continue
        all_meshes.extend(imported)

        if name.lower() == "meshes.obj":
            debug["num_meshes_obj"] += 1
            continue

        mid = parse_model_id_from_name(name)
        if not mid:
            # Fallback: json2obj furniture naming uses "<SuperCategory>_<model_id>_<idx>.obj".
            # If regex failed but prefix is Lighting_, still treat as lighting for diagnostics.
            if name.startswith("Lighting_"):
                lamp_instances.append(
                    {
                        "lamp_id": lamp_idx,
                        "model_id": "",
                        "category": "Lighting(FromFilename)",
                        "objects": imported,
                        "source_obj": name,
                    }
                )
                lamp_idx += 1
                debug["num_lighting_detected"] += 1
            else:
                debug["num_mid_missing"] += 1
            continue

        info = model_info_map.get(mid)
        if not info:
            debug["num_mid_not_in_model_info"] += 1
            if len(debug["examples_not_in_model_info"]) < 10:
                debug["examples_not_in_model_info"].append({"file": name, "model_id": mid})
            continue
        super_cat = str(info.get("super-category", "")).strip().lower()
        if super_cat != "lighting":
            debug["num_not_lighting"] += 1
            if len(debug["examples_not_lighting"]) < 10:
                debug["examples_not_lighting"].append(
                    {"file": name, "model_id": mid, "super-category": info.get("super-category"), "category": info.get("category")}
                )
            continue

        lamp_instances.append(
            {
                "lamp_id": lamp_idx,
                "model_id": mid,
                "category": str(info.get("category", "")),
                "objects": imported,
                "source_obj": name,
            }
        )
        lamp_idx += 1
        debug["num_lighting_detected"] += 1

    return all_meshes, lamp_instances, debug


def room_scene_name(front_room_root: str, room_dir: str):
    rel = os.path.relpath(room_dir, os.path.abspath(front_room_root))
    return rel.replace(os.sep, "__")


def main():
    args = parse_args()
    random.seed(args.seed)

    model_info = load_model_info(args.model_info)
    rooms = discover_room_dirs(args.front_room_root)
    if args.max_rooms > 0:
        rooms = rooms[: args.max_rooms]

    if not rooms:
        print("No rooms found. Need directories containing meshes.obj.")
        return

    os.makedirs(args.output, exist_ok=True)

    summary = {
        "front_room_root": os.path.abspath(args.front_room_root),
        "model_info": os.path.abspath(args.model_info),
        "num_rooms": len(rooms),
        "rooms": [],
    }
    preview_mat = None

    for r_i, room_dir in enumerate(rooms):
        clear_scene()
        setup_cycles(args)
        setup_world(args.world_strength)
        cam = ensure_camera()
        if args.preview_images:
            preview_mat = get_or_create_preview_material()

        scene_name = room_scene_name(args.front_room_root, room_dir)
        out_room = os.path.join(args.output, scene_name)
        os.makedirs(out_room, exist_ok=True)

        print(f"[{r_i + 1}/{len(rooms)}] room={scene_name}")
        meshes, lamp_instances, room_debug = import_room_and_detect_lamps(room_dir, model_info)
        if not meshes:
            print("  -> skip: no mesh imported")
            continue
        print(
            "  -> detect:"
            f" obj={room_debug['num_obj_files']}"
            f" lighting={room_debug['num_lighting_detected']}"
            f" mid_missing={room_debug['num_mid_missing']}"
            f" mid_not_found={room_debug['num_mid_not_in_model_info']}"
            f" not_lighting={room_debug['num_not_lighting']}"
        )

        bmin, bmax = obj_world_bounds(meshes)
        room_center = Vector(((bmin.x + bmax.x) * 0.5, (bmin.y + bmax.y) * 0.5, (bmin.z + bmax.z) * 0.5))
        room_radius = max((bmax.x - bmin.x), (bmax.y - bmin.y)) * 0.6
        room_floor_z = bmin.z
        room_h = max(2.0, bmax.z - bmin.z)

        lights_meta = []
        for lamp in lamp_instances:
            meta = add_light_for_lamp(lamp["lamp_id"], lamp["objects"], lamp["category"])
            meta["lamp_id"] = lamp["lamp_id"]
            meta["model_id"] = lamp["model_id"]
            meta["source_obj"] = lamp["source_obj"]
            lights_meta.append(meta)

        with open(os.path.join(out_room, "lights.json"), "w", encoding="utf-8") as f:
            json.dump(lights_meta, f, ensure_ascii=False, indent=2)
        with open(os.path.join(out_room, "detect_debug.json"), "w", encoding="utf-8") as f:
            json.dump(room_debug, f, ensure_ascii=False, indent=2)

        renders_meta = []

        # 1) Room-level global views.
        g_points = ring_points(
            center=room_center,
            radius=max(room_radius, 1.8),
            n=args.global_views,
            z=room_floor_z + min(1.6, room_h * 0.5),
        )
        for i, p in enumerate(g_points):
            cam.location = p
            look_at(cam, Vector((room_center.x, room_center.y, room_floor_z + min(1.2, room_h * 0.4))))
            img_path = os.path.join(out_room, "global", f"{i:03d}.png")
            render_still(img_path)
            if preview_mat is not None:
                prev_path = os.path.join(out_room, "global_preview", f"{i:03d}.png")
                render_preview_still(prev_path, preview_mat)
            renders_meta.append(
                {
                    "type": "global",
                    "path": os.path.relpath(img_path, out_room),
                    "camera_pos": [cam.location.x, cam.location.y, cam.location.z],
                    "camera_matrix_world": [list(row) for row in cam.matrix_world],
                }
            )

        # 2) Per-lamp close-up views, to validate lamp distribution.
        for lamp in lights_meta:
            lc = Vector(lamp["lamp_center"])
            lamp_dir = os.path.join(out_room, "lamps", f"lamp_{lamp['lamp_id']:03d}")
            lp = ring_points(
                center=lc,
                radius=max(0.6, min(2.0, room_radius * 0.35)),
                n=args.lamp_views,
                z=max(room_floor_z + 0.8, min(lc.z + 0.3, room_floor_z + room_h - 0.1)),
            )
            for j, p in enumerate(lp):
                cam.location = p
                look_at(cam, lc)
                img_path = os.path.join(lamp_dir, f"{j:03d}.png")
                render_still(img_path)
                if preview_mat is not None:
                    prev_path = os.path.join(out_room, "lamps_preview", f"lamp_{lamp['lamp_id']:03d}", f"{j:03d}.png")
                    render_preview_still(prev_path, preview_mat)
                renders_meta.append(
                    {
                        "type": "lamp_closeup",
                        "lamp_id": lamp["lamp_id"],
                        "path": os.path.relpath(img_path, out_room),
                        "camera_pos": [cam.location.x, cam.location.y, cam.location.z],
                        "camera_matrix_world": [list(row) for row in cam.matrix_world],
                    }
                )

        with open(os.path.join(out_room, "renders.json"), "w", encoding="utf-8") as f:
            json.dump(renders_meta, f, ensure_ascii=False, indent=2)

        summary["rooms"].append(
            {
                "room": scene_name,
                "room_dir": room_dir,
                "num_lamps_detected": len(lights_meta),
                "num_images": len(renders_meta),
            }
        )
        print(f"  -> lamps={len(lights_meta)} images={len(renders_meta)}")

    with open(os.path.join(args.output, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("Done.")


if __name__ == "__main__":
    main()
