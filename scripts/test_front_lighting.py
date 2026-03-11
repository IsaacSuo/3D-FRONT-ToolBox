#!/usr/bin/env python3
"""
Standalone 3D-FRONT lighting test renderer (Cycles, Blender).

Goal:
- Import rooms exported by json2obj.py
- Detect lamp furniture from model_info.json (super-category == Lighting)
- Add synthetic physically-plausible lights per lamp category
- Render room-level validation views only

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

# Ensure sibling script imports work under Blender `-P` regardless of launch cwd.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from find_placement_surface import find_best_surface_result


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
    p.add_argument("--lamp-views", type=int, default=6, help="deprecated: ignored (no lamp close-up rendering)")
    p.add_argument("--res-x", type=int, default=1600)
    p.add_argument("--res-y", type=int, default=1200)
    p.add_argument("--lens", type=float, default=50.0)
    p.add_argument("--samples", type=int, default=512)
    p.add_argument("--preview-images", action="store_true", help="also render light-independent preview images")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="GPU", choices=["GPU", "CPU"])
    p.add_argument("--gpu-backend", default="CUDA", choices=["CUDA", "OPTIX", "HIP", "METAL", "ONEAPI"])
    p.add_argument("--world-strength", type=float, default=0.02, help="tiny ambient fill only")
    p.add_argument("--up-axis", default="Y", choices=["X", "Y", "Z"], help="up axis of imported room OBJ")
    p.add_argument("--camera-margin", type=float, default=1.4)
    p.add_argument("--camera-furniture-clearance", type=float, default=0.15)
    p.add_argument("--target-d-min", type=float, default=0.35)
    p.add_argument("--target-d-max", type=float, default=1.20)
    p.add_argument("--target-d-step", type=float, default=0.05)
    p.add_argument("--use-hemisphere", action="store_true", default=True)
    return p.parse_args(blender_args())


def axis_index(axis: str) -> int:
    return {"X": 0, "Y": 1, "Z": 2}[str(axis).upper()]


def get_comp(v: Vector, idx: int) -> float:
    return float((v.x, v.y, v.z)[idx])


def set_comp(v: Vector, idx: int, value: float) -> Vector:
    arr = [v.x, v.y, v.z]
    arr[idx] = float(value)
    return Vector((arr[0], arr[1], arr[2]))


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
    def _read_mtl_texture_map(obj_path: str) -> Dict[str, str]:
        obj_path = os.path.abspath(obj_path)
        obj_dir = os.path.dirname(obj_path)
        mtllib = None
        try:
            with open(obj_path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    s = line.strip()
                    if s.lower().startswith("mtllib "):
                        mtllib = s.split(None, 1)[1].strip()
                        break
        except Exception:
            return {}
        if not mtllib:
            return {}
        mtl_path = os.path.join(obj_dir, mtllib)
        if not os.path.isfile(mtl_path):
            return {}

        tex_map = {}
        cur_mtl = None
        try:
            with open(mtl_path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    s = line.strip()
                    if not s or s.startswith("#"):
                        continue
                    if s.lower().startswith("newmtl "):
                        cur_mtl = s.split(None, 1)[1].strip()
                        continue
                    if cur_mtl and s.lower().startswith("map_kd "):
                        raw = s.split(None, 1)[1].strip()
                        # handle simple map options by taking last token as path
                        tex = raw.split()[-1]
                        tex = tex.replace("\\", "/")
                        tex_path = tex if os.path.isabs(tex) else os.path.normpath(os.path.join(obj_dir, tex))
                        tex_map[cur_mtl] = tex_path
        except Exception:
            return {}
        return tex_map

    def _bind_textures(imported_objs: List[bpy.types.Object], tex_map: Dict[str, str]):
        if not tex_map:
            return
        for obj in imported_objs:
            for slot in obj.material_slots:
                mat = slot.material
                if mat is None:
                    continue
                mname = mat.name
                tex_path = tex_map.get(mname)
                if tex_path is None and "." in mname and mname.rsplit(".", 1)[1].isdigit():
                    tex_path = tex_map.get(mname.rsplit(".", 1)[0])
                if not tex_path or not os.path.isfile(tex_path):
                    continue

                if not mat.use_nodes:
                    mat.use_nodes = True
                nodes = mat.node_tree.nodes
                links = mat.node_tree.links

                bsdf = None
                for n in nodes:
                    if n.type == "BSDF_PRINCIPLED":
                        bsdf = n
                        break
                if bsdf is None:
                    bsdf = nodes.new(type="ShaderNodeBsdfPrincipled")
                    out = None
                    for n in nodes:
                        if n.type == "OUTPUT_MATERIAL":
                            out = n
                            break
                    if out is None:
                        out = nodes.new(type="ShaderNodeOutputMaterial")
                    links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])

                # Reuse existing image texture node if possible to avoid node bloat.
                tex_node = None
                for n in nodes:
                    if n.type == "TEX_IMAGE":
                        tex_node = n
                        break
                if tex_node is None:
                    tex_node = nodes.new(type="ShaderNodeTexImage")
                tex_node.image = bpy.data.images.load(tex_path, check_existing=True)

                # Replace old Base Color links with this texture.
                base_input = bsdf.inputs.get("Base Color")
                if base_input is not None:
                    for lk in list(base_input.links):
                        links.remove(lk)
                    links.new(tex_node.outputs["Color"], base_input)

    tex_map = _read_mtl_texture_map(filepath)
    scene = bpy.context.scene
    before = set(o.name for o in scene.objects)
    # Blender 5 `wm.obj_import` may resolve absolute MTL texture paths incorrectly
    # in some environments. Prefer legacy importer first if available.
    imported_ok = False
    if hasattr(bpy.ops, "import_scene") and hasattr(bpy.ops.import_scene, "obj"):
        try:
            bpy.ops.import_scene.obj(filepath=filepath)
            imported_ok = True
        except Exception:
            imported_ok = False
    if not imported_ok:
        bpy.ops.wm.obj_import(filepath=filepath)
    after = set(o.name for o in scene.objects)
    imported_meshes = [scene.objects[n] for n in (after - before) if scene.objects[n].type == "MESH"]
    _bind_textures(imported_meshes, tex_map)
    return imported_meshes


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


def add_light_for_lamp(instance_id: int, lamp_objs: List[bpy.types.Object], category: str, up_idx: int):
    bmin, bmax = obj_world_bounds(lamp_objs)
    center = Vector(((bmin.x + bmax.x) * 0.5, (bmin.y + bmax.y) * 0.5, (bmin.z + bmax.z) * 0.5))
    up_min = get_comp(bmin, up_idx)
    up_max = get_comp(bmax, up_idx)

    ltype, power, cct, radius = lamp_profile(category)
    color = kelvin_to_rgb(cct)

    if ltype == "AREA":
        data = bpy.data.lights.new(name=f"L_{instance_id:04d}", type="AREA")
        data.energy = power
        data.color = color
        data.shape = "DISK"
        data.size = max(radius * 2.0, 0.05)
        # Ceiling-like lamps: light source slightly below lamp bbox top.
        loc = set_comp(center, up_idx, max(up_min + 0.02, up_max - radius))
    else:
        data = bpy.data.lights.new(name=f"L_{instance_id:04d}", type="POINT")
        data.energy = power
        data.color = color
        data.shadow_soft_size = max(radius, 0.03)
        # Point lights in the upper half of the lamp volume.
        loc = set_comp(center, up_idx, (get_comp(center, up_idx) + up_max) * 0.5)

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


def ensure_camera(lens: float = 50.0):
    scene = bpy.context.scene
    cam_data = bpy.data.cameras.new(name="EvalCamera")
    cam = bpy.data.objects.new("EvalCamera", cam_data)
    scene.collection.objects.link(cam)
    scene.camera = cam
    cam_data.lens = float(lens)
    return cam


def create_anchor(center: Vector, cube_size_m: float):
    scene = bpy.context.scene
    name = "ANCHOR"
    if name in scene.objects:
        bpy.data.objects.remove(scene.objects[name], do_unlink=True)
    anchor = bpy.data.objects.new(name, None)
    scene.collection.objects.link(anchor)
    anchor.location = center
    anchor["cube_size_m"] = float(cube_size_m)
    return anchor


def ring_points(center: Vector, radius: float, n: int, up_value: float, up_idx: int):
    pts = []
    n = max(1, int(n))
    h_axes = [0, 1, 2]
    h_axes.remove(up_idx)
    a0, a1 = h_axes[0], h_axes[1]
    for i in range(n):
        a = 2.0 * math.pi * float(i) / float(n)
        arr = [center.x, center.y, center.z]
        arr[a0] = arr[a0] + radius * math.cos(a)
        arr[a1] = arr[a1] + radius * math.sin(a)
        arr[up_idx] = up_value
        pts.append(Vector((arr[0], arr[1], arr[2])))
    return pts


def generate_fibonacci_points_world(n_samples: int, radius: float, center_loc: Vector, up_idx: int, hemisphere: bool = True):
    points = []
    phi = math.pi * (3.0 - math.sqrt(5.0))
    h_axes = [0, 1, 2]
    h_axes.remove(up_idx)
    a0, a1 = h_axes[0], h_axes[1]
    for i in range(n_samples):
        safe_n = n_samples - 1 if n_samples > 1 else 1
        z = 1 - (i / safe_n) if hemisphere else 1 - (i / safe_n) * 2
        radius_at_z = math.sqrt(max(0.0, 1.0 - z * z)) * radius
        theta = phi * i
        x = math.cos(theta) * radius_at_z
        y = math.sin(theta) * radius_at_z
        z_world = z * radius
        arr = [center_loc.x, center_loc.y, center_loc.z]
        arr[a0] += x
        arr[a1] += y
        arr[up_idx] += z_world
        points.append(Vector((arr[0], arr[1], arr[2])))
    return points


def compute_safe_distance_3d(cam_obj: bpy.types.Object, res_x: int, res_y: int, target_diameter: float, margin: float):
    target_radius = max(1e-6, float(target_diameter) * 0.5)
    fov_h = float(cam_obj.data.angle)
    ar = float(res_x) / float(max(res_y, 1))
    fov_v = 2.0 * math.atan(math.tan(fov_h * 0.5) / ar)
    narrow = max(1e-6, min(fov_h, fov_v))
    return (target_radius * float(margin)) / math.sin(narrow * 0.5)


def render_still(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    bpy.context.scene.render.filepath = path
    bpy.ops.render.render(write_still=True)

def get_or_create_preview_material():
    name = "__PreviewClay__"
    mat = bpy.data.materials.get(name)
    if mat:
        return mat
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()
    out = nodes.new(type="ShaderNodeOutputMaterial")
    bsdf = nodes.new(type="ShaderNodeBsdfPrincipled")
    bsdf.inputs["Base Color"].default_value = (0.72, 0.72, 0.72, 1.0)
    bsdf.inputs["Roughness"].default_value = 1.0
    if "Specular IOR Level" in bsdf.inputs:
        bsdf.inputs["Specular IOR Level"].default_value = 0.0
    elif "Specular" in bsdf.inputs:
        bsdf.inputs["Specular"].default_value = 0.0
    links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    return mat

def render_preview_still(path: str, preview_mat):
    if preview_mat is None or preview_mat.name not in bpy.data.materials:
        preview_mat = get_or_create_preview_material()
    scene = bpy.context.scene
    layer = scene.view_layers[0]
    old_override = layer.material_override
    old_engine = scene.render.engine
    old_samples = scene.cycles.samples
    old_world_strength = None
    try:
        layer.material_override = preview_mat
        scene.render.engine = "CYCLES"
        scene.cycles.samples = min(32, max(1, old_samples))
        # Use bright ambient world for light-independent geometry preview.
        if scene.world and scene.world.use_nodes:
            bg = scene.world.node_tree.nodes.get("Background")
            if bg and "Strength" in bg.inputs:
                old_world_strength = bg.inputs["Strength"].default_value
                bg.inputs["Strength"].default_value = max(1.0, old_world_strength)
        render_still(path)
    finally:
        if old_world_strength is not None and scene.world and scene.world.use_nodes:
            bg = scene.world.node_tree.nodes.get("Background")
            if bg and "Strength" in bg.inputs:
                bg.inputs["Strength"].default_value = old_world_strength
        layer.material_override = old_override
        scene.render.engine = old_engine
        scene.cycles.samples = old_samples


def import_room_and_detect_lamps(room_dir: str, model_info_map: Dict[str, dict]):
    all_meshes = []
    shell_meshes = []
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
            shell_meshes.extend(imported)
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

    if not shell_meshes:
        shell_meshes = list(all_meshes)
    return all_meshes, shell_meshes, lamp_instances, debug


def _min_surface_distance(point: Vector, objs: List[bpy.types.Object], depsgraph) -> float:
    min_dist = float("inf")
    for obj in objs:
        eval_obj = obj.evaluated_get(depsgraph)
        if eval_obj.type != "MESH":
            continue
        try:
            p_local = eval_obj.matrix_world.inverted() @ point
            hit, loc, _normal, _face = eval_obj.closest_point_on_mesh(p_local)
            if not hit:
                continue
            loc_world = eval_obj.matrix_world @ loc
            d = (loc_world - point).length
            if d < min_dist:
                min_dist = d
        except Exception:
            continue
    return min_dist

def _segment_blocked_by_shell(start: Vector, end: Vector, shell_objs: List[bpy.types.Object], depsgraph, eps: float = 1e-3) -> bool:
    """
    Return True if segment start->end hits shell geometry before reaching end.
    """
    vec = end - start
    seg_len = vec.length
    if seg_len <= eps:
        return False
    direction = vec.normalized()
    origin = start + direction * eps
    max_dist = max(0.0, seg_len - eps * 2.0)
    if max_dist <= 0:
        return False

    for obj in shell_objs:
        eval_obj = obj.evaluated_get(depsgraph)
        if eval_obj.type != "MESH":
            continue
        try:
            inv = eval_obj.matrix_world.inverted()
            o_local = inv @ origin
            # direction is vector; use inverse 3x3 for local transform.
            d_local = (inv.to_3x3() @ direction).normalized()
            hit, loc, _normal, _face = eval_obj.ray_cast(o_local, d_local, distance=max_dist)
            if hit:
                hit_dist = (eval_obj.matrix_world @ loc - origin).length
                if hit_dist < max_dist - eps:
                    return True
        except Exception:
            continue
    return False


def _fit_camera_points(points: List[Vector], center: Vector, shell_objs: List[bpy.types.Object], clearance: float = 0.12):
    """
    Pull camera points toward center until they are no longer embedded in shell geometry.
    """
    depsgraph = bpy.context.evaluated_depsgraph_get()
    fitted = []
    for p in points:
        cur = Vector((p.x, p.y, p.z))
        for _ in range(30):
            d = _min_surface_distance(cur, shell_objs, depsgraph)
            blocked = _segment_blocked_by_shell(cur, center, shell_objs, depsgraph)
            # Must satisfy both:
            # 1) not embedded in shell
            # 2) line of sight to room center is not blocked by shell (outside-wall points fail here)
            if d >= clearance and not blocked:
                break
            cur = center.lerp(cur, 0.85)  # move 15% toward center
        fitted.append(cur)
    return fitted


def room_scene_name(front_room_root: str, room_dir: str):
    rel = os.path.relpath(room_dir, os.path.abspath(front_room_root))
    return rel.replace(os.sep, "__")


def _camera_outside_room_aabb(cam_pos: Vector, bmin: Vector, bmax: Vector, margin: float = 1e-4) -> bool:
    return (
        cam_pos.x < bmin.x - margin or cam_pos.x > bmax.x + margin or
        cam_pos.y < bmin.y - margin or cam_pos.y > bmax.y + margin or
        cam_pos.z < bmin.z - margin or cam_pos.z > bmax.z + margin
    )


def _clamp_into_room_aabb(p: Vector, bmin: Vector, bmax: Vector, margin: float = 0.08) -> Vector:
    return Vector((
        min(max(p.x, bmin.x + margin), bmax.x - margin),
        min(max(p.y, bmin.y + margin), bmax.y - margin),
        min(max(p.z, bmin.z + margin), bmax.z - margin),
    ))


def find_room_target_surface(room_dir: str, args):
    result = find_best_surface_result(
        room_dir=room_dir,
        up_axis=args.up_axis,
        num_views=args.global_views,
        lens=args.lens,
        res_x=args.res_x,
        res_y=args.res_y,
        use_hemisphere=args.use_hemisphere,
        target_d_min=args.target_d_min,
        target_d_max=args.target_d_max,
        target_d_step=args.target_d_step,
        camera_margin=args.camera_margin,
        camera_furniture_clearance=args.camera_furniture_clearance,
        require_all_cameras=True,
    )
    best = result.get("best_surface")
    return result, best


def main():
    args = parse_args()
    random.seed(args.seed)
    up_idx = axis_index(args.up_axis)

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
        "args": vars(args),
        "num_rooms": len(rooms),
        "rooms": [],
    }
    preview_mat = None

    for r_i, room_dir in enumerate(rooms):
        clear_scene()
        setup_cycles(args)
        setup_world(args.world_strength)
        cam = ensure_camera(args.lens)
        if args.preview_images:
            preview_mat = get_or_create_preview_material()

        scene_name = room_scene_name(args.front_room_root, room_dir)
        out_room = os.path.join(args.output, scene_name)
        os.makedirs(out_room, exist_ok=True)

        print(f"[{r_i + 1}/{len(rooms)}] room={scene_name}")
        meshes, shell_meshes, lamp_instances, room_debug = import_room_and_detect_lamps(room_dir, model_info)
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
        lights_meta = []
        for lamp in lamp_instances:
            meta = add_light_for_lamp(lamp["lamp_id"], lamp["objects"], lamp["category"], up_idx)
            meta["lamp_id"] = lamp["lamp_id"]
            meta["model_id"] = lamp["model_id"]
            meta["source_obj"] = lamp["source_obj"]
            lights_meta.append(meta)

        with open(os.path.join(out_room, "lights.json"), "w", encoding="utf-8") as f:
            json.dump(lights_meta, f, ensure_ascii=False, indent=2)
        with open(os.path.join(out_room, "detect_debug.json"), "w", encoding="utf-8") as f:
            json.dump(room_debug, f, ensure_ascii=False, indent=2)

        placement_result, best_surface = find_room_target_surface(room_dir, args)
        with open(os.path.join(out_room, "placement_surface.json"), "w", encoding="utf-8") as f:
            json.dump(placement_result, f, ensure_ascii=False, indent=2)

        if best_surface is None:
            print("  -> stop room: no feasible best surface")
            summary["rooms"].append(
                {
                    "room": scene_name,
                    "room_dir": room_dir,
                    "status": "skipped_no_best_surface",
                }
            )
            if len(rooms) == 1:
                print("  -> stop: single specified room has no feasible best surface")
                break
            print("  -> continue: try next room")
            continue

        target_center = Vector(best_surface.get("anchor_center") or best_surface.get("placement_center", best_surface["centroid"]))
        target_diameter = float(best_surface["target_diameter"])
        anchor = create_anchor(target_center, target_diameter)
        safe_distance_3d = compute_safe_distance_3d(cam, args.res_x, args.res_y, target_diameter, args.camera_margin)

        renders_meta = []

        # 1) Room-level global views (directly generated around best-surface anchor).
        g_points = generate_fibonacci_points_world(
            n_samples=args.global_views,
            radius=safe_distance_3d,
            center_loc=target_center,
            up_idx=up_idx,
            hemisphere=args.use_hemisphere,
        )
        for i, p in enumerate(g_points):
            cam.location = p
            look_at(cam, target_center)
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

        with open(os.path.join(out_room, "renders.json"), "w", encoding="utf-8") as f:
            json.dump(renders_meta, f, ensure_ascii=False, indent=2)

        outside_cnt = 0
        for r in renders_meta:
            cp = Vector(r["camera_pos"])
            if _camera_outside_room_aabb(cp, bmin, bmax):
                outside_cnt += 1
        if outside_cnt > 0:
            print(f"  -> warning: {outside_cnt}/{len(renders_meta)} cameras outside room AABB")

        summary["rooms"].append(
            {
                "room": scene_name,
                "room_dir": room_dir,
                "num_lamps_detected": len(lights_meta),
                "num_images": len(renders_meta),
                "target_center": [target_center.x, target_center.y, target_center.z],
                "target_diameter": target_diameter,
                "anchor": {
                    "name": anchor.name,
                    "location": [anchor.location.x, anchor.location.y, anchor.location.z],
                    "cube_size_m": float(anchor["cube_size_m"]),
                },
                "safe_distance_3d": safe_distance_3d,
                "room_bounds": {
                    "min": [bmin.x, bmin.y, bmin.z],
                    "max": [bmax.x, bmax.y, bmax.z],
                    "up_axis": args.up_axis,
                },
                "outside_aabb_camera_count": outside_cnt,
            }
        )
        print(f"  -> lamps={len(lights_meta)} images={len(renders_meta)}")

    with open(os.path.join(args.output, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("Done.")


if __name__ == "__main__":
    main()
