#!/usr/bin/env python3
"""
Pick a robust placement surface inside the already-loaded Blender scene.

Three stages:
1) "Rain" from ceiling to find upward-facing support points.
2) Shoot hemisphere rays per candidate and use the minimum hit distance as
   the local safe radius.
3) Select the candidate with the largest safe radius.
"""

import math
from typing import Dict, List, Optional, Sequence, Tuple

import bpy
from mathutils import Vector


def _axis_vector(axis: str) -> Vector:
    a = str(axis or "Z").upper()
    if a == "X":
        return Vector((1.0, 0.0, 0.0))
    if a == "Y":
        return Vector((0.0, 1.0, 0.0))
    return Vector((0.0, 0.0, 1.0))


def _scene_mesh_objects() -> List[bpy.types.Object]:
    out = []
    for obj in bpy.context.scene.objects:
        if obj.type != "MESH":
            continue
        if obj.hide_get() or obj.hide_render:
            continue
        out.append(obj)
    return out


def _world_bounds(objs: Sequence[bpy.types.Object]) -> Optional[Tuple[Vector, Vector]]:
    if not objs:
        return None
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


def _fibonacci_dirs_hemisphere(n: int, up: Vector) -> List[Vector]:
    up = up.normalized()
    n = max(12, int(n))
    phi = math.pi * (3.0 - math.sqrt(5.0))
    dirs = []

    helper = Vector((1.0, 0.0, 0.0))
    if abs(up.dot(helper)) > 0.9:
        helper = Vector((0.0, 1.0, 0.0))
    right = up.cross(helper).normalized()
    fwd = right.cross(up).normalized()

    for i in range(n):
        # z in [0, 1] for hemisphere
        z = (i + 0.5) / n
        r = math.sqrt(max(0.0, 1.0 - z * z))
        theta = phi * i
        x = math.cos(theta) * r
        y = math.sin(theta) * r
        d = (right * x + fwd * y + up * z).normalized()
        if d.dot(up) >= 0.0:
            dirs.append(d)
    return dirs


def _iter_rain_grid(
    bmin: Vector,
    bmax: Vector,
    up_axis: str,
    step: float,
    margin: float,
) -> List[Vector]:
    axes = [0, 1, 2]
    up_idx = {"X": 0, "Y": 1, "Z": 2}[str(up_axis).upper()]
    axes.remove(up_idx)
    a0, a1 = axes[0], axes[1]

    lo0 = bmin[a0] + margin
    hi0 = bmax[a0] - margin
    lo1 = bmin[a1] + margin
    hi1 = bmax[a1] - margin
    if hi0 <= lo0 or hi1 <= lo1:
        return []

    points = []
    v0 = lo0
    while v0 <= hi0 + 1e-9:
        v1 = lo1
        while v1 <= hi1 + 1e-9:
            arr = [0.0, 0.0, 0.0]
            arr[a0] = v0
            arr[a1] = v1
            arr[up_idx] = bmax[up_idx]
            points.append(Vector((arr[0], arr[1], arr[2])))
            v1 += step
        v0 += step
    return points


def _quantize_point(p: Vector, voxel: float) -> Tuple[int, int, int]:
    inv = 1.0 / max(1e-6, float(voxel))
    return (
        int(round(p.x * inv)),
        int(round(p.y * inv)),
        int(round(p.z * inv)),
    )


def _camera_radius_from_diameter(
    target_diameter: float,
    lens_mm: float,
    res_x: int,
    res_y: int,
    sensor_width_mm: float,
    camera_margin: float,
) -> float:
    target_radius = max(1e-6, float(target_diameter) * 0.5)
    lens = max(1e-6, float(lens_mm))
    sensor_w = max(1e-6, float(sensor_width_mm))
    fov_h = 2.0 * math.atan(sensor_w / (2.0 * lens))
    ar = float(res_x) / float(max(1, int(res_y)))
    fov_v = 2.0 * math.atan(math.tan(fov_h * 0.5) / max(1e-6, ar))
    narrow = max(1e-6, min(fov_h, fov_v))
    return (target_radius * float(camera_margin)) / math.sin(narrow * 0.5)


def _ray_exit_distance_in_aabb(origin: Vector, direction: Vector, bmin: Vector, bmax: Vector) -> float:
    """
    Distance from an (assumed inside) point to the AABB boundary along `direction`.
    """
    eps = 1e-9
    t_candidates = []
    for i in (0, 1, 2):
        d = float(direction[i])
        if d > eps:
            t = (float(bmax[i]) - float(origin[i])) / d
            if t > 0.0:
                t_candidates.append(t)
        elif d < -eps:
            t = (float(bmin[i]) - float(origin[i])) / d
            if t > 0.0:
                t_candidates.append(t)
    if not t_candidates:
        return float("inf")
    return min(t_candidates)


def _diameter_limit_from_safe_radius(
    safe_radius: float,
    lens_mm: float,
    res_x: int,
    res_y: int,
    sensor_width_mm: float,
    camera_margin: float,
) -> float:
    # If diameter = 1.0m, required camera radius is `scale`.
    scale = max(1e-6, _camera_radius_from_diameter(1.0, lens_mm, res_x, res_y, sensor_width_mm, camera_margin))
    return max(0.0, float(safe_radius)) / scale


def _center_probe_debug(
    scene: bpy.types.Scene,
    depsgraph,
    bmin: Vector,
    bmax: Vector,
    up: Vector,
    up_idx: int,
    hemisphere_rays: int,
    ray_max: float,
) -> Dict:
    room_center = Vector((
        (bmin.x + bmax.x) * 0.5,
        (bmin.y + bmax.y) * 0.5,
        (bmin.z + bmax.z) * 0.5,
    ))
    ceiling = bmax[up_idx] + 0.3
    down = -up
    eps = 0.01

    down_origin = Vector(room_center)
    down_origin[up_idx] = ceiling
    down_hit, down_loc, down_n, _f, down_obj, _m = scene.ray_cast(
        depsgraph, down_origin, down, distance=max(1.0, ceiling - bmin[up_idx] + 0.3)
    )
    down_info = {
        "origin": [float(down_origin.x), float(down_origin.y), float(down_origin.z)],
        "direction": [float(down.x), float(down.y), float(down.z)],
        "hit": bool(down_hit),
        "distance": float((Vector(down_loc) - down_origin).length) if down_hit else None,
        "object": str(down_obj.name) if down_hit and down_obj is not None else None,
        "location": [float(down_loc.x), float(down_loc.y), float(down_loc.z)] if down_hit else None,
        "normal": [float(down_n.x), float(down_n.y), float(down_n.z)] if down_hit else None,
    }

    hemi_dirs = _fibonacci_dirs_hemisphere(int(hemisphere_rays), up)
    hemi_origin = room_center + up * eps
    rays = []
    for i, d in enumerate(hemi_dirs):
        o = hemi_origin + d * eps
        hit, loc, normal, _f, obj, _m = scene.ray_cast(depsgraph, o, d, distance=float(ray_max))
        if hit:
            dist = (Vector(loc) - o).length
            rays.append(
                {
                    "idx": int(i),
                    "direction": [float(d.x), float(d.y), float(d.z)],
                    "hit": True,
                    "distance": float(dist),
                    "object": str(obj.name) if obj is not None else None,
                    "location": [float(loc.x), float(loc.y), float(loc.z)],
                    "normal": [float(normal.x), float(normal.y), float(normal.z)],
                }
            )
        else:
            dist = min(float(ray_max), _ray_exit_distance_in_aabb(o, d, bmin, bmax))
            rays.append(
                {
                    "idx": int(i),
                    "direction": [float(d.x), float(d.y), float(d.z)],
                    "hit": False,
                    "distance": float(dist),
                    "object": None,
                    "location": None,
                    "normal": None,
                }
            )

    return {
        "room_center": [float(room_center.x), float(room_center.y), float(room_center.z)],
        "down_ray": down_info,
        "hemisphere_origin": [float(hemi_origin.x), float(hemi_origin.y), float(hemi_origin.z)],
        "hemisphere_ray_count": len(rays),
        "hemisphere_rays": rays,
    }


def find_placement_surface(
    up_axis: str = "Z",
    ray_grid_step: float = 0.05,
    wall_margin: float = 0.05,
    max_tilt_deg: float = 60.0,
    hemisphere_rays: int = 256,
    ray_max: float = 4.0,
    voxel_size: float = 0.04,
    min_safe_radius: float = 0.20,
    probe_lift: float = 0.20,
) -> Dict:
    scene = bpy.context.scene
    depsgraph = bpy.context.evaluated_depsgraph_get()
    mesh_objs = _scene_mesh_objects()
    bounds = _world_bounds(mesh_objs)
    if not bounds:
        return {
            "num_candidates": 0,
            "num_evaluated": 0,
            "voxel_size": float(voxel_size),
            "best_surface": None,
            "reject_stats": {"no_mesh_objects": 1},
            "top_candidates": [],
        }

    bmin, bmax = bounds
    up = _axis_vector(up_axis).normalized()
    up_idx = {"X": 0, "Y": 1, "Z": 2}[str(up_axis).upper()]
    ceiling = bmax[up_idx] + 0.3
    floor = bmin[up_idx] - 0.3
    rain_points = _iter_rain_grid(bmin, bmax, up_axis, float(ray_grid_step), float(wall_margin))

    tilt_cos = math.cos(math.radians(float(max_tilt_deg)))
    seen = set()
    candidates = []
    reject_wall_or_tilt = 0
    reject_duplicate = 0

    down = -up
    ray_distance = max(1.0, ceiling - floor)
    for seed in rain_points:
        origin = Vector(seed)
        origin[up_idx] = ceiling
        hit, loc, normal, _face, hit_obj, _m = scene.ray_cast(depsgraph, origin, down, distance=ray_distance)
        if not hit or hit_obj is None:
            continue
        n = Vector(normal).normalized()
        if n.dot(up) < tilt_cos:
            reject_wall_or_tilt += 1
            continue
        key = _quantize_point(loc, float(voxel_size))
        if key in seen:
            reject_duplicate += 1
            continue
        seen.add(key)
        candidates.append(
            {
                "point": Vector(loc),
                "normal": n,
                "object_name": str(hit_obj.name),
            }
        )

    hemi_dirs = _fibonacci_dirs_hemisphere(int(hemisphere_rays), up)
    eps = 0.01
    lift = max(float(eps), float(probe_lift))
    evaluated = []
    reject_too_tight = 0
    for c in candidates:
        p = c["point"]
        min_dist = float(ray_max)
        for d in hemi_dirs:
            o = p + up * lift + d * eps
            hit, loc, _n, _f, _obj, _m = scene.ray_cast(depsgraph, o, d, distance=float(ray_max))
            if hit:
                dist = (Vector(loc) - o).length
            else:
                # If mesh misses (e.g. room shell not watertight), use room AABB
                # boundary as a conservative geometric cap for this direction.
                dist = min(float(ray_max), _ray_exit_distance_in_aabb(o, d, bmin, bmax))
            if dist < min_dist:
                min_dist = dist
                # Early break: already below minimum safe threshold,
                # this candidate will be rejected anyway.
                if min_dist < float(min_safe_radius):
                    break
        if min_dist < float(min_safe_radius):
            reject_too_tight += 1
            continue
        item = dict(c)
        item["safe_radius"] = float(min_dist)
        evaluated.append(item)

    evaluated.sort(key=lambda x: x["safe_radius"], reverse=True)
    best = evaluated[0] if evaluated else None

    top = []
    for c in evaluated[:12]:
        p = c["point"]
        top.append(
            {
                "object_file": c["object_name"],
                "centroid": [float(p.x), float(p.y), float(p.z)],
                "placement_center": [float(p.x), float(p.y), float(p.z)],
                "anchor_center": [float(p.x), float(p.y), float(p.z)],
                "d_max_geom": float(c["safe_radius"]),
                "target_diameter": float(c["safe_radius"] * 2.0),
                "camera_ok": 0,
                "camera_total": 0,
                "surface_area": float(voxel_size) * float(voxel_size),
                "room_center_dist": 0.0,
            }
        )

    best_surface = None
    if best is not None:
        p = best["point"]
        best_surface = {
            "object_file": best["object_name"],
            "centroid": [float(p.x), float(p.y), float(p.z)],
            "placement_center": [float(p.x), float(p.y), float(p.z)],
            "anchor_center": [float(p.x), float(p.y), float(p.z)],
            "normal": [float(best["normal"].x), float(best["normal"].y), float(best["normal"].z)],
            "d_max_geom": float(best["safe_radius"]),
            "target_diameter": float(best["safe_radius"] * 2.0),
            "camera_ok": 0,
            "camera_total": 0,
            "surface_area": float(voxel_size) * float(voxel_size),
            "room_center_dist": 0.0,
        }

    return {
        "num_candidates": len(candidates),
        "num_evaluated": len(evaluated),
        "voxel_size": float(voxel_size),
        "best_surface": best_surface,
        "reject_stats": {
            "wall_or_tilt_surface": int(reject_wall_or_tilt),
            "duplicate_candidate": int(reject_duplicate),
            "too_tight_safe_radius": int(reject_too_tight),
        },
        "debug_center_probe": _center_probe_debug(
            scene=scene,
            depsgraph=depsgraph,
            bmin=bmin,
            bmax=bmax,
            up=up,
            up_idx=up_idx,
            hemisphere_rays=int(hemisphere_rays),
            ray_max=float(ray_max),
        ),
        "top_candidates": top,
    }


def find_best_surface_result(
    room_dir: str = "",
    up_axis: str = "Z",
    num_views: int = 16,
    lens: float = 50.0,
    res_x: int = 1600,
    res_y: int = 1200,
    sensor_width: float = 36.0,
    use_hemisphere: bool = True,
    target_d_min: float = 0.15,
    target_d_max: float = 0.45,
    target_d_step: float = 0.05,
    hemisphere_rays: int = 256,
    ray_max: float = 4.0,
    min_safe_radius: float = 0.20,
    camera_margin: float = 1.40,
    camera_furniture_clearance: float = 0.15,
    wall_margin: float = 0.12,
    min_area: float = 0.06,
    height_bin: float = 0.02,
    max_tilt_deg: float = 10.0,
    **_kwargs,
) -> Dict:
    # Compatibility wrapper expected by batch_render.py and test_front_lighting.py.
    _ = (room_dir, target_d_step, camera_furniture_clearance, min_area, height_bin, use_hemisphere)
    probe_lift = max(float(target_d_min) * 0.5, float(min_safe_radius))

    base = find_placement_surface(
        up_axis=up_axis,
        wall_margin=float(wall_margin),
        max_tilt_deg=float(max_tilt_deg),
        hemisphere_rays=int(hemisphere_rays),
        ray_max=float(ray_max),
        min_safe_radius=float(min_safe_radius),
        probe_lift=float(probe_lift),
    )

    best = base.get("best_surface")
    tops = base.get("top_candidates") or []
    if not best:
        base["reject_stats"] = dict(base.get("reject_stats") or {})
        base["reject_stats"]["no_place_candidates"] = 1
        return base

    # Convert local safe radius into a diameter compatible with downstream camera logic.
    d_max = max(0.0, float(best.get("d_max_geom", 0.0)))
    by_camera = _diameter_limit_from_safe_radius(
        safe_radius=d_max,
        lens_mm=lens,
        res_x=res_x,
        res_y=res_y,
        sensor_width_mm=sensor_width,
        camera_margin=camera_margin,
    )
    d = max(float(target_d_min), min(float(target_d_max), by_camera))

    camera_r = _camera_radius_from_diameter(d, lens, res_x, res_y, sensor_width, camera_margin)
    cam_ok = int(camera_r <= d_max + 1e-6)
    cam_total = max(1, int(num_views))

    best["target_diameter"] = float(d)
    best["camera_ok"] = cam_ok * cam_total
    best["camera_total"] = cam_total

    for t in tops:
        local_d_max = max(0.0, float(t.get("d_max_geom", 0.0)))
        by_cam_t = _diameter_limit_from_safe_radius(
            safe_radius=local_d_max,
            lens_mm=lens,
            res_x=res_x,
            res_y=res_y,
            sensor_width_mm=sensor_width,
            camera_margin=camera_margin,
        )
        td = max(float(target_d_min), min(float(target_d_max), by_cam_t))
        tr = _camera_radius_from_diameter(td, lens, res_x, res_y, sensor_width, camera_margin)
        tok = int(tr <= local_d_max + 1e-6)
        t["target_diameter"] = float(td)
        t["camera_ok"] = tok * cam_total
        t["camera_total"] = cam_total

    base["best_surface"] = best
    base["top_candidates"] = tops
    return base
