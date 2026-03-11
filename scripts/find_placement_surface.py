#!/usr/bin/env python3
"""
Find a centered, camera-feasible support plane in a 3D-FRONT room export.

Design goal (non-heuristic):
1) enumerate horizontal support planes from geometry
2) enforce hard constraints (area / camera feasibility)
3) lexicographic optimization:
   a) maximize feasible camera count
   b) minimize distance to room center
   c) maximize support area
"""

import os
import re
import json
import math
import argparse
from collections import defaultdict

import numpy as np


def parse_args():
    p = argparse.ArgumentParser("Find centered placement surface")
    p.add_argument("--room-dir", required=True, help="front_scenes/<house>/<room>")
    p.add_argument("--up-axis", default="Y", choices=["X", "Y", "Z"])
    p.add_argument("--output", default=None)
    p.add_argument("--num-views", type=int, default=16)
    p.add_argument("--lens", type=float, default=50.0)
    p.add_argument("--res-x", type=int, default=1600)
    p.add_argument("--res-y", type=int, default=1200)
    p.add_argument("--sensor-width", type=float, default=36.0)
    p.add_argument("--use-hemisphere", action="store_true", default=True)
    p.add_argument("--target-d-min", type=float, default=0.35)
    p.add_argument("--target-d-max", type=float, default=1.20)
    p.add_argument("--target-d-step", type=float, default=0.05)
    p.add_argument("--camera-margin", type=float, default=1.35)
    p.add_argument("--camera-furniture-clearance", type=float, default=0.15)
    p.add_argument("--wall-margin", type=float, default=0.12)
    p.add_argument("--min-area", type=float, default=0.06)
    p.add_argument("--height-bin", type=float, default=0.02)
    p.add_argument("--max-tilt-deg", type=float, default=10.0)
    return p.parse_args()


def axis_index(axis: str):
    return {"X": 0, "Y": 1, "Z": 2}[axis]


def parse_obj(path):
    v = []
    f = []
    with open(path, "r", encoding="utf-8", errors="ignore") as fp:
        for line in fp:
            if line.startswith("v "):
                p = line.strip().split()
                if len(p) >= 4:
                    v.append([float(p[1]), float(p[2]), float(p[3])])
            elif line.startswith("f "):
                ids = []
                for tok in line.strip().split()[1:]:
                    idx = tok.split("/")[0]
                    if idx:
                        ids.append(int(idx) - 1)
                if len(ids) >= 3:
                    for i in range(1, len(ids) - 1):
                        f.append([ids[0], ids[i], ids[i + 1]])
    if not v or not f:
        return None, None
    return np.asarray(v, np.float64), np.asarray(f, np.int64)


def tri_area_normal_center(v0, v1, v2):
    e1 = v1 - v0
    e2 = v2 - v0
    n = np.cross(e1, e2)
    nlen = np.linalg.norm(n)
    if nlen < 1e-12:
        return None
    area = 0.5 * nlen
    return area, n / nlen, (v0 + v1 + v2) / 3.0


def generate_fibonacci_points(n_samples, radius, center_loc, hemisphere=True):
    points = []
    phi = math.pi * (3.0 - math.sqrt(5.0))
    for i in range(n_samples):
        safe_n = n_samples - 1 if n_samples > 1 else 1
        z = 1 - (i / safe_n) if hemisphere else 1 - (i / safe_n) * 2
        radius_at_z = math.sqrt(max(0.0, 1 - z * z)) * radius
        theta = phi * i
        x = math.cos(theta) * radius_at_z
        y = math.sin(theta) * radius_at_z
        z_world = z * radius
        points.append(np.array([x, y, z_world], dtype=np.float64) + center_loc)
    return points


def compute_fov(lens_mm, sensor_width_mm, res_x, res_y):
    fov_h = 2.0 * math.atan((sensor_width_mm * 0.5) / max(lens_mm, 1e-6))
    ar = float(res_x) / float(max(res_y, 1))
    fov_v = 2.0 * math.atan(math.tan(fov_h * 0.5) / ar)
    return fov_h, fov_v


def build_candidates(room_dir, up_idx, max_tilt_cos, height_bin):
    # Horizontal plane axes.
    h_axes = [0, 1, 2]
    h_axes.remove(up_idx)
    a0, a1 = h_axes

    # Collect room bounds from meshes.obj if present, fallback all objs.
    room_obj = os.path.join(room_dir, "meshes.obj")
    bounds_verts = []
    if os.path.isfile(room_obj):
        v, _ = parse_obj(room_obj)
        if v is not None:
            bounds_verts.append(v)
    if not bounds_verts:
        for fn in os.listdir(room_dir):
            if fn.lower().endswith(".obj"):
                v, _ = parse_obj(os.path.join(room_dir, fn))
                if v is not None:
                    bounds_verts.append(v)
    if not bounds_verts:
        raise RuntimeError("No valid OBJ geometry in room-dir")

    all_v = np.concatenate(bounds_verts, axis=0)
    room_min = np.min(all_v, axis=0)
    room_max = np.max(all_v, axis=0)
    room_center = (room_min + room_max) * 0.5

    # Candidate support surfaces from all obj except obvious non-supports.
    skip_prefix = ("lighting_",)
    clusters = defaultdict(lambda: {
        "obj": "",
        "area": 0.0,
        "cent_acc": np.zeros(3, dtype=np.float64),
        "min2": np.array([1e9, 1e9], dtype=np.float64),
        "max2": np.array([-1e9, -1e9], dtype=np.float64),
    })

    for fn in sorted(os.listdir(room_dir)):
        if not fn.lower().endswith(".obj"):
            continue
        lfn = fn.lower()
        if lfn.startswith(skip_prefix):
            continue

        v, f = parse_obj(os.path.join(room_dir, fn))
        if v is None:
            continue
        for tri in f:
            v0, v1, v2 = v[tri[0]], v[tri[1]], v[tri[2]]
            st = tri_area_normal_center(v0, v1, v2)
            if st is None:
                continue
            area, n, c = st
            if area < 1e-6:
                continue
            if n[up_idx] < max_tilt_cos:
                continue

            h_bin = int(round(c[up_idx] / max(height_bin, 1e-6)))
            key = (fn, h_bin)
            k = clusters[key]
            k["obj"] = fn
            k["area"] += area
            k["cent_acc"] += c * area

            tri2 = np.array([[v0[a0], v0[a1]], [v1[a0], v1[a1]], [v2[a0], v2[a1]]], dtype=np.float64)
            k["min2"] = np.minimum(k["min2"], np.min(tri2, axis=0))
            k["max2"] = np.maximum(k["max2"], np.max(tri2, axis=0))

    candidates = []
    for (_, hbin), c in clusters.items():
        if c["area"] <= 0:
            continue
        centroid = c["cent_acc"] / c["area"]
        span2 = c["max2"] - c["min2"]
        candidates.append({
            "object_file": c["obj"],
            "height": float(hbin * height_bin),
            "area": float(c["area"]),
            "centroid": centroid,
            "span2d": span2,
        })

    return room_min, room_max, room_center, candidates


def collect_furniture_aabbs(room_dir: str):
    """
    Collect furniture AABBs from room root OBJ files (excluding meshes.obj and lighting).
    """
    boxes = []
    for fn in sorted(os.listdir(room_dir)):
        if not fn.lower().endswith(".obj"):
            continue
        lfn = fn.lower()
        if lfn == "meshes.obj" or lfn.startswith("lighting_"):
            continue
        v, _ = parse_obj(os.path.join(room_dir, fn))
        if v is None or len(v) == 0:
            continue
        bmin = np.min(v, axis=0)
        bmax = np.max(v, axis=0)
        boxes.append((fn, bmin, bmax))
    return boxes


def point_box_distance_2d(p2: np.ndarray, bmin2: np.ndarray, bmax2: np.ndarray) -> float:
    q = np.maximum(np.maximum(bmin2 - p2, 0.0), p2 - bmax2)
    return float(np.linalg.norm(q))


def choose_empty_point_on_surface(
    candidate,
    target_diameter: float,
    room_center: np.ndarray,
    furniture_aabbs,
    up_idx: int,
    ignore_object_file: str,
    grid_n: int = 17,
    extra_margin: float = 0.03,
):
    """
    Pick a free placement point on support surface (not plain centroid):
    - inside support span
    - keeps object footprint away from other furniture projections
    """
    h_axes = [0, 1, 2]
    h_axes.remove(up_idx)
    a0, a1 = h_axes

    center3 = np.asarray(candidate["centroid"], dtype=np.float64)
    span = np.asarray(candidate["surface_span2d"], dtype=np.float64)
    half = max(1e-6, float(target_diameter) * 0.5)
    min2 = np.array([center3[a0] - span[0] * 0.5, center3[a1] - span[1] * 0.5], dtype=np.float64)
    max2 = np.array([center3[a0] + span[0] * 0.5, center3[a1] + span[1] * 0.5], dtype=np.float64)

    # Object center must stay inside support with its footprint.
    lo2 = min2 + half
    hi2 = max2 - half
    if np.any(lo2 >= hi2):
        return None, -1e9

    # Candidate points on a regular grid.
    xs = np.linspace(lo2[0], hi2[0], grid_n)
    ys = np.linspace(lo2[1], hi2[1], grid_n)
    room2 = np.array([room_center[a0], room_center[a1]], dtype=np.float64)

    best_p = None
    best_score = -1e18
    for x in xs:
        for y in ys:
            p2 = np.array([x, y], dtype=np.float64)
            min_clear = 1e18

            blocked = False
            for name, bmin3, bmax3 in furniture_aabbs:
                if name == ignore_object_file:
                    continue
                bmin2 = np.array([bmin3[a0], bmin3[a1]], dtype=np.float64)
                bmax2 = np.array([bmax3[a0], bmax3[a1]], dtype=np.float64)
                d2 = point_box_distance_2d(p2, bmin2, bmax2)
                # footprint separation in 2D
                if d2 < (half + extra_margin):
                    blocked = True
                    break
                if d2 < min_clear:
                    min_clear = d2
            if blocked:
                continue

            # Prefer emptier point, then closer to room center.
            dist_center = float(np.linalg.norm(p2 - room2))
            score = (min_clear * 10.0) - dist_center
            if score > best_score:
                best_score = score
                p3 = center3.copy()
                p3[a0] = p2[0]
                p3[a1] = p2[1]
                best_p = p3

    return best_p, best_score


def point_to_aabb_distance(p: np.ndarray, bmin: np.ndarray, bmax: np.ndarray) -> float:
    # Distance from point to AABB in 3D.
    q = np.maximum(np.maximum(bmin - p, 0.0), p - bmax)
    return float(np.linalg.norm(q))


def camera_feasible_count(
    candidate_center,
    target_d,
    room_min,
    room_max,
    up_idx,
    num_views,
    use_hemisphere,
    safe_distance,
    furniture_aabbs,
    furniture_clearance,
):
    # Build fibonacci points in canonical XYZ and remap canonical Z->up_idx.
    local_center = np.array([candidate_center[0], candidate_center[1], candidate_center[2]], dtype=np.float64)
    pts = generate_fibonacci_points(num_views, safe_distance, local_center, hemisphere=use_hemisphere)
    ok = 0
    collision = 0
    for p in pts:
        # Re-check inside room AABB.
        in_room = (
            room_min[0] <= p[0] <= room_max[0] and
            room_min[1] <= p[1] <= room_max[1] and
            room_min[2] <= p[2] <= room_max[2]
        )
        if not in_room:
            continue

        collides = False
        for _name, bmin, bmax in furniture_aabbs:
            d = point_to_aabb_distance(p, bmin, bmax)
            if d < furniture_clearance:
                collides = True
                break
        if collides:
            collision += 1
            continue
        else:
            ok += 1
    return ok, len(pts), collision


def find_best_surface_result(
    room_dir: str,
    up_axis: str = "Y",
    num_views: int = 16,
    lens: float = 50.0,
    res_x: int = 1600,
    res_y: int = 1200,
    sensor_width: float = 36.0,
    use_hemisphere: bool = True,
    target_d_min: float = 0.35,
    target_d_max: float = 1.20,
    target_d_step: float = 0.05,
    camera_margin: float = 1.35,
    camera_furniture_clearance: float = 0.15,
    wall_margin: float = 0.12,
    min_area: float = 0.06,
    height_bin: float = 0.02,
    max_tilt_deg: float = 10.0,
):
    room_dir = os.path.abspath(room_dir)
    up_idx = axis_index(up_axis)

    max_tilt_cos = math.cos(math.radians(max_tilt_deg))
    room_min, room_max, room_center, candidates = build_candidates(
        room_dir, up_idx, max_tilt_cos, height_bin
    )
    furniture_aabbs = collect_furniture_aabbs(room_dir)

    room_min_c = room_min + wall_margin
    room_max_c = room_max - wall_margin

    fov_h, fov_v = compute_fov(lens, sensor_width, res_x, res_y)
    narrow = min(fov_h, fov_v)

    d_values = []
    d = target_d_min
    while d <= target_d_max + 1e-9:
        d_values.append(round(d, 6))
        d += target_d_step

    evals = []
    for c in candidates:
        if c["area"] < min_area:
            continue
        center = c["centroid"].copy()
        if np.any(center < room_min_c) or np.any(center > room_max_c):
            continue
        span_limit = max(0.0, min(float(c["span2d"][0]), float(c["span2d"][1])) * 0.9)
        if span_limit <= 0:
            continue

        best_local = None
        for td in d_values:
            if td > span_limit:
                continue
            place_point, place_score = choose_empty_point_on_surface(
                candidate={
                    "centroid": center.tolist(),
                    "surface_span2d": [float(c["span2d"][0]), float(c["span2d"][1])],
                },
                target_diameter=td,
                room_center=room_center,
                furniture_aabbs=furniture_aabbs,
                up_idx=up_idx,
                ignore_object_file=c["object_file"],
            )
            if place_point is None:
                continue
            r = td * 0.5
            safe_dist = (r * camera_margin) / max(math.sin(narrow * 0.5), 1e-6)
            ok, total, coll = camera_feasible_count(
                place_point,
                td,
                room_min_c,
                room_max_c,
                up_idx,
                num_views,
                use_hemisphere,
                safe_dist,
                furniture_aabbs,
                camera_furniture_clearance,
            )
            rec = {
                "object_file": c["object_file"],
                "centroid": center.tolist(),
                "placement_center": place_point.tolist(),
                "placement_score": float(place_score),
                "height": float(c["height"]),
                "surface_area": float(c["area"]),
                "surface_span2d": [float(c["span2d"][0]), float(c["span2d"][1])],
                "target_diameter": float(td),
                "safe_distance_3d": float(safe_dist),
                "camera_ok": int(ok),
                "camera_total": int(total),
                "camera_ok_ratio": float(ok / max(total, 1)),
                "camera_collision_count": int(coll),
                "room_center_dist": float(np.linalg.norm(center - room_center)),
            }
            key = (rec["camera_ok"], rec["target_diameter"], rec["surface_area"])
            if best_local is None or key > (
                best_local["camera_ok"], best_local["target_diameter"], best_local["surface_area"]
            ):
                best_local = rec
        if best_local is not None:
            evals.append(best_local)

    evals.sort(key=lambda x: (-x["camera_ok"], x["room_center_dist"], -x["surface_area"]))
    best = evals[0] if evals else None
    return {
        "room_dir": room_dir,
        "up_axis": up_axis,
        "room_bounds": {"min": room_min.tolist(), "max": room_max.tolist()},
        "num_candidates": len(candidates),
        "num_evaluated": len(evals),
        "best_surface": best,
        "top_candidates": evals[:20],
        "camera_generation": {
            "num_views": num_views,
            "lens": lens,
            "res_x": res_x,
            "res_y": res_y,
            "sensor_width": sensor_width,
            "camera_margin": camera_margin,
            "camera_furniture_clearance": camera_furniture_clearance,
            "use_hemisphere": use_hemisphere,
        },
        "furniture_aabb_count": len(furniture_aabbs),
    }


def main():
    args = parse_args()
    out = find_best_surface_result(
        room_dir=args.room_dir,
        up_axis=args.up_axis,
        num_views=args.num_views,
        lens=args.lens,
        res_x=args.res_x,
        res_y=args.res_y,
        sensor_width=args.sensor_width,
        use_hemisphere=args.use_hemisphere,
        target_d_min=args.target_d_min,
        target_d_max=args.target_d_max,
        target_d_step=args.target_d_step,
        camera_margin=args.camera_margin,
        camera_furniture_clearance=args.camera_furniture_clearance,
        wall_margin=args.wall_margin,
        min_area=args.min_area,
        height_bin=args.height_bin,
        max_tilt_deg=args.max_tilt_deg,
    )

    out_path = args.output or os.path.join(os.path.abspath(args.room_dir), "placement_surface.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"saved: {out_path}")
    best = out.get("best_surface")
    if best:
        print(
            f"best: obj={best['object_file']} center={best['centroid']} "
            f"d={best['target_diameter']:.2f} camera_ok={best['camera_ok']}/{best['camera_total']}"
        )
    else:
        print("best: none")


if __name__ == "__main__":
    main()
