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
from collections import deque

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

    raw_candidates = []
    for (_, hbin), c in clusters.items():
        if c["area"] <= 0:
            continue
        centroid = c["cent_acc"] / c["area"]
        span2 = c["max2"] - c["min2"]
        raw_candidates.append({
            "object_file": c["obj"],
            "height": float(hbin * height_bin),
            "hbin": int(hbin),
            "area": float(c["area"]),
            "centroid": centroid,
            "span2d": span2,
            "min2": c["min2"].copy(),
            "max2": c["max2"].copy(),
            "support_files": [c["obj"]],
        })

    # Merge fragmented supports at same height-bin by nearby/overlapping 2D spans.
    def _boxes_near(a_min, a_max, b_min, b_max, gap=0.08):
        sep_x = max(0.0, max(b_min[0] - a_max[0], a_min[0] - b_max[0]))
        sep_y = max(0.0, max(b_min[1] - a_max[1], a_min[1] - b_max[1]))
        return (sep_x <= gap) and (sep_y <= gap)

    by_h = defaultdict(list)
    for rc in raw_candidates:
        by_h[rc["hbin"]].append(rc)

    candidates = []
    for hbin, items in by_h.items():
        groups = []
        for it in items:
            groups.append({
                "hbin": hbin,
                "height": it["height"],
                "area": float(it["area"]),
                "cent_acc": np.asarray(it["centroid"], dtype=np.float64) * float(it["area"]),
                "min2": it["min2"].copy(),
                "max2": it["max2"].copy(),
                "support_files": set(it.get("support_files") or []),
            })

        changed = True
        while changed and len(groups) > 1:
            changed = False
            i = 0
            while i < len(groups):
                j = i + 1
                while j < len(groups):
                    gi, gj = groups[i], groups[j]
                    if _boxes_near(gi["min2"], gi["max2"], gj["min2"], gj["max2"]):
                        gi["area"] += gj["area"]
                        gi["cent_acc"] += gj["cent_acc"]
                        gi["min2"] = np.minimum(gi["min2"], gj["min2"])
                        gi["max2"] = np.maximum(gi["max2"], gj["max2"])
                        gi["support_files"].update(gj["support_files"])
                        groups.pop(j)
                        changed = True
                    else:
                        j += 1
                i += 1

        for g in groups:
            area = max(float(g["area"]), 1e-9)
            centroid = g["cent_acc"] / area
            candidates.append({
                "object_file": f"hbin_{hbin}",
                "height": float(g["height"]),
                "area": area,
                "centroid": centroid,
                "span2d": g["max2"] - g["min2"],
                "support_files": sorted(list(g["support_files"])),
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


def build_voxel_occupancy(room_min: np.ndarray, room_max: np.ndarray, furniture_aabbs, voxel_size: float):
    vs = max(1e-3, float(voxel_size))
    room_min = np.asarray(room_min, dtype=np.float64)
    room_max = np.asarray(room_max, dtype=np.float64)
    dims = np.maximum(1, np.ceil((room_max - room_min) / vs).astype(np.int64) + 1)
    occ = np.zeros((int(dims[0]), int(dims[1]), int(dims[2])), dtype=np.uint8)

    def _to_idx(p):
        g = np.floor((p - room_min) / vs).astype(np.int64)
        return np.clip(g, 0, dims - 1)

    for _name, bmin, bmax in furniture_aabbs:
        lo = _to_idx(np.asarray(bmin, dtype=np.float64))
        hi = _to_idx(np.asarray(bmax, dtype=np.float64))
        occ[lo[0] : hi[0] + 1, lo[1] : hi[1] + 1, lo[2] : hi[2] + 1] = 1

    return {
        "origin": room_min,
        "voxel": vs,
        "dims": dims,
        "occ": occ,
    }


def distance_field_from_occupancy(occ: np.ndarray, voxel_size: float):
    # 6-neighbor brushfire distance (L1 metric in voxel grid) as a robust 3D clearance proxy.
    inf = np.int32(2**30 - 1)
    dist = np.full(occ.shape, inf, dtype=np.int32)
    q = deque()
    occ_idx = np.argwhere(occ > 0)
    for i, j, k in occ_idx:
        dist[i, j, k] = 0
        q.append((int(i), int(j), int(k)))

    if not q:
        # No occupied voxel: treat as very large clearance.
        return np.full(occ.shape, 1e9, dtype=np.float64)

    sx, sy, sz = occ.shape
    neigh = ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))
    while q:
        x, y, z = q.popleft()
        nd = dist[x, y, z] + 1
        for dx, dy, dz in neigh:
            nx, ny, nz = x + dx, y + dy, z + dz
            if nx < 0 or ny < 0 or nz < 0 or nx >= sx or ny >= sy or nz >= sz:
                continue
            if nd < dist[nx, ny, nz]:
                dist[nx, ny, nz] = nd
                q.append((nx, ny, nz))
    return dist.astype(np.float64) * float(voxel_size)


def query_distance_field(df_pack, p3: np.ndarray):
    origin = df_pack["origin"]
    vs = float(df_pack["voxel"])
    dims = df_pack["dims"]
    df = df_pack["df"]
    g = np.floor((np.asarray(p3, dtype=np.float64) - origin) / vs).astype(np.int64)
    if np.any(g < 0) or np.any(g >= dims):
        return 0.0
    return float(df[int(g[0]), int(g[1]), int(g[2])])


def room_boundary_clearance(p3: np.ndarray, room_min_c: np.ndarray, room_max_c: np.ndarray) -> float:
    p = np.asarray(p3, dtype=np.float64)
    if np.any(p < room_min_c) or np.any(p > room_max_c):
        return 0.0
    return float(np.min(np.minimum(p - room_min_c, room_max_c - p)))


def point_box_distance_2d(p2: np.ndarray, bmin2: np.ndarray, bmax2: np.ndarray) -> float:
    q = np.maximum(np.maximum(bmin2 - p2, 0.0), p2 - bmax2)
    return float(np.linalg.norm(q))


def _is_ignored_support(name: str, ignore_object_file) -> bool:
    if ignore_object_file is None:
        return False
    if isinstance(ignore_object_file, (set, list, tuple)):
        return name in ignore_object_file
    return name == ignore_object_file


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
                if _is_ignored_support(name, ignore_object_file):
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


def list_empty_points_on_surface(
    candidate,
    room_center: np.ndarray,
    furniture_aabbs,
    up_idx: int,
    ignore_object_file: str,
    grid_n: int = 21,
    extra_margin: float = 0.03,
    top_k: int = 8,
):
    """
    Enumerate feasible placement points on support surface, sorted by
    (more clearance, closer to room center).
    """
    h_axes = [0, 1, 2]
    h_axes.remove(up_idx)
    a0, a1 = h_axes

    center3 = np.asarray(candidate["centroid"], dtype=np.float64)
    span = np.asarray(candidate["surface_span2d"], dtype=np.float64)
    min2 = np.array([center3[a0] - span[0] * 0.5, center3[a1] - span[1] * 0.5], dtype=np.float64)
    max2 = np.array([center3[a0] + span[0] * 0.5, center3[a1] + span[1] * 0.5], dtype=np.float64)

    xs = np.linspace(min2[0], max2[0], grid_n)
    ys = np.linspace(min2[1], max2[1], grid_n)
    room2 = np.array([room_center[a0], room_center[a1]], dtype=np.float64)

    ranked = []
    for x in xs:
        for y in ys:
            p2 = np.array([x, y], dtype=np.float64)
            min_clear = 1e18
            blocked = False
            for name, bmin3, bmax3 in furniture_aabbs:
                if _is_ignored_support(name, ignore_object_file):
                    continue
                bmin2 = np.array([bmin3[a0], bmin3[a1]], dtype=np.float64)
                bmax2 = np.array([bmax3[a0], bmax3[a1]], dtype=np.float64)
                d2 = point_box_distance_2d(p2, bmin2, bmax2)
                if d2 < extra_margin:
                    blocked = True
                    break
                if d2 < min_clear:
                    min_clear = d2
            if blocked:
                continue

            dist_center = float(np.linalg.norm(p2 - room2))
            score = (min_clear * 10.0) - dist_center
            p3 = center3.copy()
            p3[a0] = p2[0]
            p3[a1] = p2[1]
            ranked.append((score, p3))

    ranked.sort(key=lambda x: x[0], reverse=True)
    return [(p3, score) for score, p3 in ranked[:max(1, int(top_k))]]


def min_clearance_to_furniture_2d(
    p3: np.ndarray,
    furniture_aabbs,
    up_idx: int,
    ignore_object_file: str,
) -> float:
    h_axes = [0, 1, 2]
    h_axes.remove(up_idx)
    a0, a1 = h_axes
    p2 = np.array([p3[a0], p3[a1]], dtype=np.float64)
    min_clear = 1e18
    for name, bmin3, bmax3 in furniture_aabbs:
        if _is_ignored_support(name, ignore_object_file):
            continue
        bmin2 = np.array([bmin3[a0], bmin3[a1]], dtype=np.float64)
        bmax2 = np.array([bmax3[a0], bmax3[a1]], dtype=np.float64)
        d2 = point_box_distance_2d(p2, bmin2, bmax2)
        if d2 < min_clear:
            min_clear = d2
    if min_clear == 1e18:
        return 1e9
    return float(min_clear)


def estimate_max_diameter_geom(
    candidate,
    place_point: np.ndarray,
    room_min_c: np.ndarray,
    room_max_c: np.ndarray,
    furniture_aabbs,
    up_idx: int,
    ignore_object_file: str,
    anchor_height_ratio: float,
    extra_margin: float = 0.03,
    use_furniture_clearance: bool = True,
) -> float:
    h_axes = [0, 1, 2]
    h_axes.remove(up_idx)
    a0, a1 = h_axes

    center3 = np.asarray(candidate["centroid"], dtype=np.float64)
    span = np.asarray(candidate["surface_span2d"], dtype=np.float64)
    min2 = np.array([center3[a0] - span[0] * 0.5, center3[a1] - span[1] * 0.5], dtype=np.float64)
    max2 = np.array([center3[a0] + span[0] * 0.5, center3[a1] + span[1] * 0.5], dtype=np.float64)
    p2 = np.array([place_point[a0], place_point[a1]], dtype=np.float64)

    # 1) Support span constraint (footprint must stay on support).
    d_support = 2.0 * float(min(p2[0] - min2[0], max2[0] - p2[0], p2[1] - min2[1], max2[1] - p2[1]))

    # 2) Room XY constraint (keep footprint inside inner room bounds).
    d_room_xy = 2.0 * float(
        min(
            place_point[a0] - room_min_c[a0],
            room_max_c[a0] - place_point[a0],
            place_point[a1] - room_min_c[a1],
            room_max_c[a1] - place_point[a1],
        )
    )

    # 3) Furniture clearance in XY.
    if use_furniture_clearance:
        min_clear = min_clearance_to_furniture_2d(place_point, furniture_aabbs, up_idx, ignore_object_file)
        d_furniture = 2.0 * max(0.0, min_clear - extra_margin)
    else:
        d_furniture = 1e9

    # 4) Vertical room constraint for anchor center = place + up * (ratio * d).
    # Require anchor center +/- d/2 inside [room_min_c_up, room_max_c_up].
    p_up = float(place_point[up_idx])
    lo = float(room_min_c[up_idx])
    hi = float(room_max_c[up_idx])
    r = float(anchor_height_ratio)
    d_up = 1e9
    if (r + 0.5) > 1e-6:
        d_up = min(d_up, (hi - p_up) / (r + 0.5))
    if abs(r - 0.5) > 1e-6:
        # (r - 0.5) * d >= lo - p_up
        if (r - 0.5) > 0:
            d_up = min(d_up, (p_up - lo) / (r - 0.5))
        else:
            # Negative coefficient means lower bound always satisfied for d>=0 when p_up>=lo.
            pass

    d_max = min(d_support, d_room_xy, d_furniture, d_up)
    return max(0.0, float(d_max))


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
    df_pack=None,
    room_min_c=None,
    room_max_c=None,
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
        if df_pack is not None and room_min_c is not None and room_max_c is not None:
            c_occ = query_distance_field(df_pack, p)
            c_room = room_boundary_clearance(p, room_min_c, room_max_c)
            if min(c_occ, c_room) < furniture_clearance:
                collides = True
        else:
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
    require_all_cameras: bool = True,
    anchor_height_ratio: float = 0.5,
    diameter_alpha: float = 0.9,
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
    voxel_size = float(max(0.02, min(0.08, target_d_min / 6.0)))
    vox = build_voxel_occupancy(room_min_c, room_max_c, furniture_aabbs, voxel_size=voxel_size)
    vox["df"] = distance_field_from_occupancy(vox["occ"], voxel_size=voxel_size)

    fov_h, fov_v = compute_fov(lens, sensor_width, res_x, res_y)
    narrow = min(fov_h, fov_v)

    evals = []
    reject_stats = {
        "area_too_small": 0,
        "centroid_outside_room_margin": 0,
        "no_place_candidates": 0,
        "dmax_below_min": 0,
        "dhi_below_min": 0,
        "camera_infeasible": 0,
    }
    for c in candidates:
        if c["area"] < min_area:
            reject_stats["area_too_small"] += 1
            continue
        center = c["centroid"].copy()
        if np.any(center < room_min_c) or np.any(center > room_max_c):
            reject_stats["centroid_outside_room_margin"] += 1
            continue
        place_candidates = list_empty_points_on_surface(
            candidate={
                "centroid": center.tolist(),
                "surface_span2d": [float(c["span2d"][0]), float(c["span2d"][1])],
            },
            room_center=room_center,
            furniture_aabbs=furniture_aabbs,
            up_idx=up_idx,
            ignore_object_file=c.get("support_files", [c["object_file"]]),
            grid_n=21,
            top_k=10,
        )
        if not place_candidates:
            reject_stats["no_place_candidates"] += 1
            continue
        best_rec = None
        for place_point, place_score in place_candidates:
            d_max_geom = estimate_max_diameter_geom(
                candidate={
                    "centroid": center.tolist(),
                    "surface_span2d": [float(c["span2d"][0]), float(c["span2d"][1])],
                },
                place_point=place_point,
                room_min_c=room_min_c,
                room_max_c=room_max_c,
                furniture_aabbs=furniture_aabbs,
                up_idx=up_idx,
                ignore_object_file=c.get("support_files", [c["object_file"]]),
                anchor_height_ratio=anchor_height_ratio,
                use_furniture_clearance=False,
            )
            if d_max_geom < target_d_min:
                reject_stats["dmax_below_min"] += 1
                continue

            d_hi = min(float(target_d_max), float(d_max_geom) * float(diameter_alpha))
            if d_hi < target_d_min:
                reject_stats["dhi_below_min"] += 1
                continue

            # Find largest camera-feasible diameter with binary search.
            lo = float(target_d_min)
            hi = float(d_hi)
            best_d = None
            best_stats = None
            for _ in range(14):
                mid = 0.5 * (lo + hi)
                anchor_center = place_point.copy()
                anchor_center[up_idx] += float(mid) * float(anchor_height_ratio)
                if np.any(anchor_center < room_min_c) or np.any(anchor_center > room_max_c):
                    hi = mid
                    continue
                occ_clear = query_distance_field(vox, anchor_center)
                room_clear = room_boundary_clearance(anchor_center, room_min_c, room_max_c)
                if min(occ_clear, room_clear) < (0.5 * mid + 0.02):
                    hi = mid
                    continue
                r = mid * 0.5
                safe_dist = (r * camera_margin) / max(math.sin(narrow * 0.5), 1e-6)
                ok, total, coll = camera_feasible_count(
                    anchor_center,
                    mid,
                    room_min_c,
                    room_max_c,
                    up_idx,
                    num_views,
                    use_hemisphere,
                    safe_dist,
                    furniture_aabbs,
                    camera_furniture_clearance,
                    df_pack=vox,
                    room_min_c=room_min_c,
                    room_max_c=room_max_c,
                )
                feasible = (ok == total) if require_all_cameras else (ok > 0)
                if feasible:
                    best_d = mid
                    best_stats = (ok, total, coll, safe_dist, anchor_center.copy(), float(min(occ_clear, room_clear)))
                    lo = mid
                else:
                    hi = mid

            if best_d is None:
                reject_stats["camera_infeasible"] += 1
                continue

            ok, total, coll, safe_dist, anchor_center, anchor_clear_df = best_stats
            rec = {
                "object_file": c["object_file"],
                "centroid": center.tolist(),
                "placement_center": place_point.tolist(),
                "anchor_center": anchor_center.tolist(),
                "placement_score": float(place_score),
                "height": float(c["height"]),
                "surface_area": float(c["area"]),
                "surface_span2d": [float(c["span2d"][0]), float(c["span2d"][1])],
                "target_diameter": float(best_d),
                "d_max_geom": float(d_max_geom),
                "safe_distance_3d": float(safe_dist),
                "camera_ok": int(ok),
                "camera_total": int(total),
                "camera_ok_ratio": float(ok / max(total, 1)),
                "camera_collision_count": int(coll),
                "room_center_dist": float(np.linalg.norm(anchor_center - room_center)),
                "anchor_clearance_df": float(anchor_clear_df),
            }
            if (best_rec is None) or (
                (rec["target_diameter"], rec["d_max_geom"], rec["camera_ok"], -rec["room_center_dist"], rec["surface_area"])
                > (best_rec["target_diameter"], best_rec["d_max_geom"], best_rec["camera_ok"], -best_rec["room_center_dist"], best_rec["surface_area"])
            ):
                best_rec = rec

        if best_rec is not None:
            evals.append(best_rec)

    evals.sort(key=lambda x: (-x["target_diameter"], -x["d_max_geom"], -x["camera_ok"], x["room_center_dist"], -x["surface_area"]))
    best = evals[0] if evals else None
    return {
        "room_dir": room_dir,
        "up_axis": up_axis,
        "room_bounds": {"min": room_min.tolist(), "max": room_max.tolist()},
        "num_candidates": len(candidates),
        "num_evaluated": len(evals),
        "reject_stats": reject_stats,
        "voxel_size": voxel_size,
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
        require_all_cameras=True,
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
