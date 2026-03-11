#!/usr/bin/env python3
"""
V2: Find placement surface with trimesh-based geometry loading + 3D voxel distance field.

Design:
- Enumerate horizontal support planes from mesh triangles.
- Merge fragmented supports by height-bin and XY adjacency.
- Build 3D occupancy from furniture meshes via trimesh voxelization.
- Compute a 3D distance field on the voxel grid.
- Select anchor plane/point/diameter so camera set is feasible up-front.
"""

import os
import json
import math
import argparse
from collections import defaultdict, deque

import numpy as np

try:
    import trimesh
except Exception as e:
    raise RuntimeError("find_placement_surface_v2 requires trimesh") from e


def parse_args():
    p = argparse.ArgumentParser("Find placement surface v2")
    p.add_argument("--room-dir", required=True)
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
    p.add_argument("--camera-margin", type=float, default=1.35)
    p.add_argument("--camera-furniture-clearance", type=float, default=0.15)
    p.add_argument("--wall-margin", type=float, default=0.12)
    p.add_argument("--min-area", type=float, default=0.0)
    p.add_argument("--height-bin", type=float, default=0.02)
    p.add_argument("--max-tilt-deg", type=float, default=10.0)
    p.add_argument("--require-all-cameras", action="store_true", default=True)
    p.add_argument("--anchor-height-ratio", type=float, default=0.5)
    p.add_argument("--diameter-alpha", type=float, default=0.9)
    p.add_argument("--voxel-size", type=float, default=0.0, help="0 means auto")
    return p.parse_args()


def axis_index(axis: str):
    return {"X": 0, "Y": 1, "Z": 2}[axis]


def _load_mesh(path):
    mesh = trimesh.load(path, force="mesh", process=False)
    if isinstance(mesh, trimesh.Scene):
        geoms = [g for g in mesh.geometry.values() if isinstance(g, trimesh.Trimesh) and len(g.vertices) > 0]
        if not geoms:
            return None
        return trimesh.util.concatenate(geoms)
    if not isinstance(mesh, trimesh.Trimesh):
        return None
    if len(mesh.vertices) == 0:
        return None
    return mesh


def _discover_obj_paths(room_dir):
    return sorted([os.path.join(room_dir, n) for n in os.listdir(room_dir) if n.lower().endswith(".obj")])


def _room_bounds(room_dir):
    paths = _discover_obj_paths(room_dir)
    meshes = []
    room_mesh = os.path.join(room_dir, "meshes.obj")
    if os.path.isfile(room_mesh):
        m = _load_mesh(room_mesh)
        if m is not None:
            meshes.append(m)
    if not meshes:
        for p in paths:
            m = _load_mesh(p)
            if m is not None:
                meshes.append(m)
    if not meshes:
        raise RuntimeError("No valid OBJ mesh found")
    all_v = np.concatenate([m.vertices for m in meshes], axis=0)
    return np.min(all_v, axis=0), np.max(all_v, axis=0), (np.min(all_v, axis=0) + np.max(all_v, axis=0)) * 0.5


def build_support_candidates(room_dir, up_idx, max_tilt_cos, height_bin):
    raw = defaultdict(lambda: {
        "area": 0.0,
        "cent_acc": np.zeros(3, dtype=np.float64),
        "min2": np.array([1e9, 1e9], dtype=np.float64),
        "max2": np.array([-1e9, -1e9], dtype=np.float64),
        "support_files": set(),
    })
    h_axes = [0, 1, 2]
    h_axes.remove(up_idx)
    a0, a1 = h_axes

    for p in _discover_obj_paths(room_dir):
        name = os.path.basename(p)
        lname = name.lower()
        if lname.startswith("lighting_"):
            continue
        mesh = _load_mesh(p)
        if mesh is None or len(mesh.faces) == 0:
            continue

        tris = mesh.triangles
        areas = mesh.area_faces
        normals = mesh.face_normals
        cents = tris.mean(axis=1)

        valid = np.where((areas > 1e-8) & (normals[:, up_idx] >= max_tilt_cos))[0]
        for idx in valid:
            c = cents[idx]
            a = float(areas[idx])
            hb = int(round(c[up_idx] / max(height_bin, 1e-6)))
            k = (name, hb)
            ent = raw[k]
            ent["area"] += a
            ent["cent_acc"] += c * a
            tri2 = np.array([[tris[idx][0][a0], tris[idx][0][a1]], [tris[idx][1][a0], tris[idx][1][a1]], [tris[idx][2][a0], tris[idx][2][a1]]], dtype=np.float64)
            ent["min2"] = np.minimum(ent["min2"], np.min(tri2, axis=0))
            ent["max2"] = np.maximum(ent["max2"], np.max(tri2, axis=0))
            ent["support_files"].add(name)

    raw_cands = []
    for (name, hb), v in raw.items():
        if v["area"] <= 0:
            continue
        raw_cands.append({
            "hbin": hb,
            "height": float(hb * height_bin),
            "area": float(v["area"]),
            "centroid": v["cent_acc"] / v["area"],
            "min2": v["min2"].copy(),
            "max2": v["max2"].copy(),
            "support_files": set(v["support_files"]),
        })

    def near(a_min, a_max, b_min, b_max, gap=0.08):
        sep_x = max(0.0, max(b_min[0] - a_max[0], a_min[0] - b_max[0]))
        sep_y = max(0.0, max(b_min[1] - a_max[1], a_min[1] - b_max[1]))
        return (sep_x <= gap) and (sep_y <= gap)

    by_h = defaultdict(list)
    for c in raw_cands:
        by_h[c["hbin"]].append(c)

    out = []
    for hb, items in by_h.items():
        groups = []
        for it in items:
            groups.append({
                "hbin": hb,
                "height": it["height"],
                "area": it["area"],
                "cent_acc": np.asarray(it["centroid"], dtype=np.float64) * it["area"],
                "min2": it["min2"].copy(),
                "max2": it["max2"].copy(),
                "support_files": set(it["support_files"]),
            })

        changed = True
        while changed and len(groups) > 1:
            changed = False
            i = 0
            while i < len(groups):
                j = i + 1
                while j < len(groups):
                    if near(groups[i]["min2"], groups[i]["max2"], groups[j]["min2"], groups[j]["max2"]):
                        groups[i]["area"] += groups[j]["area"]
                        groups[i]["cent_acc"] += groups[j]["cent_acc"]
                        groups[i]["min2"] = np.minimum(groups[i]["min2"], groups[j]["min2"])
                        groups[i]["max2"] = np.maximum(groups[i]["max2"], groups[j]["max2"])
                        groups[i]["support_files"].update(groups[j]["support_files"])
                        groups.pop(j)
                        changed = True
                    else:
                        j += 1
                i += 1

        for g in groups:
            area = max(g["area"], 1e-9)
            centroid = g["cent_acc"] / area
            out.append({
                "object_file": f"hbin_{hb}",
                "height": float(g["height"]),
                "area": float(area),
                "centroid": centroid,
                "span2d": g["max2"] - g["min2"],
                "support_files": sorted(list(g["support_files"])),
            })

    return out


def collect_furniture_meshes(room_dir):
    out = []
    for p in _discover_obj_paths(room_dir):
        name = os.path.basename(p).lower()
        if name == "meshes.obj" or name.startswith("lighting_"):
            continue
        m = _load_mesh(p)
        if m is not None:
            out.append((os.path.basename(p), m))
    return out


def build_voxel_df(room_min_c, room_max_c, furniture_meshes, voxel_size):
    room_min_c = np.asarray(room_min_c, dtype=np.float64)
    room_max_c = np.asarray(room_max_c, dtype=np.float64)
    vs = float(max(1e-3, voxel_size))
    dims = np.maximum(1, np.ceil((room_max_c - room_min_c) / vs).astype(np.int64) + 1)
    occ = np.zeros((int(dims[0]), int(dims[1]), int(dims[2])), dtype=np.uint8)

    def to_idx(p):
        g = np.floor((np.asarray(p, dtype=np.float64) - room_min_c) / vs).astype(np.int64)
        return np.clip(g, 0, dims - 1)

    for _name, mesh in furniture_meshes:
        try:
            vox = mesh.voxelized(vs)
            pts = np.asarray(vox.points, dtype=np.float64)
            if pts.size == 0:
                continue
            ijk = np.floor((pts - room_min_c) / vs).astype(np.int64)
            valid = (
                (ijk[:, 0] >= 0) & (ijk[:, 0] < dims[0]) &
                (ijk[:, 1] >= 0) & (ijk[:, 1] < dims[1]) &
                (ijk[:, 2] >= 0) & (ijk[:, 2] < dims[2])
            )
            ijk = ijk[valid]
            occ[ijk[:, 0], ijk[:, 1], ijk[:, 2]] = 1
        except Exception:
            bmin, bmax = mesh.bounds
            lo = to_idx(bmin)
            hi = to_idx(bmax)
            occ[lo[0] : hi[0] + 1, lo[1] : hi[1] + 1, lo[2] : hi[2] + 1] = 1

    inf = np.int32(2**30 - 1)
    dist = np.full(occ.shape, inf, dtype=np.int32)
    q = deque()
    idx = np.argwhere(occ > 0)
    for i, j, k in idx:
        dist[i, j, k] = 0
        q.append((int(i), int(j), int(k)))

    if len(q) == 0:
        df = np.full(occ.shape, 1e9, dtype=np.float64)
    else:
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
        df = dist.astype(np.float64) * vs

    return {"origin": room_min_c, "voxel": vs, "dims": dims, "df": df}


def query_df(df_pack, p3):
    p = np.asarray(p3, dtype=np.float64)
    g = np.floor((p - df_pack["origin"]) / df_pack["voxel"]).astype(np.int64)
    if np.any(g < 0) or np.any(g >= df_pack["dims"]):
        return 0.0
    return float(df_pack["df"][int(g[0]), int(g[1]), int(g[2])])


def room_clearance(p3, room_min_c, room_max_c):
    p = np.asarray(p3, dtype=np.float64)
    if np.any(p < room_min_c) or np.any(p > room_max_c):
        return 0.0
    return float(np.min(np.minimum(p - room_min_c, room_max_c - p)))


def generate_fibonacci_points_world(n_samples, radius, center_loc, up_idx, hemisphere=True):
    points = []
    phi = math.pi * (3.0 - math.sqrt(5.0))
    h_axes = [0, 1, 2]
    h_axes.remove(up_idx)
    a0, a1 = h_axes
    for i in range(n_samples):
        safe_n = n_samples - 1 if n_samples > 1 else 1
        z = 1 - (i / safe_n) if hemisphere else 1 - (i / safe_n) * 2
        radius_at_z = math.sqrt(max(0.0, 1 - z * z)) * radius
        theta = phi * i
        x = math.cos(theta) * radius_at_z
        y = math.sin(theta) * radius_at_z
        z_world = z * radius
        arr = np.asarray(center_loc, dtype=np.float64).copy()
        arr[a0] += x
        arr[a1] += y
        arr[up_idx] += z_world
        points.append(arr)
    return points


def compute_fov(lens_mm, sensor_width_mm, res_x, res_y):
    fov_h = 2.0 * math.atan((sensor_width_mm * 0.5) / max(lens_mm, 1e-6))
    ar = float(res_x) / float(max(res_y, 1))
    fov_v = 2.0 * math.atan(math.tan(fov_h * 0.5) / ar)
    return fov_h, fov_v


def _point_box_distance_2d(p2, bmin2, bmax2):
    q = np.maximum(np.maximum(bmin2 - p2, 0.0), p2 - bmax2)
    return float(np.linalg.norm(q))


def _is_ignored(name, ignore_files):
    if ignore_files is None:
        return False
    if isinstance(ignore_files, (list, tuple, set)):
        return name in ignore_files
    return name == ignore_files


def list_place_points(candidate, room_center, furniture_meshes, up_idx, ignore_files, grid_n=21, top_k=10, extra_margin=0.03):
    h_axes = [0, 1, 2]
    h_axes.remove(up_idx)
    a0, a1 = h_axes

    c3 = np.asarray(candidate["centroid"], dtype=np.float64)
    span = np.asarray(candidate["span2d"], dtype=np.float64)
    min2 = np.array([c3[a0] - span[0] * 0.5, c3[a1] - span[1] * 0.5], dtype=np.float64)
    max2 = np.array([c3[a0] + span[0] * 0.5, c3[a1] + span[1] * 0.5], dtype=np.float64)

    xs = np.linspace(min2[0], max2[0], grid_n)
    ys = np.linspace(min2[1], max2[1], grid_n)
    rc2 = np.array([room_center[a0], room_center[a1]], dtype=np.float64)

    ranked = []
    for x in xs:
        for y in ys:
            p2 = np.array([x, y], dtype=np.float64)
            min_clear = 1e18
            blocked = False
            for name, mesh in furniture_meshes:
                if _is_ignored(name, ignore_files):
                    continue
                bmin, bmax = mesh.bounds
                bmin2 = np.array([bmin[a0], bmin[a1]], dtype=np.float64)
                bmax2 = np.array([bmax[a0], bmax[a1]], dtype=np.float64)
                d2 = _point_box_distance_2d(p2, bmin2, bmax2)
                if d2 < extra_margin:
                    blocked = True
                    break
                min_clear = min(min_clear, d2)
            if blocked:
                continue
            if min_clear == 1e18:
                min_clear = 1e9
            score = min_clear * 10.0 - float(np.linalg.norm(p2 - rc2))
            p3 = c3.copy()
            p3[a0], p3[a1] = p2[0], p2[1]
            ranked.append((score, p3))

    ranked.sort(key=lambda x: x[0], reverse=True)
    return [(p, s) for s, p in ranked[: max(1, int(top_k))]]


def estimate_dmax(candidate, place_point, room_min_c, room_max_c, up_idx, anchor_height_ratio):
    h_axes = [0, 1, 2]
    h_axes.remove(up_idx)
    a0, a1 = h_axes

    c3 = np.asarray(candidate["centroid"], dtype=np.float64)
    span = np.asarray(candidate["span2d"], dtype=np.float64)
    min2 = np.array([c3[a0] - span[0] * 0.5, c3[a1] - span[1] * 0.5], dtype=np.float64)
    max2 = np.array([c3[a0] + span[0] * 0.5, c3[a1] + span[1] * 0.5], dtype=np.float64)
    p2 = np.array([place_point[a0], place_point[a1]], dtype=np.float64)

    # Keep support span as a soft cap (very fragmented support can under-estimate).
    d_support = 2.0 * float(min(p2[0] - min2[0], max2[0] - p2[0], p2[1] - min2[1], max2[1] - p2[1]))
    d_room_xy = 2.0 * float(min(
        place_point[a0] - room_min_c[a0], room_max_c[a0] - place_point[a0],
        place_point[a1] - room_min_c[a1], room_max_c[a1] - place_point[a1],
    ))

    p_up = float(place_point[up_idx])
    lo = float(room_min_c[up_idx])
    hi = float(room_max_c[up_idx])
    r = float(anchor_height_ratio)
    d_up = 1e9
    if (r + 0.5) > 1e-6:
        d_up = min(d_up, (hi - p_up) / (r + 0.5))
    if abs(r - 0.5) > 1e-6 and (r - 0.5) > 0:
        d_up = min(d_up, (p_up - lo) / (r - 0.5))

    # Relax support cap by allowing larger diameter guided by room bounds.
    return max(0.0, float(min(max(d_support, 0.6 * d_room_xy), d_room_xy, d_up)))


def camera_feasible(anchor_center, td, room_min_c, room_max_c, up_idx, num_views, use_hemisphere, camera_margin, furniture_clearance, df_pack, lens, sensor_width, res_x, res_y, require_all_cameras):
    fov_h, fov_v = compute_fov(lens, sensor_width, res_x, res_y)
    narrow = max(1e-6, min(fov_h, fov_v))
    radius = (0.5 * td * camera_margin) / math.sin(narrow * 0.5)
    pts = generate_fibonacci_points_world(num_views, radius, np.asarray(anchor_center, dtype=np.float64), up_idx=up_idx, hemisphere=use_hemisphere)

    ok = 0
    coll = 0
    for p in pts:
        if np.any(p < room_min_c) or np.any(p > room_max_c):
            continue
        clear = min(query_df(df_pack, p), room_clearance(p, room_min_c, room_max_c))
        if clear < furniture_clearance:
            coll += 1
            continue
        ok += 1

    total = len(pts)
    feasible = (ok == total) if require_all_cameras else (ok > 0)
    return feasible, ok, total, coll, radius


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
    min_area: float = 0.02,
    height_bin: float = 0.02,
    max_tilt_deg: float = 10.0,
    require_all_cameras: bool = True,
    anchor_height_ratio: float = 0.5,
    diameter_alpha: float = 0.9,
    voxel_size: float = 0.0,
):
    room_dir = os.path.abspath(room_dir)
    up_idx = axis_index(up_axis)

    room_min, room_max, room_center = _room_bounds(room_dir)
    room_min_c = room_min + float(wall_margin)
    room_max_c = room_max - float(wall_margin)

    max_tilt_cos = math.cos(math.radians(max_tilt_deg))
    candidates = build_support_candidates(room_dir, up_idx, max_tilt_cos, height_bin)
    furniture_meshes = collect_furniture_meshes(room_dir)

    if voxel_size <= 0:
        voxel_size = float(max(0.02, min(0.08, target_d_min / 6.0)))
    df_pack = build_voxel_df(room_min_c, room_max_c, furniture_meshes, voxel_size)

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
        center = np.asarray(c["centroid"], dtype=np.float64)
        if np.any(center < room_min_c) or np.any(center > room_max_c):
            reject_stats["centroid_outside_room_margin"] += 1
            continue

        place_pts = list_place_points(c, room_center, furniture_meshes, up_idx, c.get("support_files", []), grid_n=21, top_k=12)
        if not place_pts:
            reject_stats["no_place_candidates"] += 1
            continue

        best_rec = None
        for place_point, place_score in place_pts:
            dmax = estimate_dmax(c, place_point, room_min_c, room_max_c, up_idx, anchor_height_ratio)
            if dmax < target_d_min:
                reject_stats["dmax_below_min"] += 1
                continue
            dhi = min(float(target_d_max), float(dmax) * float(diameter_alpha))
            if dhi < target_d_min:
                reject_stats["dhi_below_min"] += 1
                continue

            lo, hi = float(target_d_min), float(dhi)
            best_d = None
            best_stats = None
            for _ in range(14):
                mid = 0.5 * (lo + hi)
                anchor = np.asarray(place_point, dtype=np.float64).copy()
                anchor[up_idx] += float(anchor_height_ratio) * mid

                # Anchor center must have enough 3D clearance to host half-diameter cube radius.
                center_clear = min(query_df(df_pack, anchor), room_clearance(anchor, room_min_c, room_max_c))
                if center_clear < (0.5 * mid + 0.02):
                    hi = mid
                    continue

                feasible, ok, total, coll, safe_dist = camera_feasible(
                    anchor, mid, room_min_c, room_max_c, up_idx,
                    num_views, use_hemisphere, camera_margin,
                    camera_furniture_clearance, df_pack,
                    lens, sensor_width, res_x, res_y,
                    require_all_cameras,
                )
                if feasible:
                    best_d = mid
                    best_stats = (ok, total, coll, safe_dist, anchor.copy(), center_clear)
                    lo = mid
                else:
                    hi = mid

            if best_d is None:
                reject_stats["camera_infeasible"] += 1
                continue

            ok, total, coll, safe_dist, anchor_center, center_clear = best_stats
            rec = {
                "object_file": c["object_file"],
                "centroid": center.tolist(),
                "placement_center": np.asarray(place_point, dtype=np.float64).tolist(),
                "anchor_center": anchor_center.tolist(),
                "placement_score": float(place_score),
                "height": float(c["height"]),
                "surface_area": float(c["area"]),
                "surface_span2d": [float(c["span2d"][0]), float(c["span2d"][1])],
                "target_diameter": float(best_d),
                "d_max_geom": float(dmax),
                "anchor_clearance_df": float(center_clear),
                "safe_distance_3d": float(safe_dist),
                "camera_ok": int(ok),
                "camera_total": int(total),
                "camera_ok_ratio": float(ok / max(total, 1)),
                "camera_collision_count": int(coll),
                "room_center_dist": float(np.linalg.norm(anchor_center - room_center)),
                "support_files": c.get("support_files", []),
            }
            if (best_rec is None) or (
                (rec["target_diameter"], rec["d_max_geom"], rec["camera_ok"], -rec["room_center_dist"], rec["surface_area"]) >
                (best_rec["target_diameter"], best_rec["d_max_geom"], best_rec["camera_ok"], -best_rec["room_center_dist"], best_rec["surface_area"])
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
        "voxel_size": float(voxel_size),
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
        "furniture_mesh_count": len(furniture_meshes),
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
        require_all_cameras=args.require_all_cameras,
        anchor_height_ratio=args.anchor_height_ratio,
        diameter_alpha=args.diameter_alpha,
        voxel_size=args.voxel_size,
    )

    out_path = args.output or os.path.join(os.path.abspath(args.room_dir), "placement_surface_v2.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"saved: {out_path}")
    b = out.get("best_surface")
    if b:
        print(f"best: d={b['target_diameter']:.3f} cam={b['camera_ok']}/{b['camera_total']} anchor={b['anchor_center']}")
    else:
        print("best: none")


if __name__ == "__main__":
    main()
