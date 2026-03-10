#!/usr/bin/env python3
"""
Fix MTL texture bindings in json2obj exports.

What it does:
- Remove `map_Ka` lines (Blender OBJ importer warns unsupported).
- Resolve `map_Kd` texture paths to real files.
- Copy textures next to each .mtl file with stable local names.
- Rewrite `map_Kd` to local filenames.
"""

import os
import re
import sys
import shutil
import hashlib
import argparse


def parse_args():
    p = argparse.ArgumentParser("Fix MTL texture references")
    p.add_argument("--root", required=True, help="front_scenes root directory")
    p.add_argument("--dry-run", action="store_true", help="print changes without writing files")
    p.add_argument("--verbose", action="store_true", help="print per-file details")
    return p.parse_args()


def stable_local_name(src_path: str, prefix: str = "tex") -> str:
    src_norm = os.path.abspath(src_path)
    h = hashlib.md5(src_norm.encode("utf-8", errors="ignore")).hexdigest()[:8]
    return f"{prefix}_{h}_{os.path.basename(src_path)}"


def resolve_texture_path(tex_token: str, mtl_dir: str):
    tex = tex_token.strip().replace("\\", "/")
    if not tex:
        return None

    candidates = []
    if os.path.isabs(tex):
        candidates.append(tex)
    else:
        candidates.append(os.path.normpath(os.path.join(mtl_dir, tex)))

    # Handle bad join cases seen in Blender logs.
    if tex.startswith("/"):
        candidates.append(tex[1:])
    if tex.startswith("//"):
        candidates.append("/" + tex.lstrip("/"))

    for p in candidates:
        if p and os.path.isfile(p):
            return os.path.abspath(p)
    return None


def fix_one_mtl(mtl_path: str, dry_run: bool, verbose: bool):
    mtl_dir = os.path.dirname(mtl_path)
    changed = False
    removed_ka = 0
    rewritten_kd = 0
    unresolved_kd = 0
    copied_tex = 0
    out_lines = []

    with open(mtl_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            low = s.lower()

            if low.startswith("map_ka "):
                changed = True
                removed_ka += 1
                continue

            if low.startswith("map_kd "):
                raw = s.split(None, 1)[1].strip() if len(s.split(None, 1)) > 1 else ""
                tex_token = raw.split()[-1] if raw else ""
                resolved = resolve_texture_path(tex_token, mtl_dir)
                if resolved:
                    local_name = stable_local_name(resolved)
                    dst = os.path.join(mtl_dir, local_name)
                    if not os.path.exists(dst):
                        if not dry_run:
                            shutil.copy2(resolved, dst)
                        copied_tex += 1
                    out_lines.append(f"map_Kd {local_name}\n")
                    rewritten_kd += 1
                    if local_name != tex_token:
                        changed = True
                    continue
                else:
                    unresolved_kd += 1
                    out_lines.append(line)
                    continue

            out_lines.append(line)

    if changed and not dry_run:
        with open(mtl_path, "w", encoding="utf-8") as f:
            f.writelines(out_lines)

    if verbose and (changed or removed_ka or rewritten_kd or unresolved_kd):
        print(
            f"[fix_mtl] {mtl_path} "
            f"changed={changed} map_Ka_removed={removed_ka} "
            f"map_Kd_rewritten={rewritten_kd} map_Kd_unresolved={unresolved_kd} copied={copied_tex}"
        )

    return {
        "changed": int(changed),
        "removed_ka": removed_ka,
        "rewritten_kd": rewritten_kd,
        "unresolved_kd": unresolved_kd,
        "copied_tex": copied_tex,
    }


def main():
    args = parse_args()
    root = os.path.abspath(args.root)
    if not os.path.isdir(root):
        print(f"error: root does not exist: {root}", file=sys.stderr)
        sys.exit(1)

    total_mtl = 0
    changed_mtl = 0
    removed_ka = 0
    rewritten_kd = 0
    unresolved_kd = 0
    copied_tex = 0

    for dp, _, fs in os.walk(root):
        for fn in fs:
            if not fn.lower().endswith(".mtl"):
                continue
            total_mtl += 1
            st = fix_one_mtl(os.path.join(dp, fn), args.dry_run, args.verbose)
            changed_mtl += st["changed"]
            removed_ka += st["removed_ka"]
            rewritten_kd += st["rewritten_kd"]
            unresolved_kd += st["unresolved_kd"]
            copied_tex += st["copied_tex"]

    print(
        f"[fix_mtl][summary] root={root} mtl={total_mtl} changed={changed_mtl} "
        f"map_Ka_removed={removed_ka} map_Kd_rewritten={rewritten_kd} "
        f"map_Kd_unresolved={unresolved_kd} copied_textures={copied_tex} dry_run={args.dry_run}"
    )


if __name__ == "__main__":
    main()

