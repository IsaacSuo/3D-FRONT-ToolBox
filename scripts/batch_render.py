import bpy
import os
import glob
import math
import random
import json
import shutil
import hashlib
from mathutils import Vector, Matrix

# ==============================================================================
#                               配置区域
# ==============================================================================

# Anchor half-size（米）。ANCHOR 立方体边长为 1m，则 half-size=0.5m。
ANCHOR_HALF_SIZE_M = 0.5

MATERIAL_CONFIG = {
    # origin / mirror / glass_clear / glass_frosted / glass_tinted / plastic / ceramic / metal_painted / rubber
    # === 材质类型选择 ===
    # - fixed: 使用 type
    # - weighted: 按 type_weights 采样（用于一批数据集按比例混合材质族）
    "type_sampling": "weighted",  # fixed | weighted
    "type": "glass_clear",
    # 示例（会自动归一化；未知 key 会被忽略）：
    # "type_weights": {"glass_clear": 0.4, "plastic": 0.2, "ceramic": 0.2, "metal_painted": 0.1, "rubber": 0.1}
    # 推荐 B（通用材质覆盖）
    "type_weights": {
        "glass_clear": 0.25,
        "glass_frosted": 0.10,
        "glass_tinted": 0.05,
        "plastic": 0.25,
        "ceramic": 0.15,
        "metal_painted": 0.15,
        "rubber": 0.05,
    },
    # 仅对 glass_tinted 生效：体积吸收颜色与密度（密度越大颜色越深，且随厚度变化）
    "tint_color_rgba": (0.85, 0.95, 1.0, 1.0),
    "tint_density": 0.5,
    # 仅对 glass_frosted 生效
    "frosted_roughness": 0.2,
    # glass 系列 IOR
    "ior": 1.5,

    # === 玻璃观感控制 ===
    # clear/tinted 玻璃建议用 SHARP，避免看起来“发糊”；frosted 仍用 GGX/BECKMANN 更自然
    # 设为 False 则完全不改 Glass BSDF 的 distribution（用 Blender 默认）
    "glass_apply_distribution": True,
    "glass_distribution_clear": "GGX",
    "glass_distribution_frosted": "GGX",

    # === 随机材质（建议 per-object，保持单个物体跨视角一致）===
    # 注意：随机并不等于不真实。请只在物理合理范围内随机，并将采样结果写入 material.json 便于复现。
    "randomize": True,
    # 固定种子（同一批次可复现）；最终每个模型还会混入 glb 路径的稳定哈希
    "seed": 0,
    "random_glass": {
        # IOR：普通玻璃通常在 1.45~1.55 这一带
        "ior_min": 1.47,
        "ior_max": 1.53,
        # clear glass：roughness 建议非常小（多数接近 0，少数略磨砂）
        "clear_roughness_min": 0.0,
        "clear_roughness_max": 0.002,
        # frosted：roughness 更大
        "frosted_roughness_min": 0.10,
        "frosted_roughness_max": 0.60,
        # 在 glass_clear 上以小概率加“极轻微”体积吸收（仍像玻璃，不做糖果色）
        "allow_clear_tint_prob": 0.15,
        "tint_density_min": 0.01,
        "tint_density_max": 0.15,
        # 基础 tint 颜色 + 小幅抖动（保持低饱和）
        "tint_base_rgb": (0.92, 0.97, 1.00),
        "tint_jitter": 0.04,
    },

    # === 其他材质族（Principled BSDF）===
    "random_plastic": {
        "roughness_min": 0.05,
        "roughness_max": 0.60,
        "specular_min": 0.35,
        "specular_max": 0.55,
        "clearcoat_min": 0.0,
        "clearcoat_max": 0.25,
        "clearcoat_roughness_min": 0.0,
        "clearcoat_roughness_max": 0.20,
        # 颜色（低饱和更真实；需要更花可以调大）
        "value_min": 0.15,
        "value_max": 0.95,
        "saturation_max": 0.55,
    },
    "random_ceramic": {
        "roughness_min": 0.03,
        "roughness_max": 0.45,
        "specular_min": 0.45,
        "specular_max": 0.70,
        "clearcoat_min": 0.0,
        "clearcoat_max": 0.35,
        "clearcoat_roughness_min": 0.0,
        "clearcoat_roughness_max": 0.25,
        "value_min": 0.40,
        "value_max": 1.00,
        "saturation_max": 0.30,
    },
    # 金属喷漆：用 dielectric + clearcoat 近似（不做 metallic=1，避免“裸金属”外观）
    "random_metal_painted": {
        "roughness_min": 0.04,
        "roughness_max": 0.35,
        "specular_min": 0.40,
        "specular_max": 0.60,
        "clearcoat_min": 0.20,
        "clearcoat_max": 1.00,
        "clearcoat_roughness_min": 0.0,
        "clearcoat_roughness_max": 0.15,
        "value_min": 0.20,
        "value_max": 0.95,
        "saturation_max": 0.70,
    },
    "random_rubber": {
        "roughness_min": 0.60,
        "roughness_max": 0.95,
        "specular_min": 0.02,
        "specular_max": 0.20,
        "value_min": 0.02,
        "value_max": 0.25,
        "saturation_max": 0.20,
        "sheen_min": 0.0,
        "sheen_max": 0.25,
        "sheen_tint_min": 0.0,
        "sheen_tint_max": 0.50,
    },
}

WORLD_CONFIG = {
    # 设为空字符串则不改 World（使用 .blend 自带的 World 节点）
    "hdri_path": "./HDRI/",
    "strength": 1.0,
    # 只用 Z 轴旋转（度）
    "rotation_z_deg": 0.0,
}

CONFIG_PATHS = {
    # 场景来源模式：
    # - "blend": 读取 blend_file（原有逻辑）
    # - "front_room": 读取 json2obj 导出的房间目录（推荐服务器批量）
    "scene_source": "front_room",

    # 可填 .blend 文件路径，或填文件夹路径（将递归遍历文件夹内所有 .blend/.blender 文件）
    "blend_file": "./scenes/",
    # json2obj 输出根目录（形如: <root>/<house_uid>/<room_id>/...）
    "front_room_root": "./front_scenes/",
    "objects_dir": "./objects/objaverse_lvis_real/", 
    "output_dir": "./output/",
    "anchor_name": "ANCHOR",
}

FRONT_SCENE_CONFIG = {
    # 导入房间时优先使用 room_dir/meshes.obj；否则递归导入该房间目录内所有 obj
    "prefer_merged_meshes_obj": True,
    "include_room_furniture": True,

    # 依据房间里的灯具家具自动补光（近似真实室内灯具照明）
    "auto_add_lamp_lights": True,
    "lamp_keywords": [
        "lamp", "light", "chandelier", "pendant", "ceiling",
        "table_lamp", "floor_lamp", "wall_lamp"
    ],
    "lamp_power_w": 40.0,
    "lamp_radius_m": 0.08,
    "lamp_color_temp_k": 3200,
}

LOGIC_CONFIG = {
    "target_count":    100,
    "min_size_mb":     5,
    "max_size_mb":     50,

    # 可选：指定 objects_dir 下的 <hash>.glb 文件（填列表/字符串皆可）。
    # 非空时将跳过 target_count/min_size_mb/max_size_mb 的逻辑，直接渲染这里指定的模型。
    # 示例: ["0a1b2c3d4e5f.glb", "abcd1234"]  (后者会自动补 .glb)
    "object_hashes":   ["a4b3213dc9034cefa99e35f19272f382.glb"],
    
    "num_views":       10,
    "use_hemisphere":  True,
    "target_diameter": 1.0,
    "margin":          1.4,
    "lens":            50,
    "test_every":      2,

    # === 材质测试：单个 GLB 渲染所有材质族（方便 sanity check）===
    # enable=True 时，会对每个模型按 types 依次重设材质并完整渲染一遍。
    # 注意：脚本会强制只对第 1 个 glb 做 sweep（避免组数爆炸）。
    "material_sweep": {
        "enable": False,
        "types": [
            "glass_clear",
            "glass_frosted",
            "glass_tinted",
            "plastic",
            "ceramic",
            "metal_painted",
            "rubber",
            "mirror",
        ],
    },
}

RENDER_CONFIG = {
    "engine":           'CYCLES',
    "device":           'GPU',
    "gpu_backend":      'CUDA', 
    "gpu_indices":      [0], 
    
    # === 采样质量 ===
    "samples_max":      8192,
    # 自适应采样的最小样本数（科研级建议不要太低）
    "samples_min":      2048,
    "noise_threshold":  0.0005,
    "time_limit":       0,
    "use_denoising":    False,
    
    # === 光路反弹 (新增项) ===
    # 确保复杂物体内部不会变黑
    "light_paths": {
        "max_bounces":       32,  # 总反弹
        "diffuse_bounces":   16,  # 漫反射
        "glossy_bounces":    16,  # 镜面反射
        "transparent_max":   32,  # 透明穿透 (玻璃/Alpha)
        "transmission":      16,  # 透射
    },

    # === 焦散（科研/真实向）===
    # 玻璃/强高光/室内间接光建议开启；但会更难收敛、更慢
    "caustics_reflective": True,
    "caustics_refractive": True,
    # blur_glossy>0 会“近似”降低高频高光（更稳但更不物理）；要保留物理焦散建议 0.0
    "blur_glossy": 0.0,

    "use_light_tree":   False,

    "res_x":            1600,
    "res_y":            1200,
    "res_percent":      100,
}

# ==============================================================================
#                          工具函数
# ==============================================================================

def get_file_size_mb(file_path):
    try:
        return os.path.getsize(file_path) / (1024 * 1024)
    except:
        return 0

def filter_files_by_size_and_sample(directory, min_mb, max_mb, count):
    if not os.path.exists(directory):
        print(f"错误: 目录不存在 {directory}")
        return []
        
    all_files = glob.glob(os.path.join(directory, "*.glb"))
    valid_candidates = []
    
    print(f"正在扫描 {len(all_files)} 个文件的大小...")
    for f in all_files:
        size = get_file_size_mb(f)
        if min_mb <= size <= max_mb:
            valid_candidates.append((f, size))
            
    if not valid_candidates:
        return []
    
    if len(valid_candidates) <= count:
        return valid_candidates
    else:
        return random.sample(valid_candidates, count)

def resolve_object_glb_files(objects_dir, object_hashes):
    """
    从 objects_dir 中按 <hash>.glb 精确选择文件。
    object_hashes 支持：
      - 字符串："abcd1234" 或 "abcd1234.glb"
      - 列表/元组：["a", "b.glb", ...]
    返回 [(glb_path, size_mb), ...]，顺序与输入一致（去重保序）。
    """
    if not object_hashes:
        return []

    if isinstance(object_hashes, (str, bytes)):
        items = [object_hashes]
    else:
        try:
            items = list(object_hashes)
        except TypeError:
            items = [str(object_hashes)]

    if objects_dir.startswith("//"):
        obj_dir = os.path.abspath(objects_dir[2:])
    else:
        obj_dir = os.path.abspath(objects_dir)

    seen = set()
    resolved = []
    for raw in items:
        if raw is None:
            continue
        name = str(raw).strip()
        if not name:
            continue
        if not name.lower().endswith(".glb"):
            name = f"{name}.glb"
        if name in seen:
            continue
        seen.add(name)
        p = os.path.join(obj_dir, name)
        if not os.path.isfile(p):
            print(f"错误: 指定的 glb 不存在: {p}")
            continue
        resolved.append((p, get_file_size_mb(p)))

    return resolved

def resolve_blend_files(blend_path):
    """
    支持传入 .blend 文件路径或目录路径：
      - 文件：返回 [file]
      - 目录：递归遍历目录下所有 .blend/.blender 文件（按路径排序）
    说明：这里将 `//` 视为相对当前工作目录（而非相对某个 .blend）。
    """
    if not blend_path:
        return []
    p = (blend_path or "").strip()
    if not p:
        return []

    if p.startswith("//"):
        resolved = os.path.abspath(p[2:])
    else:
        resolved = os.path.abspath(p)

    if os.path.isfile(resolved):
        return [resolved]

    if os.path.isdir(resolved):
        exts = (".blend", ".blender")
        files = []
        for root, _, names in os.walk(resolved):
            for name in names:
                lname = name.lower()
                if lname.endswith(exts):
                    files.append(os.path.join(root, name))
        files.sort()
        return files

    return []

def resolve_front_room_dirs(front_room_root):
    """
    发现 json2obj 导出的房间目录：目录内包含 meshes.obj 即视为一个可渲染房间。
    返回 [(room_dir_abs, scene_name), ...]
    """
    if not front_room_root:
        return []
    p = (front_room_root or "").strip()
    if not p:
        return []

    root = os.path.abspath(p[2:]) if p.startswith("//") else os.path.abspath(p)
    if not os.path.isdir(root):
        return []

    rooms = []
    for cur_root, _, names in os.walk(root):
        if "meshes.obj" not in names:
            continue
        rel = os.path.relpath(cur_root, root).replace(os.sep, "__")
        scene_name = rel if rel and rel != "." else os.path.basename(cur_root)
        rooms.append((cur_root, scene_name))

    rooms.sort(key=lambda x: x[1])
    return rooms

def _import_obj(filepath):
    try:
        bpy.ops.wm.obj_import(filepath=filepath)
    except Exception:
        bpy.ops.import_scene.obj(filepath=filepath)

def _get_world_bounds(mesh_objects):
    if not mesh_objects:
        return None
    mins = Vector((1e18, 1e18, 1e18))
    maxs = Vector((-1e18, -1e18, -1e18))
    for obj in mesh_objects:
        for c in obj.bound_box:
            wc = obj.matrix_world @ Vector(c)
            mins.x = min(mins.x, wc.x)
            mins.y = min(mins.y, wc.y)
            mins.z = min(mins.z, wc.z)
            maxs.x = max(maxs.x, wc.x)
            maxs.y = max(maxs.y, wc.y)
            maxs.z = max(maxs.z, wc.z)
    return mins, maxs

def _kelvin_to_rgb(temp_k):
    # 近似黑体颜色（sRGB 0~1）
    t = max(1000.0, min(40000.0, float(temp_k))) / 100.0
    if t <= 66:
        r = 255.0
        g = 99.4708025861 * math.log(t) - 161.1195681661
        b = 0.0 if t <= 19 else (138.5177312231 * math.log(t - 10.0) - 305.0447927307)
    else:
        r = 329.698727446 * ((t - 60.0) ** -0.1332047592)
        g = 288.1221695283 * ((t - 60.0) ** -0.0755148492)
        b = 255.0
    clamp = lambda v: max(0.0, min(1.0, v / 255.0))
    return (clamp(r), clamp(g), clamp(b))

def _auto_add_room_lights_from_lamps(room_meshes):
    if not FRONT_SCENE_CONFIG.get("auto_add_lamp_lights", True):
        return
    keywords = [str(k).lower() for k in (FRONT_SCENE_CONFIG.get("lamp_keywords") or [])]
    if not keywords:
        return

    power = float(FRONT_SCENE_CONFIG.get("lamp_power_w", 40.0))
    radius = float(FRONT_SCENE_CONFIG.get("lamp_radius_m", 0.08))
    color = _kelvin_to_rgb(float(FRONT_SCENE_CONFIG.get("lamp_color_temp_k", 3200)))

    count = 0
    for obj in room_meshes:
        lname = obj.name.lower()
        if not any(k in lname for k in keywords):
            continue

        bounds = _get_world_bounds([obj])
        if not bounds:
            continue
        bmin, bmax = bounds
        loc = Vector(((bmin.x + bmax.x) * 0.5, (bmin.y + bmax.y) * 0.5, bmax.z - 0.03))

        light_data = bpy.data.lights.new(name=f"LampLight_{count:04d}", type='POINT')
        light_data.energy = power
        light_data.color = color
        light_data.shadow_soft_size = radius
        light_obj = bpy.data.objects.new(light_data.name, light_data)
        bpy.context.scene.collection.objects.link(light_obj)
        light_obj.location = loc
        count += 1

    print(f"  -> 自动灯具补光数量: {count}")

def prepare_scene_from_front_room(room_dir):
    """
    从 json2obj 的房间目录构建当前 Blender 场景，并返回 anchor 对象。
    """
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene

    room_dir = os.path.abspath(room_dir)
    mesh_objs = []

    # 1) 优先导入 meshes.obj（墙地顶等）
    merged_mesh = os.path.join(room_dir, "meshes.obj")
    if FRONT_SCENE_CONFIG.get("prefer_merged_meshes_obj", True) and os.path.isfile(merged_mesh):
        before = set(o.name for o in scene.objects)
        _import_obj(merged_mesh)
        after = set(o.name for o in scene.objects)
        mesh_objs.extend([scene.objects[n] for n in (after - before) if scene.objects[n].type == 'MESH'])

    # 2) 导入房间家具（可作为“真实灯具”来源）
    if FRONT_SCENE_CONFIG.get("include_room_furniture", True):
        furniture_objs = sorted(glob.glob(os.path.join(room_dir, "*.obj")))
        for fp in furniture_objs:
            if os.path.basename(fp).lower() == "meshes.obj":
                continue
            before = set(o.name for o in scene.objects)
            _import_obj(fp)
            after = set(o.name for o in scene.objects)
            mesh_objs.extend([scene.objects[n] for n in (after - before) if scene.objects[n].type == 'MESH'])

    # 3) 如果仍为空，递归兜底导入
    if not mesh_objs:
        for fp in sorted(glob.glob(os.path.join(room_dir, "**", "*.obj"), recursive=True)):
            before = set(o.name for o in scene.objects)
            _import_obj(fp)
            after = set(o.name for o in scene.objects)
            mesh_objs.extend([scene.objects[n] for n in (after - before) if scene.objects[n].type == 'MESH'])

    if not mesh_objs:
        print(f"错误: 房间目录未导入任何网格: {room_dir}")
        return None

    # 4) 自动推断 Anchor（房间平面中心 + 地面高度）
    bounds = _get_world_bounds(mesh_objs)
    if not bounds:
        return None
    bmin, bmax = bounds
    cx, cy = (bmin.x + bmax.x) * 0.5, (bmin.y + bmax.y) * 0.5
    floor_z = bmin.z

    anchor = bpy.data.objects.new(CONFIG_PATHS.get("anchor_name", "ANCHOR"), None)
    scene.collection.objects.link(anchor)
    anchor.location = Vector((cx, cy, floor_z + ANCHOR_HALF_SIZE_M))

    _auto_add_room_lights_from_lamps(mesh_objs)
    return anchor

def generate_fibonacci_points(n_samples, radius, center_loc, hemisphere=True):
    points = []
    phi = math.pi * (3.0 - math.sqrt(5.0))
    for i in range(n_samples):
        safe_n = n_samples - 1 if n_samples > 1 else 1
        z = 1 - (i / safe_n) if hemisphere else 1 - (i / safe_n) * 2
        
        radius_at_z = math.sqrt(1 - z * z) * radius
        theta = phi * i
        
        x = math.cos(theta) * radius_at_z
        y = math.sin(theta) * radius_at_z
        z_world = z * radius
        
        points.append(Vector((x, y, z_world)) + center_loc)
    return points

def setup_world_hdri(scene):
    hdri_path = WORLD_CONFIG.get("hdri_path", "").strip()
    if not hdri_path:
        return

    resolved = bpy.path.abspath(hdri_path) if hdri_path.startswith("//") else os.path.abspath(hdri_path)
    if not os.path.exists(resolved):
        print(f"错误: 找不到 HDRI 路径: {resolved}")
        raise FileNotFoundError(resolved)

    if os.path.isdir(resolved):
        exts = {".hdr", ".exr"}
        candidates = []
        for name in os.listdir(resolved):
            p = os.path.join(resolved, name)
            if os.path.isfile(p) and os.path.splitext(name)[1].lower() in exts:
                candidates.append(p)
        if not candidates:
            print(f"错误: HDRI 文件夹为空或无 .hdr/.exr: {resolved}")
            raise FileNotFoundError(resolved)
        resolved = random.choice(candidates)

    world = scene.world
    if not world:
        world = bpy.data.worlds.new(name="World")
        scene.world = world

    world.use_nodes = True
    tree = world.node_tree
    nodes = tree.nodes
    links = tree.links
    nodes.clear()

    tex_coord = nodes.new(type="ShaderNodeTexCoord")
    tex_coord.location = (-900, 0)

    mapping = nodes.new(type="ShaderNodeMapping")
    mapping.location = (-700, 0)
    rotation_z = math.radians(float(WORLD_CONFIG.get("rotation_z_deg", 0.0)))
    mapping.inputs["Rotation"].default_value[2] = rotation_z

    env_tex = nodes.new(type="ShaderNodeTexEnvironment")
    env_tex.location = (-500, 0)
    env_tex.image = bpy.data.images.load(resolved, check_existing=True)
    print(f"HDRI: {resolved}")

    bg = nodes.new(type="ShaderNodeBackground")
    bg.location = (-250, 0)
    bg.inputs["Strength"].default_value = float(WORLD_CONFIG.get("strength", 1.0))

    out = nodes.new(type="ShaderNodeOutputWorld")
    out.location = (0, 0)

    links.new(tex_coord.outputs["Generated"], mapping.inputs["Vector"])
    links.new(mapping.outputs["Vector"], env_tex.inputs["Vector"])
    links.new(env_tex.outputs["Color"], bg.inputs["Color"])
    links.new(bg.outputs["Background"], out.inputs["Surface"])

def _get_or_create_principled_material(mat_name):
    mat = bpy.data.materials.get(mat_name)
    if not mat:
        mat = bpy.data.materials.new(name=mat_name)
    if not mat.use_nodes:
        mat.use_nodes = True
    return mat

def _new_material(mat_name):
    mat = bpy.data.materials.new(name=mat_name)
    mat.use_nodes = True
    return mat

def _stable_int_from_string(s):
    h = hashlib.md5(str(s).encode("utf-8", errors="ignore")).hexdigest()
    return int(h[:8], 16)

def _weighted_choice(rng, items):
    """
    items: [(key, weight>0), ...]
    returns key or None
    """
    total = 0.0
    for _k, w in items:
        try:
            fw = float(w)
        except Exception:
            continue
        if fw > 0:
            total += fw
    if total <= 0:
        return None
    r = rng.random() * total
    acc = 0.0
    for k, w in items:
        try:
            fw = float(w)
        except Exception:
            continue
        if fw <= 0:
            continue
        acc += fw
        if r <= acc:
            return k
    return items[-1][0] if items else None

def _clamp01(x):
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else float(x)

def _srgb_to_linear_1(x):
    x = _clamp01(float(x))
    if x <= 0.04045:
        return x / 12.92
    return ((x + 0.055) / 1.055) ** 2.4

def _rgb_srgb_to_linear(rgb):
    try:
        r, g, b = float(rgb[0]), float(rgb[1]), float(rgb[2])
    except Exception:
        r, g, b = 1.0, 1.0, 1.0
    return (_srgb_to_linear_1(r), _srgb_to_linear_1(g), _srgb_to_linear_1(b))

def _hsv_to_rgb(h, s, v):
    # h,s,v in [0,1]
    h = float(h) % 1.0
    s = _clamp01(float(s))
    v = _clamp01(float(v))
    i = int(h * 6.0)
    f = (h * 6.0) - i
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)
    i = i % 6
    if i == 0:
        return (v, t, p)
    if i == 1:
        return (q, v, p)
    if i == 2:
        return (p, v, t)
    if i == 3:
        return (p, q, v)
    if i == 4:
        return (t, p, v)
    return (v, p, q)

def _sample_albedo_linear(rng, *, value_min, value_max, saturation_max):
    # sample in sRGB-ish HSV, then convert to linear for shader inputs
    h = rng.random()
    s = rng.uniform(0.0, float(saturation_max))
    v = rng.uniform(float(value_min), float(value_max))
    rgb_srgb = _hsv_to_rgb(h, s, v)
    r, g, b = _rgb_srgb_to_linear(rgb_srgb)
    return (float(r), float(g), float(b), 1.0)

def _rand_log_uniform(rng, a, b):
    a = float(a)
    b = float(b)
    if a <= 0 or b <= 0 or b < a:
        return float(a)
    return math.exp(rng.uniform(math.log(a), math.log(b)))

def sample_material_params_for_object(glb_path):
    """
    返回 dict（会写入到 material.json），并保证：
      - per-object 固定（同一模型不同视角一致）
      - 可复现（由 MATERIAL_CONFIG['seed'] + glb_path 稳定哈希决定）
    """
    seed0 = int(MATERIAL_CONFIG.get("seed", 0) or 0)
    seed = (seed0 ^ _stable_int_from_string(os.path.abspath(glb_path))) & 0xFFFFFFFF
    rng = random.Random(seed)

    # 先决定材质类型（支持按比例混合）
    sampling_mode = (MATERIAL_CONFIG.get("type_sampling", "fixed") or "fixed").strip().lower()
    if sampling_mode == "weighted":
        weights = MATERIAL_CONFIG.get("type_weights") or {}
        # 过滤未知类型，避免拼写错误造成 silent-bug
        allowed = {
            "origin",
            "mirror",
            "glass",
            "glass_clear",
            "glass_frosted",
            "glass_tinted",
            "plastic",
            "ceramic",
            "metal_painted",
            "rubber",
        }
        items = [(str(k).strip().lower(), v) for k, v in weights.items() if str(k).strip().lower() in allowed]
        picked = _weighted_choice(rng, items)
        material_type = picked if picked else (MATERIAL_CONFIG.get("type", "mirror") or "mirror")
    else:
        material_type = (MATERIAL_CONFIG.get("type", "mirror") or "mirror")
    material_type = str(material_type).strip().lower()

    params = {
        "material_type": material_type,
        "randomize": bool(MATERIAL_CONFIG.get("randomize", False)),
        "seed": int(seed),
    }

    if not params["randomize"]:
        # 仅做“按比例挑材质族”的场景：不采样族内参数
        return params

    if material_type in ("glass", "glass_clear", "glass_frosted", "glass_tinted"):
        cfg = MATERIAL_CONFIG.get("random_glass") or {}

        ior = rng.uniform(float(cfg.get("ior_min", 1.47)), float(cfg.get("ior_max", 1.53)))
        params["ior"] = float(ior)

        if material_type == "glass_frosted":
            rmin = float(cfg.get("frosted_roughness_min", 0.10))
            rmax = float(cfg.get("frosted_roughness_max", 0.60))
            rough = rng.uniform(rmin, rmax)
        else:
            rmin = float(cfg.get("clear_roughness_min", 0.0005))
            rmax = float(cfg.get("clear_roughness_max", 0.02))
            rough = _rand_log_uniform(rng, rmin, rmax)
        params["roughness"] = float(rough)

        # distribution：仍由 MATERIAL_CONFIG 控制（你可以固定 GGX 或 BECKMANN）
        params["distribution"] = None

        # tint：glass_tinted 必开；glass_clear 可按概率开“极轻微”吸收
        use_tint = (material_type == "glass_tinted")
        if material_type in ("glass", "glass_clear") and not use_tint:
            p = float(cfg.get("allow_clear_tint_prob", 0.0))
            use_tint = (rng.random() < p)
        params["use_tint_volume"] = bool(use_tint)

        if params["use_tint_volume"]:
            dmin = float(cfg.get("tint_density_min", 0.01))
            dmax = float(cfg.get("tint_density_max", 0.15))
            density = rng.uniform(dmin, dmax)
            base = cfg.get("tint_base_rgb", (0.92, 0.97, 1.0))
            jitter = float(cfg.get("tint_jitter", 0.04))
            try:
                br, bg, bb = float(base[0]), float(base[1]), float(base[2])
            except Exception:
                br, bg, bb = 0.92, 0.97, 1.0
            tr = _clamp01(br + rng.uniform(-jitter, jitter))
            tg = _clamp01(bg + rng.uniform(-jitter, jitter))
            tb = _clamp01(bb + rng.uniform(-jitter, jitter))
            params["tint_color_rgba"] = (float(tr), float(tg), float(tb), 1.0)
            params["tint_density"] = float(density)

    elif material_type in ("plastic", "ceramic", "metal_painted", "rubber"):
        if material_type == "plastic":
            cfg = MATERIAL_CONFIG.get("random_plastic") or {}
            params["base_color_rgba"] = _sample_albedo_linear(
                rng,
                value_min=cfg.get("value_min", 0.15),
                value_max=cfg.get("value_max", 0.95),
                saturation_max=cfg.get("saturation_max", 0.55),
            )
            params["roughness"] = float(rng.uniform(cfg.get("roughness_min", 0.05), cfg.get("roughness_max", 0.60)))
            params["specular"] = float(rng.uniform(cfg.get("specular_min", 0.35), cfg.get("specular_max", 0.55)))
            params["clearcoat"] = float(rng.uniform(cfg.get("clearcoat_min", 0.0), cfg.get("clearcoat_max", 0.25)))
            params["clearcoat_roughness"] = float(rng.uniform(cfg.get("clearcoat_roughness_min", 0.0), cfg.get("clearcoat_roughness_max", 0.20)))
            params["metallic"] = 0.0

        elif material_type == "ceramic":
            cfg = MATERIAL_CONFIG.get("random_ceramic") or {}
            params["base_color_rgba"] = _sample_albedo_linear(
                rng,
                value_min=cfg.get("value_min", 0.40),
                value_max=cfg.get("value_max", 1.00),
                saturation_max=cfg.get("saturation_max", 0.30),
            )
            params["roughness"] = float(rng.uniform(cfg.get("roughness_min", 0.03), cfg.get("roughness_max", 0.45)))
            params["specular"] = float(rng.uniform(cfg.get("specular_min", 0.45), cfg.get("specular_max", 0.70)))
            params["clearcoat"] = float(rng.uniform(cfg.get("clearcoat_min", 0.0), cfg.get("clearcoat_max", 0.35)))
            params["clearcoat_roughness"] = float(rng.uniform(cfg.get("clearcoat_roughness_min", 0.0), cfg.get("clearcoat_roughness_max", 0.25)))
            params["metallic"] = 0.0

        elif material_type == "metal_painted":
            cfg = MATERIAL_CONFIG.get("random_metal_painted") or {}
            params["base_color_rgba"] = _sample_albedo_linear(
                rng,
                value_min=cfg.get("value_min", 0.20),
                value_max=cfg.get("value_max", 0.95),
                saturation_max=cfg.get("saturation_max", 0.70),
            )
            params["roughness"] = float(rng.uniform(cfg.get("roughness_min", 0.04), cfg.get("roughness_max", 0.35)))
            params["specular"] = float(rng.uniform(cfg.get("specular_min", 0.40), cfg.get("specular_max", 0.60)))
            params["clearcoat"] = float(rng.uniform(cfg.get("clearcoat_min", 0.20), cfg.get("clearcoat_max", 1.00)))
            params["clearcoat_roughness"] = float(rng.uniform(cfg.get("clearcoat_roughness_min", 0.0), cfg.get("clearcoat_roughness_max", 0.15)))
            params["metallic"] = 0.0

        elif material_type == "rubber":
            cfg = MATERIAL_CONFIG.get("random_rubber") or {}
            params["base_color_rgba"] = _sample_albedo_linear(
                rng,
                value_min=cfg.get("value_min", 0.02),
                value_max=cfg.get("value_max", 0.25),
                saturation_max=cfg.get("saturation_max", 0.20),
            )
            params["roughness"] = float(rng.uniform(cfg.get("roughness_min", 0.60), cfg.get("roughness_max", 0.95)))
            params["specular"] = float(rng.uniform(cfg.get("specular_min", 0.02), cfg.get("specular_max", 0.20)))
            params["metallic"] = 0.0
            params["clearcoat"] = 0.0
            params["clearcoat_roughness"] = 0.0
            params["sheen"] = float(rng.uniform(cfg.get("sheen_min", 0.0), cfg.get("sheen_max", 0.25)))
            params["sheen_tint"] = float(rng.uniform(cfg.get("sheen_tint_min", 0.0), cfg.get("sheen_tint_max", 0.50)))

    return params

def _build_object_dir_paths(base_dir):
    paths = {
        "train_img": os.path.join(base_dir, "train", "images"),
        "train_mask": os.path.join(base_dir, "train", "mask"),
        "test_img": os.path.join(base_dir, "test", "images"),
        "test_mask": os.path.join(base_dir, "test", "mask"),
    }
    for p in paths.values():
        os.makedirs(p, exist_ok=True)
    return paths

def _reset_principled_material_nodes(mat):
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    out_node = nodes.new(type="ShaderNodeOutputMaterial")
    out_node.location = (300, 0)

    bsdf = nodes.new(type="ShaderNodeBsdfPrincipled")
    bsdf.location = (0, 0)

    links.new(bsdf.outputs.get("BSDF"), out_node.inputs.get("Surface"))
    return bsdf, out_node

def _reset_glass_material_nodes(mat):
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    out_node = nodes.new(type="ShaderNodeOutputMaterial")
    out_node.location = (300, 0)

    glass = nodes.new(type="ShaderNodeBsdfGlass")
    glass.location = (0, 0)

    links.new(glass.outputs.get("BSDF"), out_node.inputs.get("Surface"))
    return glass, out_node

def apply_material(obj):
    """
    type:
      - origin: keep imported materials
      - mirror: override with a mirror-like Principled BSDF
      - glass_clear: override with clear glass Principled BSDF
      - glass_frosted: override with frosted glass Principled BSDF
      - glass_tinted: clear glass + Volume Absorption (tinted by thickness)
    """
    # 可选：支持 per-object 随机化参数（通过 obj 自定义属性传入）
    material_params = None
    try:
        raw = obj.get("_material_params_json", None)
        if raw:
            material_params = json.loads(raw)
    except Exception:
        material_params = None

    # 如果有 per-object 参数（随机/按比例选择），优先使用其中的 material_type
    material_type = None
    if material_params and material_params.get("material_type"):
        material_type = str(material_params.get("material_type")).strip().lower()
    if not material_type:
        material_type = (MATERIAL_CONFIG.get("type", "mirror") or "mirror").strip().lower()
    if material_type == "origin":
        return

    if material_type == "mirror":
        if material_params and material_params.get("randomize", False):
            mat = _new_material(f"Auto_Mirror_{obj.name}")
        else:
            mat = _get_or_create_principled_material("Auto_Mirror_Material")
        bsdf, _out = _reset_principled_material_nodes(mat)
        if "Base Color" in bsdf.inputs:
            bsdf.inputs["Base Color"].default_value = (1.0, 1.0, 1.0, 1.0)
        if "Metallic" in bsdf.inputs:
            bsdf.inputs["Metallic"].default_value = 1.0
        if "Roughness" in bsdf.inputs:
            bsdf.inputs["Roughness"].default_value = 0.0
    elif material_type in ("glass", "glass_clear", "glass_frosted", "glass_tinted"):
        if material_params and material_params.get("randomize", False):
            mat = _new_material(f"Auto_Glass_{obj.name}")
        else:
            mat = _get_or_create_principled_material("Auto_Glass_Material")
        glass, out_node = _reset_glass_material_nodes(mat)
        if "Color" in glass.inputs:
            glass.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
        if "Roughness" in glass.inputs:
            if material_type == "glass_frosted":
                if material_params and "roughness" in material_params:
                    glass.inputs["Roughness"].default_value = float(material_params["roughness"])
                else:
                    glass.inputs["Roughness"].default_value = float(MATERIAL_CONFIG.get("frosted_roughness", 0.2))
            else:
                if material_params and "roughness" in material_params:
                    glass.inputs["Roughness"].default_value = float(material_params["roughness"])
                else:
                    glass.inputs["Roughness"].default_value = 0.0
        if "IOR" in glass.inputs:
            if material_params and "ior" in material_params:
                glass.inputs["IOR"].default_value = float(material_params["ior"])
            else:
                glass.inputs["IOR"].default_value = float(MATERIAL_CONFIG.get("ior", 1.5))

        # 可选：设置 Glass BSDF 的分布类型（SHARP/GGX/BECKMANN）
        if MATERIAL_CONFIG.get("glass_apply_distribution", True):
            dist_key = "glass_distribution_frosted" if material_type == "glass_frosted" else "glass_distribution_clear"
            dist = (MATERIAL_CONFIG.get(dist_key) or "").strip().upper()
            if dist and hasattr(glass, "distribution"):
                try:
                    glass.distribution = dist
                except Exception:
                    pass

        use_tint = (material_type == "glass_tinted")
        if (not use_tint) and material_params:
            use_tint = bool(material_params.get("use_tint_volume", False))

        if use_tint:
            nodes = mat.node_tree.nodes
            links = mat.node_tree.links
            vol = nodes.new(type="ShaderNodeVolumeAbsorption")
            vol.location = (0, -260)
            if material_params and "tint_color_rgba" in material_params:
                c = material_params.get("tint_color_rgba")
            else:
                c = MATERIAL_CONFIG.get("tint_color_rgba", (0.85, 0.95, 1.0, 1.0))
            try:
                vol.inputs["Color"].default_value = (float(c[0]), float(c[1]), float(c[2]), float(c[3]))
            except Exception:
                vol.inputs["Color"].default_value = (0.85, 0.95, 1.0, 1.0)
            if material_params and "tint_density" in material_params:
                vol.inputs["Density"].default_value = float(material_params["tint_density"])
            else:
                vol.inputs["Density"].default_value = float(MATERIAL_CONFIG.get("tint_density", 0.5))
            links.new(vol.outputs.get("Volume"), out_node.inputs.get("Volume"))
    elif material_type in ("plastic", "ceramic", "metal_painted", "rubber"):
        if material_params and material_params.get("randomize", False):
            mat = _new_material(f"Auto_{material_type}_{obj.name}")
        else:
            mat = _get_or_create_principled_material(f"Auto_{material_type}_Material")

        bsdf, _out = _reset_principled_material_nodes(mat)

        if material_params and "base_color_rgba" in material_params and "Base Color" in bsdf.inputs:
            c = material_params["base_color_rgba"]
            try:
                bsdf.inputs["Base Color"].default_value = (float(c[0]), float(c[1]), float(c[2]), float(c[3]))
            except Exception:
                pass

        if material_params and "metallic" in material_params and "Metallic" in bsdf.inputs:
            bsdf.inputs["Metallic"].default_value = float(material_params["metallic"])
        elif "Metallic" in bsdf.inputs:
            bsdf.inputs["Metallic"].default_value = 0.0

        if material_params and "roughness" in material_params and "Roughness" in bsdf.inputs:
            bsdf.inputs["Roughness"].default_value = float(material_params["roughness"])

        if material_params and "specular" in material_params and "Specular IOR Level" in bsdf.inputs:
            bsdf.inputs["Specular IOR Level"].default_value = float(material_params["specular"])
        elif material_params and "specular" in material_params and "Specular" in bsdf.inputs:
            bsdf.inputs["Specular"].default_value = float(material_params["specular"])

        if material_params and "clearcoat" in material_params and "Coat Weight" in bsdf.inputs:
            bsdf.inputs["Coat Weight"].default_value = float(material_params["clearcoat"])
        elif material_params and "clearcoat" in material_params and "Clearcoat" in bsdf.inputs:
            bsdf.inputs["Clearcoat"].default_value = float(material_params["clearcoat"])

        if material_params and "clearcoat_roughness" in material_params and "Coat Roughness" in bsdf.inputs:
            bsdf.inputs["Coat Roughness"].default_value = float(material_params["clearcoat_roughness"])
        elif material_params and "clearcoat_roughness" in material_params and "Clearcoat Roughness" in bsdf.inputs:
            bsdf.inputs["Clearcoat Roughness"].default_value = float(material_params["clearcoat_roughness"])

        if material_type == "rubber":
            if material_params and "sheen" in material_params and "Sheen Weight" in bsdf.inputs:
                bsdf.inputs["Sheen Weight"].default_value = float(material_params["sheen"])
            elif material_params and "sheen" in material_params and "Sheen" in bsdf.inputs:
                bsdf.inputs["Sheen"].default_value = float(material_params["sheen"])
            if material_params and "sheen_tint" in material_params and "Sheen Tint" in bsdf.inputs:
                bsdf.inputs["Sheen Tint"].default_value = float(material_params["sheen_tint"])
    else:
        print(f"警告: 未知 material_type={material_type!r}，回退到 mirror")
        MATERIAL_CONFIG["type"] = "mirror"
        return apply_material(obj)

    if obj.data.materials:
        obj.data.materials.clear()
    obj.data.materials.append(mat)

    # Join 后可能残留多槽 material_index，清空槽后需要强制归零，否则会出现“看起来没替换/发黑”等异常。
    mesh = getattr(obj, "data", None)
    polys = getattr(mesh, "polygons", None) if mesh else None
    if polys:
        for p in polys:
            p.material_index = 0

def import_and_process_glb(file_path, anchor_obj):
    if not anchor_obj: return None
    
    # 1. 解析 Anchor 的几何信息 (世界坐标系)
    anchor_loc = anchor_obj.matrix_world.translation
    anchor_dim_x = 2.0 * ANCHOR_HALF_SIZE_M
    anchor_dim_y = 2.0 * ANCHOR_HALF_SIZE_M
    anchor_dim_z = 2.0 * ANCHOR_HALF_SIZE_M
    # Anchor 原点在立方体中心：地面 = center_z - half_size
    anchor_floor_z = anchor_loc.z - ANCHOR_HALF_SIZE_M

    print(f"  -> 正在导入: {os.path.basename(file_path)} ...")
    
    # 禁用撤销、导入GLB
    bpy.context.preferences.edit.use_global_undo = False
    try:
        bpy.ops.import_scene.gltf(filepath=file_path)
    except Exception as e:
        print(f"  -> GLB导入出错: {e}")
        return None
    finally:
        bpy.context.preferences.edit.use_global_undo = True

    # 筛选 Mesh
    imported_obs = bpy.context.selected_objects
    mesh_obs = [o for o in imported_obs if o.type == 'MESH']
    if not mesh_obs:
        bpy.ops.object.delete()
        return None

    # 烘焙变换: transform world matrix
    if bpy.context.active_object and bpy.context.active_object.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')
    
    for obj in mesh_obs:
        world_matrix = obj.matrix_world.copy()
        obj.data.transform(world_matrix)
        obj.matrix_world = Matrix.Identity(4)
        obj.parent = None

    # 清理与合并
    bpy.ops.object.select_all(action='DESELECT')
    for o in imported_obs:
        if o.type != 'MESH':
            o.select_set(True)
    bpy.ops.object.delete()

    bpy.ops.object.select_all(action='DESELECT')
    for obj in mesh_obs:
        obj.select_set(True)
    final_obj = mesh_obs[0]
    bpy.context.view_layer.objects.active = final_obj
    if len(mesh_obs) > 1:
        bpy.ops.object.join()
    final_obj.name = "Imported_Artifact"

    # ==========================================================================
    #                  新逻辑：基于 Anchor 体积的缩放与对齐
    # ==========================================================================
    
    # 1. 归一化原点到几何中心
    bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')
    
    # 2. 将物体移动到 Anchor 的 X,Y 中心，但 Z 轴暂时不动
    final_obj.location.x = anchor_loc.x
    final_obj.location.y = anchor_loc.y
    final_obj.location.z = anchor_loc.z # 先放中间，稍后移动
    bpy.context.view_layer.update()

    # 3. 智能缩放：让物体“塞进”Anchor 的体积内
    # 获取物体当前的尺寸（Axis-aligned bounding box）
    obj_dim = final_obj.dimensions
    
    # 计算三个轴向的缩放比例，找出最受限的那个轴
    # 可留余量，如果物体顶满格子不好看
    margin = 1.0
    
    # 防止除以0错误
    scale_x = (anchor_dim_x * margin) / obj_dim.x if obj_dim.x > 0 else 1
    scale_y = (anchor_dim_y * margin) / obj_dim.y if obj_dim.y > 0 else 1
    scale_z = (anchor_dim_z * margin) / obj_dim.z if obj_dim.z > 0 else 1
    
    # 使用最小的缩放比例，保持物体长宽比不变 (Uniform Scale)
    target_scale = min(scale_x, scale_y, scale_z)
    
    final_obj.scale *= target_scale
    bpy.ops.object.transform_apply(scale=True)
    
    # 4. 最终落地：对齐 Anchor 的底面
    bpy.context.view_layer.update()
    
    # 计算物体当前的世界坐标最低点
    world_corners = [final_obj.matrix_world @ Vector(corner) for corner in final_obj.bound_box]
    obj_min_z = min(corner.z for corner in world_corners)
    
    # 计算需要下降/上升多少距离才能碰到 Anchor 的地板
    drop_distance = anchor_floor_z - obj_min_z
    final_obj.location.z += drop_distance

    print(f"  -> 已适配 Anchor 尺寸: 缩放倍率 {target_scale:.4f}, 落地位移 {drop_distance:.4f}")

    # ==========================================================================

    # per-object 材质随机：把采样结果挂在 object 上，apply_material() 会读取并创建独立材质实例
    try:
        params = sample_material_params_for_object(file_path)
        final_obj["_material_params_json"] = json.dumps(params, ensure_ascii=False)
    except Exception:
        pass

    apply_material(final_obj)
    final_obj.pass_index = 1
    
    return final_obj

# ==============================================================================
#                       Blender 5.0 合成器设置
# ==============================================================================

def setup_compositor_blender_5(scene):
    # 开启 Object Index 通道
    if scene.view_layers:
        scene.view_layers[0].use_pass_object_index = True
    
    if scene.compositing_node_group:
        tree = scene.compositing_node_group
        tree.nodes.clear()
    else:
        tree = bpy.data.node_groups.new("NeRF Compositing", "CompositorNodeTree")
        scene.compositing_node_group = tree

    # 1. 基础节点
    rlayers = tree.nodes.new(type="CompositorNodeRLayers")
    rlayers.location = (-400, 0)
    
    # 2. ID Mask 节点
    id_mask = tree.nodes.new(type="CompositorNodeIDMask")
    id_mask.location = (-200, 0)
    
    # [核心修复] 'index' 属性变成了输入端口
    # 我们设置端口的默认值，而不是设置节点属性
    # ID Mask 的输入通常有两个：0是数据流，1是Index值(我们这里设为1)
    if 'Index' in id_mask.inputs:
        id_mask.inputs['Index'].default_value = 1
    else:
        # 防御性写法：如果名字变了，尝试找第二个输入端口
        # 通常 inputs[0] 是 "ID value" (连线用)，inputs[1] 是 "Index" (数值用)
        if len(id_mask.inputs) > 1:
            id_mask.inputs[1].default_value = 1
            
    # 抗锯齿通常还是属性，如果报错再说，但在 5.0 中有些布尔值也变 Input 了
    # 这里先尝试属性，如果它是 Input，通常叫 'Anti-Aliasing'
    if hasattr(id_mask, 'use_antialiasing'):
        id_mask.use_antialiasing = True
    elif 'Anti-Aliasing' in id_mask.inputs:
        id_mask.inputs['Anti-Aliasing'].default_value = 1
    
    # 3. 输出节点
    f_out = tree.nodes.new(type="CompositorNodeOutputFile")
    f_out.location = (0, 0)
    f_out.label = "Mask_Output"
    
    # [关键流程] 解锁 EXR -> 切换 IMAGE -> 设置 PNG
    f_out.file_output_items.clear()
    f_out.format.media_type = 'IMAGE'
    f_out.format.file_format = 'PNG'
    f_out.format.color_mode = 'RGB'
    f_out.format.color_depth = '8'

    # 插槽设置
    mask_item = f_out.file_output_items.new(name="mask", socket_type='RGBA')
    mask_item.override_node_format = True
    mask_item.format.file_format = 'PNG'
    mask_item.format.color_mode = 'RGB'
    mask_item.format.color_depth = '8'
    
    # [色彩管理] 强制 Standard
    try:
        mask_item.format.color_management = 'OVERRIDE'
        mask_item.format.view_settings.view_transform = 'Standard'
    except: pass
    
    # 4. 连线逻辑
    # 连线：RenderLayers [IndexOB] -> ID Mask [ID value]
    # 注意：ID Mask 的第一个输入口（inputs[0]）是接收数据的
    idx_output = rlayers.outputs.get('IndexOB')
    if not idx_output:
        # 尝试模糊匹配，防止 5.0 改名
        for out in rlayers.outputs:
            if 'Index' in out.name:
                idx_output = out
                break
    
    if idx_output:
        tree.links.new(idx_output, id_mask.inputs[0])
    else:
        print(f"警告: 无法在渲染层找到 Index 输出，Mask 可能全黑。可用输出: {[o.name for o in rlayers.outputs]}")

    tree.links.new(id_mask.outputs['Alpha'], f_out.inputs["mask"])
    
    # 5. 常规 RGB 输出
    comp_node = tree.nodes.new(type='NodeGroupOutput')
    comp_node.location = (0, 200)
    
    if len(tree.interface.items_tree) == 0:
        tree.interface.new_socket(name="Image", in_out='OUTPUT', socket_type='NodeSocketColor')
    
    tree.links.new(rlayers.outputs['Image'], comp_node.inputs['Image'])
    
    return f_out
# ==============================================================================
#                          渲染与相机
# ==============================================================================

def setup_render_settings():
    scene = bpy.context.scene
    c = scene.cycles
    
    # 1. 设置渲染引擎
    scene.render.engine = RENDER_CONFIG["engine"]
    
    # 2. 获取 Cycles 偏好设置
    prefs = bpy.context.preferences.addons['cycles'].preferences
    
    # 3. 设置计算后端 (优先使用 OPTIX)
    backend = RENDER_CONFIG.get("gpu_backend", "OPTIX")
    try:
        prefs.compute_device_type = backend
    except:
        print(f"警告: 无法设置 {backend}，回退到 CUDA")
        prefs.compute_device_type = "CUDA"
        backend = "CUDA"

    # 4. 刷新设备列表
    prefs.get_devices()
    
    # 5. 智能筛选 GPU
    available_gpus = []
    for d in prefs.devices:
        if d.type == backend:
            available_gpus.append(d)
            
    if not available_gpus and backend == 'OPTIX':
        print("未找到 OPTIX 设备，尝试寻找 CUDA 设备...")
        for d in prefs.devices:
            if d.type == 'CUDA':
                available_gpus.append(d)

    print(f"--- 正在配置 GPU ({prefs.compute_device_type}) ---")
    
    # 6. 应用 GPU 选择逻辑
    gpu_indices = RENDER_CONFIG.get("gpu_indices", [])
    
    # 先禁用所有设备
    for device in prefs.devices:
        device.use = False

    active_devices = []
    for i, device in enumerate(available_gpus):
        if not gpu_indices or i in gpu_indices:
            device.use = True
            active_devices.append(f"GPU_{i}: {device.name}")
            print(f"  [√] 已启用: {device.name}")
        else:
            print(f"  [x] 已忽略: {device.name}")
            
    # 7. 设置场景渲染参数 (科研级配置)
    c.device = 'GPU'
    scene.render.film_transparent = False
    
    # === [关键修复] 降噪设置 ===
    # 使用 .get() 防止 KeyError，且如果禁用降噪，就不需要设置类型
    use_denoising = RENDER_CONFIG.get("use_denoising", False)
    c.use_denoising = use_denoising
    
    if use_denoising:
        # 只有开启时才尝试读取类型，默认回退到 OIDN
        c.denoiser = RENDER_CONFIG.get("denoiser_type", 'OPENIMAGEDENOISE')
    
    if hasattr(c, 'use_light_tree'):
        # 如果配置里没写，默认也给它关了(False)，保证速度
        should_use_tree = RENDER_CONFIG.get("use_light_tree", False)
        c.use_light_tree = should_use_tree
        print(f"  -> Light Tree (光树) 状态: {'开启' if should_use_tree else '关闭'}")
    
    # === 采样设置 ===
    # [科研级一致性] 关闭自适应采样，避免不同帧/不同视角提前停导致的质量波动
    if hasattr(c, "use_adaptive_sampling"):
        c.use_adaptive_sampling = False

    c.samples = RENDER_CONFIG.get("samples_max", 4096)

    # === [新增] 光线钳制 (Clamping) ===
    # 防止大场景中的萤火虫噪点
    c.sample_clamp_direct = RENDER_CONFIG.get("clamp_direct", 0)
    c.sample_clamp_indirect = RENDER_CONFIG.get("clamp_indirect", 10.0)

    # === [新增] 光程反弹 (Light Paths) ===
    # 防止玻璃/复杂几何体变黑
    lp = RENDER_CONFIG.get("light_paths", {})
    if lp:
        c.max_bounces = lp.get("max_bounces", 32)
        c.diffuse_bounces = lp.get("diffuse_bounces", 16)
        c.glossy_bounces = lp.get("glossy_bounces", 16)
        c.transparent_max_bounces = lp.get("transparent_max", 32)
        c.transmission_bounces = lp.get("transmission", 16)

    # === [新增] 焦散 (Caustics) ===
    # 真实向：允许折射/反射焦散（会更难收敛）
    if hasattr(c, "caustics_reflective"):
        c.caustics_reflective = bool(RENDER_CONFIG.get("caustics_reflective", True))
    if hasattr(c, "caustics_refractive"):
        c.caustics_refractive = bool(RENDER_CONFIG.get("caustics_refractive", True))
    if hasattr(c, "blur_glossy"):
        c.blur_glossy = float(RENDER_CONFIG.get("blur_glossy", 0.0))

    # === [新增] 色彩管理 (Color Management) ===
    # 强制线性工作流，避免 Filmic 改变光照强度
    if hasattr(scene.view_settings, 'view_transform'):
        scene.view_settings.view_transform = 'Standard'
        scene.view_settings.look = 'None'
        scene.view_settings.exposure = 0.0
        scene.view_settings.gamma = 1.0

    # 8. 分辨率设置
    scene.render.resolution_x = RENDER_CONFIG["res_x"]
    scene.render.resolution_y = RENDER_CONFIG["res_y"]
    scene.render.resolution_percentage = RENDER_CONFIG["res_percent"]
    
    adaptive_flag = getattr(c, "use_adaptive_sampling", None)
    print(f"--- 渲染设置完成: Samples={c.samples}, Adaptive={adaptive_flag}, Denoise={c.use_denoising} ---\n")

def create_smart_camera(anchor_obj):
    scene = bpy.context.scene
    if "UltraCam" in scene.objects:
        bpy.data.objects.remove(scene.objects["UltraCam"], do_unlink=True)

    cam_data = bpy.data.cameras.new(name='UltraCam')
    cam_obj = bpy.data.objects.new(name='UltraCam', object_data=cam_data)
    scene.collection.objects.link(cam_obj)
    scene.camera = cam_obj
    cam_data.lens = LOGIC_CONFIG["lens"]

    target_loc = anchor_obj.location if anchor_obj else Vector((0,0,0))
    if "FocusTarget" not in scene.objects:
        empty = bpy.data.objects.new("FocusTarget", None)
        scene.collection.objects.link(empty)
    target_obj = scene.objects["FocusTarget"]
    target_obj.location = target_loc 

    constraint = cam_obj.constraints.new(type='TRACK_TO')
    constraint.target = target_obj
    constraint.track_axis = 'TRACK_NEGATIVE_Z'
    constraint.up_axis = 'UP_Y'

    return cam_obj

def get_nerf_matrix(cam_obj):
    return list(map(list, cam_obj.matrix_world))

# ==============================================================================
#                          Main Loop
# ==============================================================================

def main():
    print(f"材质模式: {MATERIAL_CONFIG.get('type')}")

    if not os.path.exists(CONFIG_PATHS["output_dir"]):
        os.makedirs(CONFIG_PATHS["output_dir"])

    scene_source = str(CONFIG_PATHS.get("scene_source", "blend")).strip().lower()
    scene_entries = []
    if scene_source == "front_room":
        room_entries = resolve_front_room_dirs(CONFIG_PATHS.get("front_room_root"))
        if not room_entries:
            print(f"错误: 找不到任何可渲染房间目录（需包含 meshes.obj）: {CONFIG_PATHS.get('front_room_root')}")
            return
        scene_entries = [("front_room", room_dir, scene_name) for room_dir, scene_name in room_entries]
    else:
        blend_files = resolve_blend_files(CONFIG_PATHS.get("blend_path") or CONFIG_PATHS.get("blend_file"))
        if not blend_files:
            print(f"错误: 找不到任何 .blend/.blender 文件: {CONFIG_PATHS.get('blend_path') or CONFIG_PATHS.get('blend_file')}")
            return
        scene_entries = [("blend", bf, os.path.splitext(os.path.basename(bf))[0]) for bf in blend_files]

    override_hashes = LOGIC_CONFIG.get("object_hashes") or []
    selected_files = resolve_object_glb_files(CONFIG_PATHS["objects_dir"], override_hashes)
    if selected_files:
        print(f"使用指定 GLB 列表渲染: {len(selected_files)} 个（已跳过大小/抽样逻辑）")
    else:
        selected_files = filter_files_by_size_and_sample(
            CONFIG_PATHS["objects_dir"],
            min_mb=LOGIC_CONFIG["min_size_mb"],
            max_mb=LOGIC_CONFIG["max_size_mb"],
            count=LOGIC_CONFIG["target_count"],
        )
    if not selected_files:
        print("无文件符合条件。")
        return

    for s_idx, (src_type, src_path, scene_name) in enumerate(scene_entries):
        print(f"\n=== [{s_idx+1}/{len(scene_entries)}] 载入场景: {src_path} ===")
        if src_type == "blend":
            bpy.ops.wm.open_mainfile(filepath=src_path)
            if CONFIG_PATHS["anchor_name"] in bpy.context.scene.objects:
                anchor = bpy.context.scene.objects[CONFIG_PATHS["anchor_name"]]
            else:
                anchor = bpy.data.objects.new("TempAnchor", None)
                bpy.context.scene.collection.objects.link(anchor)
        else:
            anchor = prepare_scene_from_front_room(src_path)
            if not anchor:
                print("  -> 跳过：房间构建失败")
                continue
        setup_world_hdri(bpy.context.scene)

        setup_render_settings()
        cam_obj = create_smart_camera(anchor)
        mask_node = setup_compositor_blender_5(bpy.context.scene)

        TARGET_RADIUS = LOGIC_CONFIG["target_diameter"] / 2.0
        MARGIN = LOGIC_CONFIG["margin"]

        scene = bpy.context.scene
        fov_h = cam_obj.data.angle
        ar = scene.render.resolution_x / scene.render.resolution_y
        fov_v = 2 * math.atan(math.tan(fov_h / 2) / ar)
        safe_distance_3d = (TARGET_RADIUS * MARGIN) / math.sin(min(fov_h, fov_v) / 2)

        camera_positions_local = generate_fibonacci_points(
            n_samples=LOGIC_CONFIG["num_views"],
            radius=safe_distance_3d,
            center_loc=Vector((0, 0, 0)),
            hemisphere=LOGIC_CONFIG["use_hemisphere"],
        )

        scene_output_root = os.path.join(CONFIG_PATHS["output_dir"], scene_name) if len(scene_entries) > 1 else CONFIG_PATHS["output_dir"]
        os.makedirs(scene_output_root, exist_ok=True)

        print(f"\n=== 开始渲染 {len(selected_files)} 个模型 (scene={scene_name}) ===")

        sweep_cfg = (LOGIC_CONFIG.get("material_sweep") or {})
        sweep_enable = bool(sweep_cfg.get("enable", False))
        sweep_only_first = True  # 强制只对第 1 个 glb 做 sweep，避免组数爆炸
        sweep_types = list(sweep_cfg.get("types") or [])

        for i, (glb_path, size_mb) in enumerate(selected_files):
            model_name = os.path.splitext(os.path.basename(glb_path))[0]
            print(f"[{i+1}/{len(selected_files)}] 处理: {model_name}")

            final_obj = import_and_process_glb(glb_path, anchor)
            if not final_obj:
                continue

            def render_one_material_variant(variant_name):
                base_dir = os.path.join(scene_output_root, variant_name)
                paths = _build_object_dir_paths(base_dir)

                # 写入每个模型的材质参数（如启用随机材质/按比例选材质族，方便复现与审计）
                try:
                    raw = final_obj.get("_material_params_json", "")
                    if raw:
                        with open(os.path.join(base_dir, "material.json"), "w", encoding="utf-8") as f:
                            f.write(raw)
                except Exception:
                    pass

                transforms = {
                    "train": {"camera_angle_x": cam_obj.data.angle, "frames": []},
                    "test": {"camera_angle_x": cam_obj.data.angle, "frames": []},
                }

                for v_idx, pos_local in enumerate(camera_positions_local):
                    cam_obj.location = pos_local + anchor.location
                    bpy.context.view_layer.update()

                    is_test = (v_idx % LOGIC_CONFIG["test_every"] == 0) and (v_idx > 0)
                    subset = "test" if is_test else "train"

                    fname_no_ext = f"{v_idx:03d}"
                    fname_png = f"{fname_no_ext}.png"

                    # 1. Image 路径
                    img_dir = paths[f"{subset}_img"]
                    scene.render.filepath = os.path.join(img_dir, fname_png)

                    # 2. Mask 路径
                    mask_dir = paths[f"{subset}_mask"]
                    mask_node.directory = mask_dir
                    mask_node.file_name = fname_no_ext

                    bpy.ops.render.render(write_still=True)

                    frame_data = {
                        "file_path": f"./images/{fname_png}",
                        "transform_matrix": get_nerf_matrix(cam_obj),
                    }
                    transforms[subset]["frames"].append(frame_data)

                    # 移动重命名 Mask (确保是 000.png 而非 000mask.png)
                    candidates = glob.glob(os.path.join(mask_dir, f"{fname_no_ext}*.png"))
                    target_path = os.path.join(mask_dir, fname_png)
                    for c in candidates:
                        if c != target_path:
                            try:
                                shutil.move(c, target_path)
                            except:
                                pass

                for subset in ["train", "test"]:
                    with open(os.path.join(base_dir, f"transforms_{subset}.json"), "w") as f:
                        json.dump(transforms[subset], f, indent=4)

            if sweep_enable and sweep_types:
                if sweep_only_first and i > 0:
                    print("  -> material_sweep: 仅对第 1 个 glb 做 sweep，已跳过后续模型")
                else:
                    # 依次对同一个对象重设材质并渲染（避免重复导入模型）
                    old_sampling = MATERIAL_CONFIG.get("type_sampling")
                    old_type = MATERIAL_CONFIG.get("type")
                    try:
                        for t in sweep_types:
                            t = str(t).strip().lower()
                            if not t:
                                continue
                            # 强制固定选择该类型
                            MATERIAL_CONFIG["type_sampling"] = "fixed"
                            MATERIAL_CONFIG["type"] = t

                            try:
                                params = sample_material_params_for_object(glb_path)
                                final_obj["_material_params_json"] = json.dumps(params, ensure_ascii=False)
                            except Exception:
                                pass
                            apply_material(final_obj)

                            variant_name = f"{model_name}__{t}"
                            print(f"  -> material_sweep: {variant_name}")
                            render_one_material_variant(variant_name)
                    finally:
                        MATERIAL_CONFIG["type_sampling"] = old_sampling
                        MATERIAL_CONFIG["type"] = old_type
            else:
                render_one_material_variant(model_name)

            bpy.data.objects.remove(final_obj, do_unlink=True)
            for m in bpy.data.meshes:
                if m.users == 0:
                    bpy.data.meshes.remove(m)
            for m in bpy.data.materials:
                if m.users == 0:
                    bpy.data.materials.remove(m)

    print("=== 完成 ===")

if __name__ == "__main__":
    main()
