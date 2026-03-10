import os
try:
    import igl
except Exception:
    igl = None
import math
import numpy as np
import urllib.request
import shutil
import hashlib

def rotation_matrix(axis, theta):
    axis = np.asarray(axis)
    axis = axis / math.sqrt(np.dot(axis, axis))
    a = math.cos(theta / 2.0)
    b, c, d = -axis * math.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])

def quaternion_to_matrix(args):
    """
    Quaternion to matrix
    :param args:
    :return:
    """
    tx = args[0] + args[0]
    ty = args[1] + args[1]
    tz = args[2] + args[2]
    twx = tx * args[3]
    twy = ty * args[3]
    twz = tz * args[3]
    txx = tx * args[0]
    txy = ty * args[0]
    txz = tz * args[0]
    tyy = ty * args[1]
    tyz = tz * args[1]
    tzz = tz * args[2]

    result = np.zeros((3, 3))
    result[0, 0] = 1.0 - (tyy + tzz)
    result[0, 1] = txy - twz
    result[0, 2] = txz + twy
    result[1, 0] = txy + twz
    result[1, 1] = 1.0 - (txx + tzz)
    result[1, 2] = tyz - twx
    result[2, 0] = txz - twy
    result[2, 1] = tyz + twx
    result[2, 2] = 1.0 - (txx + tyy)
    return result

def vector_dot_matrix3(v, mat):
    rot_mat = np.mat(mat)
    vec = np.mat(v).T
    result = np.dot(rot_mat, vec)
    return np.array(result.T)[0]

def transform_v(v, instance):
    
    pos = instance.pos
    if hasattr(instance,'rotscale'):
        v = np.dot(v, np.array(instance.rotscale))
    else:
        rot = instance.rot
        scale = instance.scale
        v = v.astype(np.float64) * scale
        rotMatrix = quaternion_to_matrix(rot)
        R = np.array(rotMatrix)
        v = np.transpose(v)
        v = np.matmul(R, v)
        v = np.transpose(v)
    v = v + pos
    return v

def read_mesh_attr(mesh):
    v = mesh.xyz
    faces = mesh.faces
    vt = None
    mat = None
    if mesh.uv is not None:
        # Some 3D-FRONT meshes contain invalid UV entries (e.g. None).
        # Keep geometry export robust by dropping broken UVs for that mesh.
        try:
            vt = np.asarray(mesh.uv, dtype=np.float64)
            if vt.ndim != 2 or vt.shape[1] < 2:
                vt = None
            else:
                vt = vt[:, :2]
        except Exception:
            vt = None

        if mesh.material is not None:
            mat = mesh.material
            if vt is not None and mat.UVTransform is not None:
                try:
                    uv_m = np.asarray(mat.UVTransform, dtype=np.float64)
                    vt = uv_m[:2, :2] @ vt.T
                    vt = vt.T
                except Exception as e:
                    print(
                        f"[json2obj][invalid] mesh_uid={getattr(mesh, 'uid', None)} "
                        f"instanceid={getattr(mesh, 'instanceid', None)} "
                        f"material_uid={getattr(mat, 'uid', None)} "
                        f"material_jid={getattr(mat, 'jid', None)} "
                        f"uv_transform_apply_failed err={e}"
                    )
                    vt = None
    return v,faces,vt, mat
def quadrilateral2triangle(filepath):

    dir_path = os.path.dirname(filepath)

    objpath = os.path.join(dir_path, 'raw_model.obj')
    tri_obj = os.path.join(dir_path, 'raw_model_tri.obj')

    fid_obj = open(objpath, 'r', encoding='utf-8',errors='ignore')
    fid_tri_obj = open(tri_obj, 'w', encoding='utf-8')
    alllines = fid_obj.readlines()

    for line in alllines:
        if line.startswith('f  '):
            tri = line.split('  ')[1].split(' ')
            tri = [t for t in tri if t != '\n']
        elif line.startswith('f '):
            tri = line.split(' ')[1:]
            tri = [t for t in tri if t != '\n']
        if (line.startswith('f ') or line.startswith('f  ')) and len(tri) >= 4:
            fid_tri_obj.write('f ' + tri[0] + ' ' + tri[1] + ' ' + tri[2] + '\n')
            fid_tri_obj.write('f ' + tri[0] + ' ' + tri[2] + ' ' + tri[3] + '\n')
        else:
            fid_tri_obj.write(line)
    fid_obj.close()
    fid_tri_obj.close()

def _localize_texture(texture_path, dst_dir, prefix='tex'):
    """
    Copy texture to destination directory and return local filename.
    Returns None if source texture is unavailable.
    """
    if not texture_path:
        return None
    src = str(texture_path).strip()
    if not src:
        return None

    if not os.path.isabs(src):
        src = os.path.abspath(src)
    if not os.path.isfile(src):
        return None

    base = os.path.basename(src)
    key = hashlib.md5(src.encode('utf-8', errors='ignore')).hexdigest()[:8]
    local_name = f"{prefix}_{key}_{base}"
    dst = os.path.join(dst_dir, local_name)
    if not os.path.exists(dst):
        shutil.copy2(src, dst)
    return local_name

def _parse_map_texture_path(line):
    """
    Parse texture path from a map_K* line.
    MTL options are ignored; we take the last token as file path.
    """
    s = line.strip()
    parts = s.split(None, 1)
    if len(parts) < 2:
        return None
    raw = parts[1].strip()
    if not raw:
        return None
    return raw.split()[-1]

def save_obj(ori_obj_path, savepath, v, model_id):

    obj_dir_path = os.path.dirname(ori_obj_path)
    save_dir = os.path.dirname(savepath)

    objpath = ori_obj_path
    tri_obj = savepath

    fid_obj = open(objpath, 'r', encoding='utf-8',errors='ignore')
    fid_save_obj = open(tri_obj, 'w', encoding='utf-8')
    alllines = fid_obj.readlines()
    idx = 0
    for line in alllines:
        if line.startswith('mtllib model.mtl'):
            line = 'mtllib ' + model_id + '.mtl\n'
        if (line.startswith('v ') or line.startswith('v  ')):
            fid_save_obj.write('v ' + str(v[idx][0]) + ' ' + str(v[idx][1]) + ' ' + str(v[idx][2]) + '\n')
            idx = idx + 1
        else:
            fid_save_obj.write(line)

    fid_obj.close()
    fid_save_obj.close()


    if os.path.exists(save_dir+'/'+model_id+'.mtl'):
        return


    src_mtl = os.path.join(obj_dir_path, 'model.mtl')
    dst_mtl = os.path.join(save_dir, model_id + '.mtl')
    with open(src_mtl, 'r', encoding='utf-8', errors='ignore') as fid_obj, \
         open(dst_mtl, 'w', encoding='utf-8') as fid_save_obj:
        for line in fid_obj:
            s = line.strip()
            if s.startswith('map_Ka '):
                # Blender OBJ importer often warns that map_Ka is unsupported.
                continue
            if s.startswith('map_Kd '):
                tex_token = _parse_map_texture_path(line)
                tex_abs = None
                if tex_token:
                    tex_abs = tex_token if os.path.isabs(tex_token) else os.path.normpath(os.path.join(obj_dir_path, tex_token))
                local_tex = _localize_texture(tex_abs, save_dir, prefix=model_id) if tex_abs else None
                if local_tex is not None:
                    fid_save_obj.write('map_Kd ' + local_tex + '\n')
                else:
                    # fallback: keep original line if source texture cannot be resolved.
                    fid_save_obj.write(line)
                continue
            fid_save_obj.write(line)


def read_obj(filepath):
    if not os.path.exists(filepath):
        quadrilateral2triangle(filepath)
    # libigl Python API differs across versions:
    # - some expose read_obj
    # - some expose readOBJ
    # If both are unavailable, fall back to a lightweight OBJ vertex parser.
    if igl is not None:
        if hasattr(igl, "read_obj"):
            v, vt, _, faces, ftc, _ = igl.read_obj(filepath)
            return v
        if hasattr(igl, "readOBJ"):
            try:
                data = igl.readOBJ(filepath)
                # Newer bindings may return a tuple-like object.
                if isinstance(data, (list, tuple)) and len(data) > 0:
                    return np.asarray(data[0])
            except Exception:
                pass

    verts = []
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line.startswith("v "):
                continue
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            try:
                verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
            except Exception:
                continue
    return np.asarray(verts, dtype=np.float64)


def save_mesh(savepath, filename, *args):
    fid = open(savepath+'/'+str(filename)+'.obj', "w")

    mtl_fid = open(savepath+'/'+str(filename)+'.mtl', "w")
    fid.write('mtllib '+str(filename)+'.mtl\n')
    v_id = 1
    vt_id = 1
    for mesh in args:
        mesh_id, v, faces, vt, texture = mesh
        
        for vi in v:
            fid.write('v %f %f %f\n' % (vi[0], vi[1], vi[2]))
        if type(texture) != list:
            for vti in vt:
                fid.write('vt %f %f\n' % (vti[0], vti[1]))
            
        
        mtl_fid.write('newmtl '+str(mesh_id)+'\n')
        if type(texture) == list:
            mtl_fid.write('Kd '+str(texture[0]/255.)+' '+str(texture[1]/255.)+' '+str(texture[2]/255.)+'\n')
        else:
            local_tex = _localize_texture(texture, savepath, prefix='mesh')
            if local_tex is not None:
                mtl_fid.write('map_Kd '+local_tex+'\n')
            else:
                # Keep original path only if source texture is missing.
                mtl_fid.write('map_Kd '+str(texture)+'\n')
        fid.write('usemtl '+str(mesh_id)+'\n')
        for f in faces:
            if type(texture) == list:
                fid.write('f %d %d %d\n' % (f[0]+v_id,f[1]+v_id,f[2]+v_id))
            else:
                fid.write('f %d/%d %d/%d %d/%d\n' % (f[0]+v_id,f[0]+vt_id,f[1]+v_id,f[1]+vt_id,f[2]+v_id,f[2]+vt_id))
        if type(texture) != list:
            vt_id = vt_id + vt.shape[0]
        v_id = v_id + v.shape[0]
    fid.close()
    mtl_fid.close()

def read_mesh(mesh, meshes, material, save_path, floor_path, fid_error):
    for constructid, all_mesh in mesh.items():
        output = []
        mesh_type = 'walls'
        for uid in all_mesh:
            v, faces, vt, mat = read_mesh_attr(meshes[uid], material)
            texture = None
            if vt is not None and mat is not None:
                if mat.texture != '':
                    if not os.path.exists(floor_path+'/' + mat.jid):
                        os.mkdir(floor_path+'/' + mat.jid)
                        try:
                            urllib.request.urlretrieve(mat.texture,floor_path+'/' + mat.jid +'/texture.png')
                        except:
                            fid_error.write(meshes[uid].room_id + ': load '+ mat.texture +' texture failed\n')

                    texture = floor_path+'/' + mat.jid + '/texture.png'
            output.append([meshes[uid].instanceid, v,faces,vt,texture])
        
        save_mesh(save_path, mesh_type+'_'+constructid, *output)
