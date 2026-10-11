import bpy
import json
from mathutils import Vector, Matrix, Quaternion

# ============================================================
# 1. CONFIGURACIÓN
# ============================================================
JSON_PATH      = r"C:\Users\gabo0\Downloads\cuban_3d.json"
ARMATURE_NAME  = "MP_Armature"   # esqueleto generado por el script anterior
DEPTH_SCALE    = 1             # el eje Y del JSON viene exagerado (~3x)
SMOOTH_ALPHA   = 1            # suavizado de landmarks (1 = sin suavizar)
ROOT_MOTION    = True            # traslada la cadera según el movimiento capturado
ROOT_SCALE     = 1.0

# ============================================================
# 2. TABLAS DE LANDMARKS Y HUESOS
# ============================================================
SIDES = {
    ".L": dict(sh=11, el=13, wr=15, pk=17, ix=19, hip=23, kn=25, an=27, toe=31, ear=7),
    ".R": dict(sh=12, el=14, wr=16, pk=18, ix=20, hip=24, kn=26, an=28, toe=32, ear=8),
}
# Padres siempre antes que sus hijos (el orden importa al resolver la pose)
ORDER = ["Hips", "Spine", "Chest", "Neck", "Head"]
for _s in SIDES:
    ORDER += [b + _s for b in ("Shoulder", "UpperArm", "LowerArm", "Hand",
                               "Thigh", "Shin", "Foot")]

Y_AXIS   = Vector((0, 1, 0))
REST_LAT = Vector((1, 0, 0))     # lateral en reposo: izquierda = +X


# ============================================================
# 3. UTILIDADES
# ============================================================
def frame_matrix(y, lat):
    """Base ortonormal (columnas x, y, z) con eje Y = y y lateral aproximado = lat."""
    y = y.normalized()
    x = lat - lat.dot(y) * y
    if x.length < 1e-6:
        x = y.orthogonal()
    x.normalize()
    z = x.cross(y)
    return Matrix(((x.x, y.x, z.x), (x.y, y.y, z.y), (x.z, y.z, z.z)))


def load_frames(path):
    """Lee el JSON y devuelve (fps, lista de frames con 33 Vector en espacio Blender)."""
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    frames, prev = [], None
    for raw in data["frames"]:
        # x invertida (espejo), y escalada, z = altura
        pts = [Vector((-raw[i * 4], raw[i * 4 + 1] * DEPTH_SCALE, raw[i * 4 + 2]))
               for i in range(33)]
        if prev is not None:   # suavizado exponencial para reducir jitter
            pts = [prev[i].lerp(pts[i], SMOOTH_ALPHA) for i in range(33)]
        frames.append(pts)
        prev = pts
    return float(data.get("fps", 30.0)), frames


def build_targets(lm):
    """Devuelve {hueso: (dirección objetivo, lateral o None, mezcla 0-1)} para un frame."""
    mid_sh  = (lm[11] + lm[12]) / 2
    mid_hip = (lm[23] + lm[24]) / 2
    mid_ear = (lm[7] + lm[8]) / 2
    spine   = mid_sh - mid_hip
    hips_lat = lm[23] - lm[24]
    sh_lat   = lm[11] - lm[12]

    t = {
        "Hips":  (spine, hips_lat, 1.0),
        "Spine": (spine, (hips_lat + sh_lat) / 2, 1.0),
        "Chest": (spine, sh_lat, 1.0),
        "Neck":  (mid_ear - mid_sh, None, 0.4),               # amortiguado: la cara es ruidosa
        "Head":  (mid_ear - mid_sh, lm[7] - lm[8], 0.4),
    }
    for s, i in SIDES.items():
        t["Shoulder" + s] = (lm[i["sh"]] - mid_sh, None, 0.4)  # clavícula: movimiento sutil
        t["UpperArm" + s] = (lm[i["el"]] - lm[i["sh"]], None, 1.0)
        t["LowerArm" + s] = (lm[i["wr"]] - lm[i["el"]], None, 1.0)
        t["Hand" + s]     = ((lm[i["pk"]] + lm[i["ix"]]) / 2 - lm[i["wr"]], None, 1.0)
        t["Thigh" + s]    = (lm[i["kn"]] - lm[i["hip"]], None, 1.0)
        t["Shin" + s]     = (lm[i["an"]] - lm[i["kn"]], None, 1.0)
        t["Foot" + s]     = (lm[i["toe"]] - lm[i["an"]], None, 1.0)
    return t, mid_hip


# ============================================================
# 4. PREPARAR ESCENA Y ARMATURE
# ============================================================
arm_obj = bpy.data.objects.get(ARMATURE_NAME)
if arm_obj is None or arm_obj.type != 'ARMATURE':
    raise RuntimeError(f"No se encontró el armature '{ARMATURE_NAME}'. "
                       "Ejecuta primero el script que genera el maniquí.")

if bpy.context.object and bpy.context.object.mode != 'OBJECT':
    bpy.ops.object.mode_set(mode='OBJECT')
bpy.context.view_layer.objects.active = arm_obj

pbones = arm_obj.pose.bones
rest = {n: pbones[n].bone.matrix_local.copy() for n in ORDER}   # reposo en espacio armature

for pb in pbones:                      # reinicia pose y fuerza rotación en cuaterniones
    pb.rotation_mode = 'QUATERNION'
    pb.rotation_quaternion = Quaternion((1, 0, 0, 0))
    pb.location = Vector((0, 0, 0))

fps, frames = load_frames(JSON_PATH)
scene = bpy.context.scene
scene.render.fps = int(round(fps))
scene.frame_start, scene.frame_end = 1, len(frames)

arm_obj.animation_data_create()
arm_obj.animation_data.action = bpy.data.actions.new("MP_Anim")

# ============================================================
# 5. RESOLVER Y KEYFRAMEAR CADA FRAME
# ============================================================
ref_hip = None
prev_q = {}

for fi, lm in enumerate(frames):
    targets, mid_hip = build_targets(lm)
    if ref_hip is None:
        ref_hip = mid_hip.copy()
    P = {}                                          # matrices de pose en espacio armature

    for name in ORDER:
        pb = pbones[name]
        R = rest[name]
        R3 = R.to_3x3()

        # Matriz base heredada del padre (equivale a la fórmula de pose de Blender)
        if pb.parent:
            M = P[pb.parent.name] @ (rest[pb.parent.name].inverted() @ R)
        else:
            M = R.copy()

        d, lat, blend = targets[name]
        rest_y = R3 @ Y_AXIS
        if d.length < 1e-6:
            Dm = Matrix.Identity(3)
        else:
            if lat is not None and lat.length > 1e-6:
                # Orientación completa (dirección + giro lateral)
                Dm = frame_matrix(d, lat) @ frame_matrix(rest_y, REST_LAT).inverted()
            else:
                # Solo dirección (arco mínimo)
                Dm = rest_y.rotation_difference(d.normalized()).to_matrix()
            if blend < 1.0:
                Dm = Quaternion().slerp(Dm.to_quaternion(), blend).to_matrix()

        # Rotación local = (orientación heredada)^-1 · orientación deseada
        B3 = M.to_3x3().inverted() @ (Dm @ R3)
        q = B3.to_quaternion()
        if name in prev_q and q.dot(prev_q[name]) < 0:   # evita saltos de signo
            q.negate()
        prev_q[name] = q
        pb.rotation_quaternion = q

        loc = Vector((0, 0, 0))
        if name == "Hips" and ROOT_MOTION:
            delta = (mid_hip - ref_hip) * ROOT_SCALE
            loc = R3.inverted() @ delta                  # a espacio local del hueso
        pb.location = loc

        B4 = B3.to_4x4()
        B4.translation = loc
        P[name] = M @ B4

        pb.keyframe_insert("rotation_quaternion", frame=fi + 1)
        if name == "Hips":
            pb.keyframe_insert("location", frame=fi + 1)

scene.frame_set(1)
print(f"Animación creada: {len(frames)} frames a {fps} fps.")