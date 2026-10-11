"""
Landmarks 3D a partir de 2 cámaras SIN calibración con tablero.

La geometría de las cámaras (pose relativa, focal fina y desfase temporal) se estima con el propio baile;
la escala sale de la estatura del bailarín y se comprueba con la distancia medida entre lentes.

Uso en Colab:
    !pip install -q mediapipe
    !python triangulacion_3d.py
"""
import os
import json
import urllib.request
import warnings

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from scipy.signal import savgol_filter

warnings.filterwarnings('ignore', category=RuntimeWarning)  # nanmean/nanmedian de filas vacías

# --- CONFIGURACIÓN ---
VIDEO_CAM1 = 'cam1_basico.mp4'
VIDEO_CAM2 = 'cam2_basico.mp4'
SALIDA_JSON = 'basico_3d.json'
ALTURA_PERSONA_M = 1.60     # estatura real del bailarín (m): fija la escala final
DISTANCIA_CAMARAS_M = 4.60  # distancia real entre lentes (m): escala inicial y comprobación
FOCAL_EQUIV_MM = 26.0       # focal equivalente a 35 mm del lente principal del celular (casi todos: 24-28 mm)
OFFSET_SEG = None           # None = estimar automáticamente; o fijo: t_cam2 = t_cam1 + OFFSET_SEG
BUSQUEDA_OFFSET_SEG = 0.5   # rango de búsqueda del desfase automático (±segundos)
MIN_VIS = 0.5               # visibilidad mínima de MediaPipe en cada cámara
AUTOCALIBRAR = True         # ajuste fino de cámaras con el baile: huesos de largo constante y simétricos
OPTIMIZAR_FOCAL = True      # dentro del autoajuste, corregir la focal de cada cámara (±30 % sobre FOCAL_EQUIV_MM)
PESO_HUESOS = 30.0          # cuánto pesa la rigidez de los huesos frente al error de reproyección
VERTICAL_POR_PISO = True    # vertical = normal del plano donde se apoyan los pies (si no, postura promedio)
ALINEAR_FRENTE = True       # girar alrededor de la vertical para que el bailarín mire al frente del rig de Blender
MAX_ERROR_PX = 20.0         # error de reproyección máximo para aceptar un punto 3D
MAX_HUECO_FRAMES = 10       # huecos más cortos que esto se rellenan interpolando
SUAVIZADO_SEG = 0.2         # ventana del filtro Savitzky-Golay (0 = sin suavizado)

MODELO = 'pose_landmarker_heavy.task'
URL_MODELO = ('https://storage.googleapis.com/mediapipe-models/pose_landmarker/'
              'pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task')
N_LM = 33
PARES_IZQ_DER = [(1, 4), (2, 5), (3, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16), (17, 18),
                 (19, 20), (21, 22), (23, 24), (25, 26), (27, 28), (29, 30), (31, 32)]
PERM_LR = np.arange(N_LM)
for a, b in PARES_IZQ_DER:
    PERM_LR[a], PERM_LR[b] = b, a

# Segmentos rígidos usados en el autoajuste y pares izquierda/derecha (índices dentro de HUESOS)
HUESOS = [(11, 13), (13, 15), (12, 14), (14, 16), (23, 25), (25, 27), (24, 26), (26, 28),
          (11, 12), (23, 24), (11, 23), (12, 24)]
PARES_SIMETRICOS = [(0, 2), (1, 3), (4, 6), (5, 7), (10, 11)]


def extraer_2d(video_path):
    """Devuelve t (F,) en segundos, pts (F,33,2) en píxeles (NaN sin detección), vis (F,33) y fps.
    El resultado se guarda en <video>.pose2d.npz para no repetir MediaPipe (bórralo si cambias el video)."""
    cache = video_path + '.pose2d.npz'
    if os.path.exists(cache):
        d = np.load(cache)
        return d['t'], d['p'], d['v'], float(d['fps'])

    if not os.path.exists(MODELO):
        urllib.request.urlretrieve(URL_MODELO, MODELO)
    opciones = vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODELO),
        running_mode=vision.RunningMode.VIDEO, num_poses=1,
        min_pose_detection_confidence=0.5, min_pose_presence_confidence=0.5, min_tracking_confidence=0.5)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f'No se pudo abrir {video_path}')
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    ts, pts, vis = [], [], []

    with vision.PoseLandmarker.create_from_options(opciones) as detector:
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            # Usar el tamaño real del frame: CAP_PROP_FRAME_WIDTH/HEIGHT puede venir sin rotar en videos verticales
            h, w = frame.shape[:2]
            t = idx / fps
            rgb = np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            res = detector.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb),
                                            int(round(t * 1000)))
            p = np.full((N_LM, 2), np.nan, np.float32)
            v = np.zeros(N_LM, np.float32)
            if res.pose_landmarks:
                for i, lm in enumerate(res.pose_landmarks[0]):
                    p[i] = (lm.x * w, lm.y * h)
                    v[i] = lm.visibility if lm.visibility is not None else 1.0
            ts.append(t)
            pts.append(p)
            vis.append(v)
            idx += 1
            if idx % 300 == 0:
                print(f'  {video_path}: {idx} frames')
    cap.release()

    ts, pts, vis = np.array(ts), np.array(pts), np.array(vis)
    np.savez(cache, t=ts, p=pts, v=vis, fps=fps)
    detectados = (~np.isnan(pts[:, 0, 0])).mean() * 100
    print(f'{video_path}: {len(ts)} frames @ {fps:.2f} fps, persona detectada en {detectados:.1f}%')
    return ts, pts, vis, fps


def remuestrear(t_src, datos, t_obj):
    """Interpola linealmente datos (F,...) definidos en t_src a los tiempos t_obj.
    Esto resuelve fps distintos y desfases de fracciones de frame. NaN fuera de rango."""
    j = np.searchsorted(t_src, t_obj, side='right')
    valido = (j > 0) & (j < len(t_src))
    j0 = np.clip(j - 1, 0, len(t_src) - 1)
    j1 = np.clip(j, 0, len(t_src) - 1)
    dt = t_src[j1] - t_src[j0]
    w = np.where(dt > 0, (t_obj - t_src[j0]) / np.where(dt > 0, dt, 1), 0.0)
    w = w.reshape(-1, *([1] * (datos.ndim - 1)))
    out = (1 - w) * datos[j0] + w * datos[j1]
    out[~valido] = np.nan
    return out


def triangular(p1, p2, cal, residuos=False):
    """p1, p2: (N,2) en píxeles. Devuelve X (N,3) en metros (marco de la cámara 1) y error de reproyección (N,) px.
    Con residuos=True devuelve en su lugar los residuos (N,4) [dx1, dy1, dx2, dy2] para el autoajuste."""
    if len(p1) == 0:
        return np.zeros((0, 3)), np.zeros((0, 4) if residuos else 0)
    K1, d1, K2, d2, R, T = cal['K1'], cal['d1'], cal['K2'], cal['d2'], cal['R'], cal['T'].reshape(3, 1)
    # Quitar distorsión del lente -> coordenadas normalizadas, y triangular con [I|0] y [R|T]
    n1 = cv2.undistortPoints(p1.reshape(-1, 1, 2).astype(np.float64), K1, d1).reshape(-1, 2)
    n2 = cv2.undistortPoints(p2.reshape(-1, 1, 2).astype(np.float64), K2, d2).reshape(-1, 2)
    Xh = cv2.triangulatePoints(np.hstack([np.eye(3), np.zeros((3, 1))]), np.hstack([R, T]), n1.T, n2.T)
    X = (Xh[:3] / Xh[3]).T

    r1, _ = cv2.projectPoints(X, np.zeros(3), np.zeros(3), K1, d1)
    r2, _ = cv2.projectPoints(X, cv2.Rodrigues(R)[0], T, K2, d2)
    d1, d2 = r1.reshape(-1, 2) - p1, r2.reshape(-1, 2) - p2
    # Puntos detrás de alguna cámara = correspondencia incorrecta
    detras = (X[:, 2] <= 0) | ((X @ R.T + T.ravel())[:, 2] <= 0)
    if residuos:
        r = np.hstack([d1, d2])
        r[detras] = 50.0
        return X, r
    err = np.maximum(np.linalg.norm(d1, axis=1), np.linalg.norm(d2, axis=1))
    err[detras] = 1e6
    return X, err


def triangular_secuencia(a, b, mascara, cal):
    """Triangula todos los puntos (F,33) donde mascara es True. Error = inf donde no se triangula."""
    X = np.full(a.shape[:2] + (3,), np.nan)
    E = np.full(a.shape[:2], np.inf)
    X[mascara], E[mascara] = triangular(a[mascara], b[mascara], cal)
    return X, E


def mascara_valida(p, v):
    return (v >= MIN_VIS) & ~np.isnan(p[..., 0])


def mediana_por_frame(E):
    med = np.nanmedian(np.where(np.isfinite(E), E, np.nan), axis=1)
    return np.where(np.isnan(med), np.inf, med)


def estimar_offset(t1, p1, v1, t2, p2, v2, cal, fps1, fps2):
    """Busca el desfase temporal que minimiza el error de reproyección (requiere que la persona se mueva)."""
    sel = np.unique(np.linspace(0, len(t1) - 1, min(len(t1), 600)).astype(int))
    a, m1 = p1[sel], mascara_valida(p1[sel], v1[sel])
    if not m1.any():
        return 0.0
    TOPE = 50.0

    def costo(off):
        q2 = remuestrear(t2, p2, t1[sel] + off)
        w2 = remuestrear(t2, v2, t1[sel] + off)
        _, E = triangular_secuencia(a, q2, m1 & mascara_valida(q2, w2), cal)
        return np.minimum(E[m1], TOPE).mean()  # puntos sin pareja en cam2 cuentan como TOPE

    paso = 1.0 / max(fps1, fps2)
    gruesos = np.arange(-BUSQUEDA_OFFSET_SEG, BUSQUEDA_OFFSET_SEG + paso / 2, paso)
    mejor = min(gruesos, key=costo)
    finos = np.arange(mejor - paso, mejor + paso, paso / 10)
    mejor = min(finos, key=costo)
    print(f'Desfase estimado: cam2 = cam1 {mejor:+.4f} s (costo {costo(mejor):.2f} px)')
    return float(mejor)


def rellenar_huecos(x, max_hueco):
    """Interpola linealmente rachas de NaN de longitud <= max_hueco (no en los extremos)."""
    nan = np.isnan(x)
    if nan.all() or not nan.any():
        return x
    idx = np.arange(len(x))
    lleno = np.interp(idx, idx[~nan], x[~nan])
    bordes = np.flatnonzero(np.diff(np.r_[0, nan.astype(int), 0]))
    x = x.copy()
    for ini, fin in zip(bordes[::2], bordes[1::2]):
        if ini > 0 and fin < len(x) and fin - ini <= max_hueco:
            x[ini:fin] = lleno[ini:fin]
    return x


def suavizar(x, ventana):
    """Savitzky-Golay en cada tramo continuo sin NaN."""
    if ventana < 5:
        return x
    x = x.copy()
    ok = ~np.isnan(x)
    bordes = np.flatnonzero(np.diff(np.r_[0, ok.astype(int), 0]))
    for ini, fin in zip(bordes[::2], bordes[1::2]):
        if fin - ini >= ventana:
            x[ini:fin] = savgol_filter(x[ini:fin], ventana, 2)
    return x


def a_marco_mundo(X):
    """Pasa de la cámara 1 de OpenCV (x der, y abajo, z adelante) al sistema del JSON de animación:
    z = altura con el piso en 0, y = profundidad (alejándose de la cámara 1), x = lateral espejado
    (el script de Blender invierte x). La vertical se estima de la postura promedio (tobillos -> hombros),
    así no importa si la cámara 1 estaba inclinada."""
    v = (X[:, 11] + X[:, 12]) / 2 - (X[:, 27] + X[:, 28]) / 2
    v = v[~np.isnan(v).any(axis=1)]
    if len(v) >= 10:
        arriba = np.median(v / np.linalg.norm(v, axis=1, keepdims=True), axis=0)
    else:
        arriba = np.array([0.0, -1.0, 0.0])  # sin tobillos visibles: suponer cámara nivelada
    arriba /= np.linalg.norm(arriba)
    if VERTICAL_POR_PISO:
        arriba = vertical_por_piso(X, arriba)
    adelante = np.array([0.0, 0.0, 1.0]) - arriba[2] * arriba  # eje óptico proyectado al plano horizontal
    adelante /= np.linalg.norm(adelante)
    lateral = np.cross(adelante, arriba)
    W = X @ np.stack([lateral, adelante, arriba]).T

    if ALINEAR_FRENTE:
        # Girar alrededor de la vertical para que la línea izquierda->derecha (hombros y caderas) quede en +X:
        # así el bailarín mira hacia -Y, el frente del rig, sin importar dónde estaba la cámara 1
        lat = np.nanmedian(np.r_[W[:, 11, :2] - W[:, 12, :2], W[:, 23, :2] - W[:, 24, :2]], axis=0)
        if not np.isnan(lat).any():
            ang = np.arctan2(lat[1], lat[0])
            c, s = np.cos(ang), np.sin(ang)
            W[..., :2] = W[..., :2] @ np.array([[c, -s], [s, c]])
            print(f'Orientación: girado {np.degrees(ang):+.1f}° para que el bailarín mire al frente')

    piso = np.nanpercentile(W[:, 27:33, 2], 2)           # tobillos, talones y puntas
    centro = np.nanmedian((W[:, 23, :2] + W[:, 24, :2]) / 2, axis=0)
    W[..., :2] -= np.nan_to_num(centro)
    W[..., 2] -= 0.0 if np.isnan(piso) else piso
    W[..., 0] *= -1
    return W


def error_mediano(a, b, cal):
    _, e = triangular(a, b, cal)
    return float(np.median(e)) if e.size else np.inf


def refinar_extrinsecos(p1, v1, q2, w2, cal):
    """Estima R y T usando al propio bailarín como patrón de calibración: miles de correspondencias
    de landmarks -> matriz esencial. Los intrínsecos (K, d) se mantienen.
    Solo reemplaza la estimación actual si reduce el error de reproyección."""
    m = mascara_valida(p1, v1) & mascara_valida(q2, w2) & (v1 >= 0.8) & (w2 >= 0.8)
    if m.sum() < 200:
        print('Refinamiento de extrínsecos omitido: pocas correspondencias confiables')
        return cal
    a, b = p1[m].astype(np.float64), q2[m].astype(np.float64)
    if len(a) > 8000:
        sel = np.random.default_rng(0).choice(len(a), 8000, replace=False)
        a, b = a[sel], b[sel]
    n1 = cv2.undistortPoints(a.reshape(-1, 1, 2), cal['K1'], cal['d1']).reshape(-1, 2)
    n2 = cv2.undistortPoints(b.reshape(-1, 1, 2), cal['K2'], cal['d2']).reshape(-1, 2)
    f = (cal['K1'][0, 0] + cal['K2'][0, 0]) / 2
    E, mask = cv2.findEssentialMat(n1, n2, np.eye(3), method=cv2.RANSAC, prob=0.999, threshold=4.0 / f)
    if E is None:
        return cal
    _, R, t, _ = cv2.recoverPose(E[:3], n1, n2, np.eye(3), mask=mask)
    # La matriz esencial no da la escala: se usa la distancia medida entre lentes
    base = float(np.linalg.norm(cal['T']))
    nuevo = dict(cal, R=R, T=(t.ravel() / np.linalg.norm(t) * base).reshape(3, 1))

    e_actual, e_nuevo = error_mediano(a, b, cal), error_mediano(a, b, nuevo)
    ang = np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))
    print(f'Pose relativa por matriz esencial: {e_nuevo:.1f} px (antes {e_actual:.1f} px), '
          f'ángulo entre cámaras {ang:.1f}°, inliers {int(mask.sum())}/{len(a)}')
    return nuevo if e_nuevo < e_actual else cal


def _cal_desde_parametros(x, cal):
    """x = [rvec(3), theta, phi, log escala focal cam1, log escala focal cam2]."""
    nuevo = dict(cal)
    for k, s in (('K1', x[5]), ('K2', x[6])):
        K = cal[k].copy()
        K[0, 0] *= np.exp(s)
        K[1, 1] *= np.exp(s)
        nuevo[k] = K
    nuevo['R'] = cv2.Rodrigues(np.asarray(x[:3], np.float64))[0]
    th, ph = x[3], x[4]
    direccion = np.array([[np.sin(th) * np.cos(ph)], [np.sin(th) * np.sin(ph)], [np.cos(th)]])
    nuevo['T'] = direccion * np.linalg.norm(cal['T'])
    return nuevo


def autocalibrar(p1, v1, q2, w2, cal):
    """Ajuste fino (bundle adjustment reducido) usando la ejecución del baile:
    minimiza el error de reproyección y además exige que cada hueso mida lo mismo en todos los frames
    y que izquierda y derecha midan igual. Eso corrige la deformación en profundidad que deja la matriz
    esencial cuando la focal o la pose relativa no son exactas. La escala no se toca aquí."""
    from scipy.optimize import least_squares

    m = mascara_valida(p1, v1) & mascara_valida(q2, w2) & (v1 >= 0.7) & (w2 >= 0.7)
    frames = np.flatnonzero(m.sum(axis=1) >= 20)
    if len(frames) < 30:
        print('Autoajuste omitido: pocos frames con el cuerpo bien visible en ambas cámaras')
        return cal
    frames = frames[np.unique(np.linspace(0, len(frames) - 1, min(len(frames), 200)).astype(int))]
    a, b, m = p1[frames].astype(np.float64), q2[frames].astype(np.float64), m[frames]
    idx_huesos = [np.flatnonzero(m[:, i] & m[:, j]) for i, j in HUESOS]

    T = cal['T'].ravel() / np.linalg.norm(cal['T'])
    x0 = np.r_[cv2.Rodrigues(cal['R'])[0].ravel(), np.arccos(np.clip(T[2], -1, 1)), np.arctan2(T[1], T[0]), 0.0, 0.0]
    lim_f = 0.3 if OPTIMIZAR_FOCAL else 1e-9
    lb = np.r_[[-np.inf] * 5, -lim_f, -lim_f]
    ub = np.r_[[np.inf] * 5, lim_f, lim_f]

    def evaluar(x):
        c = _cal_desde_parametros(x, cal)
        X = np.full(a.shape[:2] + (3,), np.nan)
        X[m], r = triangular(a[m], b[m], c, residuos=True)
        partes, largos, variacion = [r.ravel()], [], []
        for (i, j), f in zip(HUESOS, idx_huesos):
            largo = np.linalg.norm(X[f, i] - X[f, j], axis=1)
            med = np.median(largo) if largo.size else 1.0
            largos.append(med)
            partes.append(PESO_HUESOS * (largo / med - 1))
            variacion.append(np.median(np.abs(largo / med - 1)))
        for h1, h2 in PARES_SIMETRICOS:
            partes.append([PESO_HUESOS * 5 * (largos[h1] / largos[h2] - 1)])
        res = np.nan_to_num(np.concatenate(partes), nan=50.0, posinf=50.0, neginf=-50.0)
        reproy = np.median(np.hypot(r[:, 0::2], r[:, 1::2]))
        return res, reproy, np.nanmean(variacion)

    _, rep0, var0 = evaluar(x0)
    sol = least_squares(lambda x: evaluar(x)[0], x0, bounds=(lb, ub), loss='soft_l1', f_scale=3.0,
                        x_scale='jac', max_nfev=300)
    _, rep1, var1 = evaluar(sol.x)
    nuevo = _cal_desde_parametros(sol.x, cal)
    ang = np.degrees(np.arccos(np.clip((np.trace(cal['R'].T @ nuevo['R']) - 1) / 2, -1, 1)))
    print(f'Autoajuste con el baile: reproyección {rep0:.2f} -> {rep1:.2f} px | variación del largo de huesos '
          f'{var0 * 100:.1f}% -> {var1 * 100:.1f}% | focal cam1 {np.exp(sol.x[5]) * 100 - 100:+.1f}%, '
          f'cam2 {np.exp(sol.x[6]) * 100 - 100:+.1f}% | rotación corregida {ang:.1f}°')
    ang_cams = np.degrees(np.arccos(np.clip((np.trace(nuevo['R']) - 1) / 2, -1, 1)))
    print(f'Ángulo entre cámaras resultante: {ang_cams:.1f}°')
    if var1 < var0 and rep1 < 1.5 * rep0 + 0.5:
        return nuevo
    print('  El autoajuste no mejoró: se mantiene la calibración anterior')
    return cal


def estimar_estatura(X):
    """Estatura aproximada a partir de la cadena tobillo-rodilla-cadera-hombro-nariz (~89 % de la estatura)."""
    def largo(i, j):
        return np.nanmedian(np.linalg.norm(X[:, i] - X[:, j], axis=1))
    pierna = (largo(25, 27) + largo(26, 28)) / 2 + (largo(23, 25) + largo(24, 26)) / 2
    hombros, caderas = (X[:, 11] + X[:, 12]) / 2, (X[:, 23] + X[:, 24]) / 2
    torso = np.nanmedian(np.linalg.norm(hombros - caderas, axis=1))
    cuello = np.nanmedian(np.linalg.norm(X[:, 0] - hombros, axis=1))
    return (pierna + torso + cuello) / 0.89


def vertical_por_piso(X, arriba):
    """Normal del plano donde se apoyan talones y puntas. No depende de que el bailarín esté derecho."""
    pts = X[:, 29:33].reshape(-1, 3)
    pts = pts[~np.isnan(pts).any(axis=1)]
    if len(pts) < 100:
        return arriba
    n = arriba
    for _ in range(5):
        h = pts @ n
        apoyados = pts[h <= np.percentile(h, 40)]      # los pies en el piso son los puntos más bajos
        _, sv, vt = np.linalg.svd(apoyados - apoyados.mean(axis=0), full_matrices=False)
        n = vt[2] * np.sign(vt[2] @ arriba)
    extension = sv[1] / np.sqrt(len(apoyados))          # cuánto se desplazan los pies sobre el piso (m)
    ang = np.degrees(np.arccos(np.clip(n @ arriba, -1, 1)))
    if extension < 0.06 or ang > 25:
        print(f'Vertical: postura promedio (pies poco desplazados o plano dudoso: {ang:.1f}°)')
        return arriba
    print(f'Vertical: plano del piso ({ang:.1f}° respecto a la postura promedio)')
    return n


NOMBRES_DIAG = {0: 'nariz', 11: 'hombro I', 12: 'hombro D', 13: 'codo I', 14: 'codo D',
                15: 'muñeca I', 16: 'muñeca D', 23: 'cadera I', 24: 'cadera D', 25: 'rodilla I',
                26: 'rodilla D', 27: 'tobillo I', 28: 'tobillo D', 31: 'punta I', 32: 'punta D'}


def diagnostico(m1, m2, E):
    """Tabla por landmark para saber si un punto se pierde por visibilidad o por error de triangulación."""
    print(f'\n{"landmark":<10}{"vis cam1":>9}{"vis cam2":>9}{"ambas":>7}{"err med":>10}{"aceptado":>10}')
    for j, nombre in NOMBRES_DIAG.items():
        ambas = m1[:, j] & m2[:, j]
        e = E[ambas, j]
        e = e[np.isfinite(e)]
        med = f'{np.median(e):.1f}px' if e.size else '-'
        print(f'{nombre:<10}{m1[:, j].mean() * 100:8.0f}%{m2[:, j].mean() * 100:8.0f}%{ambas.mean() * 100:6.0f}%'
              f'{med:>10}{(E[:, j] <= MAX_ERROR_PX).mean() * 100:9.0f}%')


def tamano_video(video_path):
    cap = cv2.VideoCapture(video_path)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise FileNotFoundError(f'No se pudo leer {video_path}')
    return frame.shape[1], frame.shape[0]


def intrinsecos_iniciales(video_path):
    """K aproximada sin tablero: focal a partir de la focal equivalente del celular, centro óptico al centro
    de la imagen y sin distorsión (el lente principal de un celular casi no distorsiona; el autoajuste afina la focal)."""
    w, h = tamano_video(video_path)
    f = max(w, h) * FOCAL_EQUIV_MM / 36.0
    return np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]], np.float64), np.zeros(5)


def distancia_sampson(F, x1, x2):
    """Distancia (px) de cada correspondencia a su línea epipolar."""
    a = np.hstack([x1, np.ones((len(x1), 1))])
    b = np.hstack([x2, np.ones((len(x2), 1))])
    Fa, Ftb = a @ F.T, b @ F
    num = np.sum(b * Fa, axis=1) ** 2
    den = Fa[:, 0] ** 2 + Fa[:, 1] ** 2 + Ftb[:, 0] ** 2 + Ftb[:, 1] ** 2
    return np.sqrt(num / np.maximum(den, 1e-12))


def estimar_offset_sin_calibracion(t1, p1, v1, t2, p2, v2, fps1, fps2):
    """Desfase temporal sin conocer las cámaras: con el desfase correcto, todas las correspondencias
    cumplen una misma geometría epipolar (matriz fundamental), y el error epipolar es mínimo."""
    sel = np.unique(np.linspace(0, len(t1) - 1, min(len(t1), 600)).astype(int))
    a, m1 = p1[sel].astype(np.float64), mascara_valida(p1[sel], v1[sel]) & (v1[sel] >= 0.7)
    TOPE = 30.0

    def costo(off):
        q2 = remuestrear(t2, p2, t1[sel] + off).astype(np.float64)
        w2 = remuestrear(t2, v2, t1[sel] + off)
        m = m1 & mascara_valida(q2, w2) & (w2 >= 0.7)
        if m.sum() < 100:
            return TOPE
        x1, x2 = a[m], q2[m]
        cv2.setRNGSeed(0)
        F, _ = cv2.findFundamentalMat(x1, x2, cv2.FM_RANSAC, 3.0, 0.999)
        if F is None:
            return TOPE
        d = np.full(m1.sum(), TOPE)                       # puntos sin pareja en cam2 cuentan como TOPE
        d[m[m1]] = np.minimum(distancia_sampson(F[:3], x1, x2), TOPE)
        return d.mean()

    paso = 1.0 / max(fps1, fps2)
    gruesos = np.arange(-BUSQUEDA_OFFSET_SEG, BUSQUEDA_OFFSET_SEG + paso / 2, paso)
    mejor = min(gruesos, key=costo)
    finos = np.arange(mejor - paso, mejor + paso, paso / 5)
    mejor = min(finos, key=costo)
    print(f'Desfase inicial (geometría epipolar): cam2 = cam1 {mejor:+.4f} s ({costo(mejor):.2f} px)')
    return float(mejor)


def main():
    print('Extrayendo 2D de Cámara 1...')
    t1, p1, v1, fps1 = extraer_2d(VIDEO_CAM1)
    print('Extrayendo 2D de Cámara 2...')
    t2, p2, v2, fps2 = extraer_2d(VIDEO_CAM2)

    # Calibración inicial sin tablero: K por la focal típica del celular, pose relativa aún desconocida
    K1, d1 = intrinsecos_iniciales(VIDEO_CAM1)
    K2, d2 = intrinsecos_iniciales(VIDEO_CAM2)
    cal = dict(K1=K1, d1=d1, K2=K2, d2=d2, R=np.eye(3), T=np.array([[DISTANCIA_CAMARAS_M], [0.0], [0.0]]))
    print(f'Focal inicial: cam1 {K1[0, 0]:.0f} px, cam2 {K2[0, 0]:.0f} px')

    # 1) Desfase sin calibración  2) pose relativa por matriz esencial  3) desfase fino por triangulación
    offset = OFFSET_SEG if OFFSET_SEG is not None else estimar_offset_sin_calibracion(
        t1, p1, v1, t2, p2, v2, fps1, fps2)
    for _ in range(2):
        q2, w2 = remuestrear(t2, p2, t1 + offset), remuestrear(t2, v2, t1 + offset)
        cal = refinar_extrinsecos(p1, v1, q2, w2, cal)
        if OFFSET_SEG is None:
            offset = estimar_offset(t1, p1, v1, t2, p2, v2, cal, fps1, fps2)
    # 4) Autoajuste con huesos rígidos y simétricos (corrige focal y deformación en profundidad)
    if AUTOCALIBRAR:
        q2, w2 = remuestrear(t2, p2, t1 + offset), remuestrear(t2, v2, t1 + offset)
        cal = autocalibrar(p1, v1, q2, w2, cal)
        if OFFSET_SEG is None:
            offset = estimar_offset(t1, p1, v1, t2, p2, v2, cal, fps1, fps2)

    # Cámara 2 llevada a los tiempos de la cámara 1 (la salida queda a los fps de la cámara 1)
    q2 = remuestrear(t2, p2, t1 + offset)
    w2 = remuestrear(t2, v2, t1 + offset)
    m1 = mascara_valida(p1, v1)

    print('Triangulando...')
    # MediaPipe a veces confunde izquierda/derecha en una sola cámara: probar ambas opciones por frame
    XA, EA = triangular_secuencia(p1, q2, m1 & mascara_valida(q2, w2), cal)
    q2s, w2s = q2[:, PERM_LR], w2[:, PERM_LR]
    XB, EB = triangular_secuencia(p1, q2s, m1 & mascara_valida(q2s, w2s), cal)
    usar_b = mediana_por_frame(EB) < 0.7 * mediana_por_frame(EA)
    X = np.where(usar_b[:, None, None], XB, XA)
    E = np.where(usar_b[:, None], EB, EA)
    w2f = np.where(usar_b[:, None], w2s, w2)

    valido = E <= MAX_ERROR_PX
    diagnostico(m1, mascara_valida(q2, w2), E)
    X[~valido] = np.nan
    vis = np.where(valido, np.minimum(v1, np.nan_to_num(w2f)), 0.0)

    # Continuidad temporal: si el esqueleto completo quedó con izq/der invertidos respecto al frame anterior, corregir
    ultimo = np.full((N_LM, 3), np.nan)
    n_giros = 0
    for f in range(len(X)):
        if np.isnan(X[f, :, 0]).all():
            continue
        d_normal = np.nanmean(np.linalg.norm(X[f] - ultimo, axis=1))
        d_invert = np.nanmean(np.linalg.norm(X[f][PERM_LR] - ultimo, axis=1))
        if d_invert < 0.5 * d_normal:
            X[f], vis[f] = X[f][PERM_LR], vis[f][PERM_LR]
            n_giros += 1
        ok = ~np.isnan(X[f, :, 0])
        ultimo[ok] = X[f][ok]

    # Rellenar huecos cortos y suavizar
    ventana = int(round(SUAVIZADO_SEG * fps1))
    ventana += (ventana % 2 == 0)
    for j in range(N_LM):
        antes = np.isnan(X[:, j, 0])
        for k in range(3):
            X[:, j, k] = suavizar(rellenar_huecos(X[:, j, k], MAX_HUECO_FRAMES), ventana)
        vis[antes & ~np.isnan(X[:, j, 0]), j] = 0.5  # puntos interpolados

    # Escala final por la estatura; la distancia entre lentes queda como comprobación de la geometría
    estatura = estimar_estatura(X)
    if ALTURA_PERSONA_M and np.isfinite(estatura):
        factor = ALTURA_PERSONA_M / estatura
        X *= factor
        print(f'Escala: con la distancia entre lentes ({DISTANCIA_CAMARAS_M:.2f} m) la estatura sale '
              f'{estatura:.2f} m; ajustada a {ALTURA_PERSONA_M:.2f} m (factor {factor:.2f})')
        if abs(factor - 1) > 0.1:
            print(f'  ⚠ Discrepancia de {abs(factor - 1) * 100:.0f}%: o la distancia medida no es de lente a lente, '
                  f'o la focal/geometría no quedó bien (revisa el autoajuste)')
    else:
        print(f'Estatura estimada del bailarín: {estatura:.2f} m')

    X = a_marco_mundo(X)

    # Lo que siga faltando: repetir el último valor conocido (visibilidad 0) para que el JSON no tenga NaN
    F = len(X)
    for j in range(N_LM):
        nan = np.isnan(X[:, j, 0])
        if nan.all():
            X[:, j] = 0.0
            continue
        idx = np.where(~nan, np.arange(F), 0)
        np.maximum.accumulate(idx, out=idx)
        idx[:np.argmax(~nan)] = np.argmax(~nan)
        X[:, j] = X[idx, j]

    # Mismo formato que el JSON de animación: 33 landmarks x [x, y, z, visibilidad] por frame
    frames_3d = [[round(float(c), 5) for j in range(N_LM) for c in (*X[f, j], vis[f, j])] for f in range(F)]
    with open(SALIDA_JSON, 'w') as fh:
        json.dump({'fps': float(fps1), 'frames': frames_3d}, fh)

    e_ok = E[valido]
    print(f'\nFrames: {F} | puntos triangulados válidos: {valido.mean() * 100:.1f}%')
    if e_ok.size:
        print(f'Error de reproyección mediano: {np.median(e_ok):.2f} px (p90 {np.percentile(e_ok, 90):.2f} px)')
    print(f'Frames con izq/der corregido en cam2: {usar_b.sum()} | inversiones globales corregidas: {n_giros}')
    print(f'¡Exportación 3D multicámara completada en {SALIDA_JSON}!')


if __name__ == '__main__':
    main()


