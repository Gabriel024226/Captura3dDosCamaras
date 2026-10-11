# Captura de Movimiento 3D con Dos Cámaras (Stereo Motion Capture)

Sistema completo de captura de movimiento (*motion capture*) 3D a partir de dos cámaras estándar (por ejemplo, teléfonos celulares) sin necesidad de patrones de calibración física (tablero de ajedrez/checkerboard).

El proyecto permite:
1. Extraer puntos clave 2D con MediaPipe Pose.
2. Sincronizar temporalmente y autocalibrar las cámaras utilizando el movimiento natural del cuerpo humano.
3. Triangular y filtrar las trayectorias 3D a escala métrica real.
4. Animar automáticamente un esqueleto (*armature*) dentro de Blender con cinemática jerárquica y soporte para giros complejos.

---

## 📁 Estructura del Repositorio

| Archivo | Descripción | Entorno de Ejecución |
| :--- | :--- | :--- |
| `two_cams_to_3d.py` | Extracción 2D, sincronización, calibración multicámara, triangulación 3D y exportación a JSON. | Python / Google Colab |
| `animate_model.py` | Importación del JSON y animación básica de esqueleto en Blender. | Blender (Scripting Workspace) |
| `animate_spins_model.py` | Importación y animación avanzada en Blender optimizada para giros (360°/spins) y oclusiones. | Blender (Scripting Workspace) |

---

## ⚙️ Descripción Detallada de los Códigos

### 1. `two_cams_to_3d.py` — Triangulación 3D Multicámara sin Tablero

Este script constituye el núcleo del procesamiento de visión por computadora. Realiza todo el flujo matemático para transformar dos grabaciones de video en una animación 3D coherente.

#### ¿Cómo funciona paso a paso?

1. **Extracción 2D con MediaPipe Pose (`extraer_2d`)**:
   - Descarga automáticamente el modelo `pose_landmarker_heavy.task` de MediaPipe.
   - Procesa los fotogramas de cada video extrayendo las coordenadas $(x, y)$ de los 33 landmarks corporales y su nivel de visibilidad/confianza.
   - Genera una caché local en disco (`<video>.pose2d.npz`) para acelerar ejecuciones posteriores.

2. **Parámetros Intrínsecos Iniciales (`intrinsecos_iniciales`)**:
   - Estima la matriz intrínseca $K$ de cada cámara sin requerir calibración de laboratorio, basándose en la distancia focal equivalente típica de smartphones (por defecto $26\text{ mm}$ en formato $35\text{ mm}$) y ubicando el punto principal en el centro del sensor.

3. **Sincronización Temporal Automática (`estimar_offset_sin_calibracion` y `estimar_offset`)**:
   - **Fase inicial**: Si no se conoce el desfase temporal entre cámaras (`OFFSET_SEG = None`), busca el desfase óptimo minimizando el error de la geometría epipolar (distancia de Sampson sobre la Matriz Fundamental con RANSAC).
   - **Remuestreo (`remuestrear`)**: Interpola temporalmente las trayectorias de la cámara 2 para sincronizarlas exactamente con los instantes de muestreo de la cámara 1, soportando diferencias de FPS o fracciones de frame.
   - **Fase fina**: Ajusta el offset minimizando el error de reproyección 3D una vez estimados los parámetros de cámara.

4. **Calibración Extrínseca por Matriz Esencial (`refinar_extrinsecos`)**:
   - Usa los puntos detectados del propio usuario en movimiento como patrón de correspondencias estéreo.
   - Calcula la Matriz Esencial ($E$) con RANSAC y descompone la rotación relativa ($R$) y dirección de traslación ($T$) mediante `cv2.recoverPose`. La magnitud de $T$ se fija con la distancia real medida entre los lentes (`DISTANCIA_CAMARAS_M`).

5. **Autoajuste y Optimización Global (`autocalibrar`)**:
   - Ejecuta un *Bundle Adjustment* reducido mediante optimización por mínimos cuadrados no lineales (`scipy.optimize.least_squares`).
   - Minimiza conjuntamente el error de reproyección y restricciones de biomecánica:
     - **Longitud constante de huesos**: Los segmentos óseos deben conservar su longitud a lo largo del tiempo.
     - **Simetría bilateral**: Los segmentos izquierdo y derecho correspondientes deben tener longitudes equivalentes.
   - Ajusta fino de la rotación entre cámaras, dirección de traslación y, opcionalmente, la distancia focal (`OPTIMIZAR_FOCAL`).

6. **Triangulación y Corrección Izquierda / Derecha**:
   - Triangula las posiciones 3D utilizando `cv2.triangulatePoints`.
   - Evalúa posibles inversiones de simetría (confusión de lado izquierdo/derecho común en detectores 2D monoculares) y selecciona la hipótesis con menor error de reproyección (`PERM_LR`).
   - Corrige saltos globales de lateralidad entre fotogramas consecutivos.

7. **Filtrado, Interpolación y Suavizado**:
   - Descarta puntos con error de reproyección superior a `MAX_ERROR_PX`.
   - Interpola huecos cortos producidos por oclusiones (`rellenar_huecos`).
   - Aplica un filtro de **Savitzky-Golay** (`savgol_filter`) a lo largo del tiempo para eliminar temblores (*jitter*) preservando las aceleraciones naturales del movimiento.

8. **Ajuste de Escala Métrica y Transformación de Escena (`a_marco_mundo`)**:
   - Escala el modelo a la estatura real de la persona (`ALTURA_PERSONA_M`).
   - Detecta la vertical del mundo y el plano del suelo mediante la posición de contacto de los pies (`vertical_por_piso`).
   - Centra las caderas en el origen y orienta el movimiento para que mire hacia el frente del rig de Blender (`ALINEAR_FRENTE`).

9. **Exportación a JSON**:
   - Guarda el archivo resultante con la tasa de cuadros (`fps`) y una secuencia de 33 landmarks con coordenadas $[x, y, z, \text{visibilidad}]$.

---

### 2. `animate_model.py` — Retargeting y Animación en Blender (Versión Base)

Script diseñado para ejecutarse dentro del editor de scripts de **Blender** con el motor `bpy`.

#### Características principales:
- **Lectura del JSON**: Carga los landmarks 3D y los adapta al espacio tridimensional de Blender.
- **Jerarquía Anatómica**: Resuelve la pose hueso por hueso en orden estricto de padres a hijos (`Hips` $\to$ `Spine` $\to$ `Chest` $\to$ `Neck` $\to$ `Head`, seguido de extremidades).
- **Construcción de Bases de Orientación (`frame_matrix`)**:
  - Para huesos con eje de torsión definido (como torso y cabeza), construye una base ortonormal completa usando la dirección principal y el vector lateral (ej. línea entre hombros o caderas).
  - Para extremidades (brazos y piernas), calcula la rotación por arco mínimo (`rotation_difference`) hacia el landmark objetivo.
- **Espacio Local y Cinemática**: Transforma cada orientación al espacio local del hueso cancelando la rotación acumulada del padre.
- **Root Motion**: Transfiere el desplazamiento global de las caderas al hueso raíz (`Hips`), permitiendo que el personaje camine o se desplace por el escenario.
- **Inserción de Keyframes**: Asigna fotogramas clave en cuaterniones (`rotation_quaternion`) y traslación (`location`).

---

### 3. `animate_spins_model.py` — Animación en Blender para Giros Complejos y Oclusiones

Versión mejorada y robusta de `animate_model.py`, diseñada específicamente para disciplinas con rotaciones completas de 360°, bailes (como salsa o ritmos latinos) o momentos donde las extremidades quedan temporalmente ocultas a las cámaras.

#### Mejoras sobre la versión base:
- **Gestión Inteligente de Visibilidad (`VIS_MIN`)**:
  - Supervisa la visibilidad de los landmarks utilizados por cada hueso.
  - Si un punto clave cae por debajo del umbral de confianza, el hueso congela o retiene su última pose válida (`last_target`) en lugar de deformarse hacia el origen.
- **Rotación Relativa Heredada sin Singularidades**:
  - En giros de 180° o cuando la persona da la espalda a las cámaras, los métodos estándar de arco mínimo pueden sufrir problemas de alineación o discontinuidades en los cuaterniones.
  - En `animate_spins_model.py`, la rotación de clavículas y extremidades se calcula a partir del vector ya transformado por el hueso padre (`heredado = M3 @ Y_AXIS`). De este modo, los brazos rotan solidariamente con el torso y solo corrigen su desviación local, previniendo torsiones no deseadas.
- **Métricas de Diagnóstico**:
  - Imprime un resumen al finalizar indicando si algún hueso careció de datos válidos durante más del 50% de la animación.

---

## 🚀 Guía de Uso Rápido

### Requisitos Previos

Instala las dependencias en tu entorno de Python:

```bash
pip install opencv-python numpy mediapipe scipy
```

### Paso 1: Procesar los Videos con `two_cams_to_3d.py`

1. Graba a la persona desde dos ángulos distintos con cámaras estáticas (por ejemplo, a 60° - 90° de separación).
2. Ajusta las variables de configuración en la cabecera de `two_cams_to_3d.py`:
   - `VIDEO_CAM1 = 'cam1.mp4'`
   - `VIDEO_CAM2 = 'cam2.mp4'`
   - `SALIDA_JSON = 'animacion_3d.json'`
   - `ALTURA_PERSONA_M = 1.70` (estatura real aproximada)
   - `DISTANCIA_CAMARAS_M = 4.50` (distancia medida entre lentes)
3. Ejecuta el script:
   ```bash
   python two_cams_to_3d.py
   ```

### Paso 2: Aplicar la Animación en Blender

1. Abre **Blender** con tu escena y el esqueleto compatible (`MP_Armature`).
2. Abre la pestaña **Scripting**.
3. Carga `animate_spins_model.py` (recomendado) o `animate_model.py`.
4. Configura la ruta del archivo JSON generado:
   ```python
   JSON_PATH = r"C:\ruta\hacia\animacion_3d.json"
   ```
5. Haz clic en **Run Script** (▶). La animación se insertará automáticamente en la línea de tiempo.

---

## 📷 Galería y Demostración

> *Espacio reservado para agregar capturas de pantalla, diagramas y resultados visuales del proyecto.*

### Disposición de las Cámaras (Setup)
<!-- PLACEHOLDER_IMG_SETUP: Agrega aquí un diagrama o foto del ángulo y posición de las cámaras -->
![Disposición de Cámaras](https://via.placeholder.com/800x400.png?text=Placeholder:+Diagrama+o+Foto+del+Setup+de+Dos+Cámaras)

---

### Detección 2D y Triangulación Epipolar
<!-- PLACEHOLDER_IMG_2D_DETECTION: Agrega aquí capturas de MediaPipe detectando los landmarks en ambas cámaras -->
![Detección 2D en Dos Cámaras](https://via.placeholder.com/800x400.png?text=Placeholder:+Detección+MediaPipe+en+Cámara+1+y+Cámara+2)

---

### Reconstrucción del Esqueleto 3D
<!-- PLACEHOLDER_IMG_3D_SKELETON: Agrega aquí una imagen de la nube de puntos 3D o del esqueleto reconstruido -->
![Reconstrucción 3D](https://via.placeholder.com/800x400.png?text=Placeholder:+Esqueleto+3D+Triangulado+y+Filtrado)

---

### Resultado Final en Blender
<!-- PLACEHOLDER_IMG_BLENDER_RESULT: Agrega aquí un render o captura del viewport de Blender animando el modelo -->
![Resultado en Blender](https://via.placeholder.com/800x400.png?text=Placeholder:+Modelo+3D+Animado+en+Blender)
