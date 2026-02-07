# Plan: Imagen → Detección SW → Simulación HW

## Objetivo

Poder meter **una imagen real** en el sistema, detectar un objeto/cara por **software**, y que la **misma imagen** fluya por la tubería hasta que el **sistema (simulación HW)** sea capaz de “detectarlo” también — es decir, que los resultados de la red en RTL coincidan con los de la referencia Python.

## ¿Es un salto muy grande?

- **Sí, si lo hacemos de golpe:** tener “la imagen entra en RTL y sale la caja detectada” implica una **DPU completa en RTL** (todas las capas, buffers, control, pre/post-proceso). Eso es mucho más que lo que tenemos hoy (primitivas + array 2×2 + stub).
- **No, si lo hacemos por etapas:** podemos ir cerrando el ciclo paso a paso:
  1. **SW detecta** en una imagen (YOLO o detector de caras).
  2. **Misma imagen** se convierte a tensor INT8 (entrada YOLOv4-tiny) y se hace correr la **primera capa (y las que queramos)** en el **modelo Python** (Phase 3).
  3. Se **exporta** la entrada de la capa (y opcionalmente la salida) para usarla después en RTL.
  4. Cuando tengamos **una capa completa en RTL** (o un bloque que ejecute una conv), alimentamos esa misma entrada y **comparamos** salida RTL vs Python.
  5. Más adelante: **varias capas** en RTL y, al final, **toda la red** en RTL → misma imagen → mismas cajas que en SW.

Así que **sí se puede**, pero como **proyecto por fases**, no en un solo paso.

## Etapas concretas

| Etapa | Qué hacemos | Estado |
|-------|-------------|--------|
| **1** | Script: cargar imagen → (opcional) detector SW (YOLO/caras) → tensor INT8 416×416×3 → primera capa en Python → guardar entrada (y salida) para RTL | **Hecho** (`run_image_to_detection.py`, guarda `image_input_layer0.npy` + pesos/bias) |
| **2a** | **Un píxel de la capa 0:** Golden Python (27 MACs + bias + LeakyReLU + requantize) con datos de la imagen guardada; TB RTL que ejecuta lo mismo y compara. | **Hecho** (`tests/layer0_patch_golden.py`, `rtl/tb/layer0_patch_tb_iv.sv`, `run_layer0_patch_check.py`) |
| **2b** | Región 4×4×32 de la capa 0 en RTL: TB `layer0_full_4x4_tb_iv.sv` + export `layer0_full_4x4_export.py` → **Hecho** (512 PASS). | **Hecho** |
| **2c** | Toda la capa 0 (208×208×32) en RTL: opcional (sim larga). | Futuro |
| **3** | Procesamiento imagen completo en Python: `run_image_to_detection.py --layers 2` guarda ref capa 0 + capa 1. Extender RTL a más capas. | **Ref Python lista**; RTL por capas. |

## Entregables etapa 1

- Script `run_image_to_detection.py` (o similar):
  - Carga una imagen (fichero o imagen de prueba).
  - Si hay detector disponible (ultralytics YOLO u OpenCV caras): corre y muestra “detecciones SW”.
  - Convierte la imagen a tensor INT8 (C, H, W) 416×416 (formato entrada YOLOv4-tiny).
  - Ejecuta la primera capa (conv_bn_leaky) con pesos placeholder/aleatorios en el modelo Python.
  - Guarda el tensor de **entrada de la capa** (p. ej. `.npy`) para futura comparación con RTL.
- Documentación: este plan y cómo ejecutar el script.

Cuando en el futuro tengamos una capa en RTL, usaremos ese mismo tensor guardado como entrada y compararemos salida RTL vs salida Python.

---

## Paso “un píxel” (Etapa 2a) — Hecho

- **Golden Python:** `tests/layer0_patch_golden.py` carga la imagen y pesos guardados, extrae el parche 3×3×3 para el píxel (0,0) y calcula la salida para **4 canales** (27 MACs + bias + LeakyReLU + requantize por canal; escala fija 655/65536 para coincidir con RTL). Escribe `layer0_patch_golden.json` y `layer0_patch_w0..w3.hex`, `layer0_patch_a.hex`, `layer0_patch_bias.hex`, `layer0_patch_expected.hex` en `image_sim_out/`.
- **TB RTL:** `rtl/tb/layer0_patch_tb_iv.sv` lee los .hex, para cada uno de los 4 canales ejecuta 27 ciclos de `mac_int8`, suma bias, pasa por `leaky_relu` y `requantize`, y compara la salida INT8 con el golden.
- **Orquestador:** `run_layer0_patch_check.py` asegura datos de imagen (ejecuta `run_image_to_detection --synthetic` si hace falta), ejecuta el golden, opcionalmente la sim RTL (si iverilog está en PATH) y muestra PASS/FAIL.
- **Salida completa capa 0:** `run_image_to_detection.py` guarda además `layer0_output_ref.npy` (salida completa de la capa 0 en Python) para comparación futura cuando exista un runner RTL de la capa completa.
- **Uso:** `python run_layer0_patch_check.py` (o `--python-only` si no tienes iverilog).
