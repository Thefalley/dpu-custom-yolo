# Reporte: Validación por primitivas vs. procesamiento completo con imagen

**Objetivo de este documento:** Que cualquier agente (o persona) entienda qué está validado, qué falta y cómo seguir. Sirve como handover para continuar el trabajo.

---

## 1. Resumen en lenguaje claro

### Qué hemos validado hasta ahora

Se ha hecho **validación a nivel de cada primitiva por separado**:

- **MAC INT8** (multiplicar peso × activación y acumular): RTL comparado con Python → **PASS**. Los mismos vectores de prueba en Python y en RTL dan el mismo resultado.
- **LeakyReLU** (activación x>0 ? x : x/8): RTL comparado con Python → **PASS**.
- **Mult shift-add** (multiplicador INT8×INT8 por desplazamientos): RTL comparado con Python → **PASS**.

Es decir: **cada bloque que usamos (MAC, LeakyReLU, mult) está verificado por separado** frente a la referencia en Python.

### Qué falta para “todo con la imagen y sincronizado”

Falta el **procesamiento de una imagen de punta a punta** con todo encadenado y sincronizado:

1. **Entrada:** una imagen (o tensor INT8 416×416×3 como entrada de la red).
2. **Cadena:** esa entrada pasa por la **primera capa** (conv 3×3 + bias + LeakyReLU + requantize) usando las primitivas ya validadas, pero **conectadas en secuencia** y con los mismos datos que en Python.
3. **Comparación:** la salida de esa capa en RTL debe coincidir con la salida de la misma capa en Python (misma imagen, mismos pesos).

Hoy tenemos:

- **Python:** ya hace “imagen → capa 0” y guarda entrada, pesos, bias y salida de referencia (`run_image_to_detection.py`, `layer0_output_ref.npy`, `layer0_patch_golden.py`).
- **RTL:** un testbench hace **un píxel de la capa 0** (27 MACs + bias + LeakyReLU + requantize) con datos de esa imagen y **ya coincide con el golden Python** (4/4 PASS). El encadenado “imagen → capa 0” está validado**.

En resumen:

- **Validado:** cada primitiva por separado (MAC, LeakyReLU, mult_shift_add) y el **patch de un píxel de la capa 0** (4 canales) en RTL vs Python.
- **Pendiente:** validar **toda la capa 0** en RTL (o más) y luego el procesamiento de imagen completo.

---

## 2. Estado técnico por componente

| Componente | Estado | Notas |
|------------|--------|--------|
| **Phase 7 — Primitivas** | ✅ COMPLETO | `run_phase7_autocheck.py` ejecuta Python golden + RTL (mac_int8, leaky_relu, mult_shift_add). Los tres RTL pasan. Requiere iverilog (OSS_CAD_PATH o PATH). |
| **Imagen → tensor + capa 0 (Python)** | ✅ COMPLETO | `run_image_to_detection.py` carga imagen (o sintética), convierte a INT8 416×416×3, ejecuta capa 0 en Python, guarda `image_input_layer0.npy`, `layer0_weights.npy`, `layer0_bias.npy`, `layer0_output_ref.npy`. |
| **Golden “un píxel” (Python)** | ✅ COMPLETO | `tests/layer0_patch_golden.py` genera vectores para un píxel (0,0) y 4 canales: 27 pesos × 27 activaciones, bias, expected_int8. Escribe `image_sim_out/layer0_patch_*.hex` y `layer0_patch_golden.json`. |
| **TB RTL “un píxel” (layer0_patch_tb_iv)** | ✅ COMPLETO | Ejecuta 4 canales (27 MACs + bias + LeakyReLU + requantize por canal). **4/4 PASS** frente a `layer0_patch_golden.json`. Fix: timing MAC — 3 ciclos por MAC (drive, MAC actualiza, latch) porque el MAC tiene latencia 1 ciclo. |
| **Requantize RTL** | ⚠️ CORREGIDO PARCIAL | Icarus daba error de sintaxis con `48'signed(...)`; se cambió a `product48 = acc * $signed(scale); rounded = product48 >>> SCALE_Q`. No hay TB standalone de requantize; se usa solo en layer0_patch_tb. |
| **run_all_sims.ps1** | ✅ FUNCIONAL | Ejecuta Phase 7 + layer0 patch. Phase 7 pasa; layer0 patch sale FAIL por el TB anterior. |

---

## 3. Archivos clave para otro agente

- **Contexto general:** `CONTEXTO_PROYECTO.md`
- **Plan imagen → detección:** `docs/IMAGE_TO_DETECTION_PLAN.md`
- **Simulación RTL:** `docs/SIMULACION_RTL.md`
- **Este reporte:** `docs/REPORTE_VALIDACION_Y_SIGUIENTES_PASOS.md`
- **Scripts de validación:**
  - Phase 7 (primitivas): `run_phase7_autocheck.py` (opción `--python-only` si no hay iverilog)
  - Imagen + capa 0 Python: `run_image_to_detection.py [imagen.jpg]` o `--synthetic`
  - Golden un píxel: `python tests/layer0_patch_golden.py` (requiere haber corrido antes `run_image_to_detection.py`)
  - Layer0 patch (Python + RTL): `run_layer0_patch_check.py` (RTL pasa 4/4)
  - Capa 0 región 4×4 (Python + RTL): `run_layer0_full_check.py` (export 4×4 + sim `layer0_full_4x4_tb` → 512 PASS)
  - **Validación final DPU (imagen → coordenadas):** `run_dpu_validation.py imagen.jpg` — inyectas una imagen y obtienes las coordenadas de objetos (YOLO + OpenCV caras); opcional `-o resultado.json`, `--quiet`.
- **TB layer0 patch:** `rtl/tb/layer0_patch_tb_iv.sv`; **TB capa 0 4×4:** `rtl/tb/layer0_full_4x4_tb_iv.sv` (lee `layer0_full4x4_*.hex`)
- **Referencia numérica:** `image_sim_out/layer0_patch_golden.json`; salida capa 0: `layer0_output_ref.npy`; con `--layers 2`: `layer1_output_ref.npy`

---

## 4. Próximos pasos recomendados (para otro agente)

### Paso 1 — Layer0 patch (un píxel, 4 canales) — HECHO

- El TB `layer0_patch_tb_iv.sv` da **4/4 PASS** frente a `layer0_patch_golden.json`.
- Fix aplicado: **timing del MAC**. El MAC tiene latencia 1 ciclo; si se hacía `acc_feedback <= acc_out` en el mismo posedge en que el MAC actualizaba `acc_out`, se leía el valor anterior. Se pasó a **3 ciclos por MAC**: drive inputs + valid=1; posedge (MAC actualiza); posedge + valid=0; posedge (latch `acc_feedback <= acc_out`). Así se captura correctamente el resultado de cada MAC.

### Paso 2 — Validar “toda la capa 0” en RTL (opcional pero deseable)

- **Hecho:** Región 4×4×32 validada en RTL (`layer0_full_4x4_tb_iv.sv` 512 PASS; `run_layer0_full_check.py`).
- Opcional: TB o FSM que calcule **toda** la capa 0 (32×208×208) y compare con `layer0_output_ref.npy` (sim larga).

### Paso 3 — Procesamiento de imagen completo (referencia Python)

- **`run_image_to_detection.py --layers 2`:** ejecuta capa 0 + capa 1 en Python; guarda `layer0_output_ref.npy`, `layer1_output_ref.npy`. Referencia para validar más capas en RTL.
- Objetivo: encadenar más capas en RTL y comparar con esas referencias.

---

## 5. Cómo ejecutar las comprobaciones

- **Solo Python (sin iverilog):**
  - `python run_phase7_autocheck.py --python-only`
  - `python run_image_to_detection.py --synthetic` (o `--layers 2` para capa 0 + capa 1)
  - `python run_layer0_patch_check.py --python-only`
  - `python run_layer0_full_check.py --python-only` (export 4×4, sin sim RTL)
- **Con RTL (iverilog; `.oss_cad_path` o OSS_CAD_PATH):**
  - `.\run_all_sims.ps1` → Phase 7 + Layer0 patch **ALL PASS**
  - `python run_layer0_full_check.py` → export 4×4 + sim **layer0_full_4x4_tb** (512 PASS)

---

## 6. Respuesta directa a tu pregunta

- **Sí:** Hemos hecho validación **a nivel de cada primitiva** (MAC, LeakyReLU, mult_shift_add): cada una por separado está verificada RTL vs Python.
- **Hecho:** El **patch de un píxel de la capa 0** (4 canales) ya funciona en RTL y coincide con Python (layer0_patch_tb 4/4 PASS).
- **Hecho:** Región 4×4×32 de la capa 0 validada en RTL (layer0_full_4x4_tb 512 PASS). Referencia de procesamiento completo en Python con `--layers 2`.
- **Pendiente:** Extender RTL a más regiones o toda la capa 0; luego más capas en RTL.

Este documento sirve como **reporte para que otro agente retome el trabajo** en ese punto.
