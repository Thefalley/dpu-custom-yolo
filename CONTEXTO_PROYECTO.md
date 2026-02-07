# Contexto del proyecto — DPU Custom YOLO

Documento único para entender el estado del proyecto y por dónde seguir.

---

## 1. Qué es este proyecto

- **Objetivo:** Diseñar y validar una **DPU (Deep Learning Processing Unit)** custom optimizada para un modelo YOLO “hardware-friendly”.
- **Metodología:** Bottom-up, por fases; cada fase se documenta y no se salta.
- **Modelo elegido:** **YOLOv4-tiny** (LeakyReLU, sin SiLU; ver `docs/PHASE1_MODEL_SELECTION.md`).
- **Workload:** ~3.45 GMACs, 21 capas conv, entrada 416×416.
- **Fases totales:** 7 (análisis software → primitivas → modelo Python → exploración arquitectura → definición HW → RTL → verificación). FPGA/Vivado/Vitis quedan fuera de estas fases.

---

## 2. Estado actual por fase

| Fase | Nombre | Estado | Qué hay |
|------|--------|--------|---------|
| **1** | YOLO model analysis | **HECHO** | `phase1_yolov4tiny_analysis.py`, `docs/PHASE1_*.md`, `yolov4-tiny.cfg`. Tabla capa a capa, MACs, tipos de conv. |
| **2** | DPU primitives | **HECHO** | `phase2_dpu_primitives.py`, `docs/PHASE2_DPU_PRIMITIVES.md`. Lista: MAC, Conv 3×3/1×1, LeakyReLU, Requantize, Bias, MaxPool, Route. |
| **3** | Python functional model | **HECHO** | `phase3_dpu_functional_model.py`, `docs/PHASE3_FUNCTIONAL_MODEL.md`. Funciones Python que imitan el HW (INT8/INT32). Referencia para verificación RTL. |
| **4** | Architecture exploration | **HECHO** | `phase4_architecture_exploration.py`, `docs/PHASE4_ARCHITECTURE_EXPLORATION.md`. Configuraciones TINY→XLARGE. **Elegida: MEDIUM 32×32** (~54 FPS, 1024 MACs/ciclo). |
| **4b** | Multiplier exploration | **HECHO** | `phase4b_multiplier_exploration.py`, `docs/PHASE4B_MULTIPLIER_EXPLORATION.md`. **DSP vs LUT (shift-and-add)**. Permite ASIC y FPGAs con pocos DSPs (ej. ZedBoard). Recomendación LUT: pipelined 2-stage. |
| **5** | Hardware architecture | **HECHO** | `docs/PHASE5_HARDWARE_ARCHITECTURE.md`. Datapath, control, memorias, interfaces internas/externas (AXI futura), diagrama de bloques, mapeo Python→HW. |
| **6** | RTL (Verilog) | **HECHO** | Primitivas en `rtl/dpu/primitives/`, subsistema `mac_array_2x2`, stub `dpu_top_stub`, testbenches en `rtl/tb/`. Simulación con verilog-sim-py + OSS CAD (iverilog). |
| **7** | Verification (AutoCheck) | **HECHO** | `run_phase7_autocheck.py`, `docs/PHASE7_VERIFICATION.md`. Compara RTL vs Python (primitivas); informe pass/fail. |

---

## 3. Decisiones importantes

- **YOLOv4-tiny (no v5/v8):** LeakyReLU en lugar de SiLU; más fácil de mapear a HW (comparador + shifter + MUX).
- **Arquitectura base:** MEDIUM 32×32 MACs, output-stationary, buffers 128 KB pesos, 64 KB entrada/salida.
- **Multiplicador:** Se puede elegir **DSP** (1 por MAC) o **LUT shift-and-add** (Phase 4b); mismo comportamiento funcional, distinto uso de recursos. Para ASIC o ZedBoard se usa la opción LUT.

---

## 4. Estructura del repositorio (resumida)

```
dpu-custom-yolo/
├── CONTEXTO_PROYECTO.md          ← Este archivo
├── README.md
├── docs/                         # Documentación por fase
│   ├── PHASE1_ANALYSIS.md, PHASE1_MODEL_SELECTION.md
│   ├── PHASE2_DPU_PRIMITIVES.md
│   ├── PHASE3_FUNCTIONAL_MODEL.md
│   ├── PHASE4_ARCHITECTURE_EXPLORATION.md
│   ├── PHASE4B_MULTIPLIER_EXPLORATION.md
│   ├── PHASE5_HARDWARE_ARCHITECTURE.md
│   ├── PHASE6_RTL_IMPLEMENTATION.md
│   ├── PHASE7_VERIFICATION.md    # AutoCheck RTL vs Python
│   ├── IMAGE_TO_DETECTION_PLAN.md # Plan imagen → detección SW → simulación HW
│   ├── REPORTE_VALIDACION_Y_SIGUIENTES_PASOS.md  # Handover: qué está validado, qué falta (imagen + sincronizado)
│   └── SIMULACION_RTL.md         # Cómo simular RTL (iverilog vs EDA Playground)
├── rtl/
│   ├── dpu/
│   │   ├── primitives/           # mac_int8, leaky_relu, requantize, mult_shift_add
│   │   ├── mac_array_2x2.sv      # Array 2×2 de MACs (para pruebas)
│   │   └── dpu_top_stub.sv       # Stub del top (sin buffers/control completos)
│   ├── tb/                       # Testbenches
│   │   ├── mac_int8_tb.sv        # Con program + clocking (EDA Playground / ModelSim)
│   │   ├── mac_int8_tb_iv.sv     # Sin clocking, para Icarus
│   │   ├── leaky_relu_tb.sv / leaky_relu_tb_iv.sv
│   │   └── mult_shift_add_tb.sv / mult_shift_add_tb_iv.sv
│   └── run_sim.py
├── tests/
│   └── test_rtl_vectors.py       # Golden Python (mismos vectores que los TB RTL)
├── verilog-sim-py/               # Repo clonado: simulación con iverilog + GTKWave
├── phase1_yolov4tiny_analysis.py
├── phase2_dpu_primitives.py
├── phase3_dpu_functional_model.py
├── phase4_architecture_exploration.py
├── phase4b_multiplier_exploration.py
├── run_dpu_sim.ps1               # Script simulación RTL (usa *_iv.sv con iverilog)
├── run_phase7_autocheck.py       # Phase 7: verificación RTL vs Python (un comando)
├── run_image_to_detection.py     # Imagen → detector SW → tensor INT8 → capa 0 Python; guarda entrada para RTL
├── run_layer0_patch_check.py    # Un píxel capa 0: golden Python + TB RTL (imagen real)
├── run_all_sims.ps1              # Todas las sims: Phase 7 + Layer0 patch (configurar OSS_CAD_PATH)
├── check_sim_tools.py            # Comprobar si iverilog/vvp están disponibles
├── yolov4-tiny.cfg
└── yolov3-tiny.cfg
```

---

## 5. Cómo ejecutar cosas

### Python (referencia y análisis)

- Fase 1: `python phase1_yolov4tiny_analysis.py`
- Fase 4: `python phase4_architecture_exploration.py`
- Fase 4b: `python phase4b_multiplier_exploration.py`
- Golden para RTL: `python tests/test_rtl_vectors.py` (vectores MAC, LeakyReLU, mult)
- **Phase 7 AutoCheck:** `python run_phase7_autocheck.py` (Python golden + simulaciones RTL; requiere iverilog en PATH para RTL). Opciones: `--python-only`, `--rtl-only`, `-q`.
- **Imagen → detección (Stage 1):** `python run_image_to_detection.py [imagen.jpg]` o `--synthetic`. Detector SW opcional (OpenCV/YOLO); guarda entrada capa 0 en `sim_out/image_input_layer0.npy` (o `image_sim_out/`) para RTL.
- **Layer0 patch (un píxel, 4 canales, imagen real):** `python run_layer0_patch_check.py` (o `--python-only`). Golden Python + TB RTL comparan 4 canales del píxel (0,0) de la capa 0; requiere iverilog para RTL.

### RTL (simulación)

- **Todas las simulaciones (Phase 7 + Layer0 patch):** Configurar iverilog (OSS CAD Suite o Icarus) y ejecutar:
  - `$env:OSS_CAD_PATH = "C:\ruta\a\oss-cad-suite\oss-cad-suite"` (o editar `$OSS_CAD_PATH` en `run_all_sims.ps1`).
  - `python check_sim_tools.py` — comprueba si iverilog/vvp están disponibles.
  - `.\run_all_sims.ps1` — ejecuta Phase 7 (primitivas) + Layer0 patch (4 canales). Ver `docs/SIMULACION_RTL.md`.
- **Solo primitivas (una a una):** `.\run_dpu_sim.ps1 mac_int8` (o leaky_relu, mult_shift_add). Icarus no soporta `program`/`clocking`; se usan los TB `*_iv.sv`.
- **Con clocking blocks (depuración):** EDA Playground (edaplayground.com) con ModelSim/VCS; TB sin `_iv`. Ver `docs/SIMULACION_RTL.md`.

---

## 6. Por dónde seguir

### Opción A — Completar RTL

- Añadir control FSM y buffers (o stubs) al `dpu_top_stub`.
- Probar mac_array más grande (ej. 4×4 u 8×8) si quieres acercarte al MEDIUM 32×32.
- Integrar requantize en el datapath y probarlo.

### Opción B — Síntesis / FPGA

- Fuera del alcance de las 7 fases actuales, pero el diseño está preparado para pasar a Vivado/Vitis cuando toque.

### Opción C — Ampliar verificación (Phase 7+)

- Incluir **mac_array_2x2** en el AutoCheck (TB + golden) y, cuando exista, un test del DPU completo.

### Opción D — Simulación a nivel imagen (imagen → detección)

- **Objetivo:** Meter una imagen real, detectar por SW (cara/objeto) y que la misma imagen fluya por la tubería hasta que el sistema (simulación HW) “detecte” lo mismo.
- **Plan por etapas:** Ver `docs/IMAGE_TO_DETECTION_PLAN.md`. Etapa 1: `run_image_to_detection.py` (imagen → tensor → capa 0 Python, guarda entrada). Etapa 2a (hecha): **un píxel** de la capa 0 con datos de imagen: `run_layer0_patch_check.py` (golden Python + TB RTL que comparan 27 MACs + bias + LeakyReLU + requantize). Siguiente: runner de toda la capa en RTL.

---

## 7. Resumen en una frase

Tienes un **DPU definido en Python y en RTL** (primitivas + array pequeño + stub de top), **documentado por fases**, con **dos opciones de multiplicador** (DSP o shift-and-add), **testbenches con y sin clocking blocks**, y **Phase 7 cerrada**: verificación automática RTL vs Python con `run_phase7_autocheck.py`. Opcional: ampliar RTL (control, buffers, array más grande) o incluir mac_array_2x2 en el AutoCheck.
