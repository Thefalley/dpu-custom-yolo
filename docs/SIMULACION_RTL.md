# Simulación RTL (Phase 6)

Instrucciones para simular los testbenches del DPU. Hay **dos opciones**: con **clocking blocks** (recomendado para depuración) o con **Icarus Verilog** local.

---

## 1. Por qué clocking blocks

Los **clocking blocks** (y `program`) aseguran que:

- Las señales se **actualizan en el momento del reloj**, no en medio del ciclo.
- El DUT **ve valores estables** antes del flanco; no se “coge el valor antiguo”.
- Se evitan **carreras** entre testbench y DUT.

**Icarus Verilog no soporta** `program` ni `clocking`. Para usarlos hace falta otra alternativa (EDA Playground o ModelSim/VCS).

---

## 2. Alternativa con clocking blocks: EDA Playground

Para usar **program + clocking block** sin instalar otro simulador:

1. Entra en **[EDA Playground](https://www.edaplayground.com/)**.
2. **Design:** sube (o pega) en orden:
   - `rtl/dpu/primitives/mac_int8.sv`
   - `rtl/tb/mac_int8_tb.sv`  (el que tiene `program` y `default clocking`)
3. **Testbench:** deja vacío o no lo uses como “testbench” separado si ya está todo en `mac_int8_tb.sv`.
4. **Simulator:** elige **ModelSim** o **VCS** (soportan clocking).
5. **Run** → verás la consola con PASS/FAIL y podrás ver waveforms.

Mismo esquema para `leaky_relu_tb.sv` y `mult_shift_add_tb.sv` (cada uno con su primitiva).

**Archivos con clocking (para EDA Playground / ModelSim / VCS):**

| Test   | RTL primitiva              | Testbench (con clocking)   |
|--------|----------------------------|----------------------------|
| MAC    | `rtl/dpu/primitives/mac_int8.sv` | `rtl/tb/mac_int8_tb.sv`   |
| LeakyReLU | `rtl/dpu/primitives/leaky_relu.sv` | `rtl/tb/leaky_relu_tb.sv` |
| Mult   | `rtl/dpu/primitives/mult_shift_add.sv` | `rtl/tb/mult_shift_add_tb.sv` |

---

## 3. Simulación local con Icarus (verilog-sim-py + OSS CAD)

Icarus no soporta clocking, así que se usan los **testbenches `*_iv.sv`** (misma disciplina: tasks + `<=` y `@(posedge clk)`).

### 3.1 OSS CAD Suite en el PATH

En PowerShell, desde la carpeta del proyecto (adaptando la ruta si hace falta):

```powershell
$oss = "C:\project\upm\oss-cad-suite\oss-cad-suite"
$env:PATH = "$oss\bin;$oss\lib;$env:PATH"
```

### 3.2 Ejecutar (sin GTKWave)

Desde la raíz del proyecto:

```powershell
python verilog-sim-py/sv_simulator.py --no-wave rtl/dpu/primitives/mac_int8.sv rtl/tb/mac_int8_tb_iv.sv --top mac_int8_tb

python verilog-sim-py/sv_simulator.py --no-wave rtl/dpu/primitives/leaky_relu.sv rtl/tb/leaky_relu_tb_iv.sv --top leaky_relu_tb

python verilog-sim-py/sv_simulator.py --no-wave rtl/dpu/primitives/mult_shift_add.sv rtl/tb/mult_shift_add_tb_iv.sv --top mult_shift_add_tb
```

### 3.3 Script todo-en-uno

```powershell
.\run_dpu_sim.ps1 mac_int8
.\run_dpu_sim.ps1 leaky_relu
.\run_dpu_sim.ps1 mult_shift_add
```

`run_dpu_sim.ps1` usa los `*_iv.sv` para Icarus.

---

## 4. Resumen de archivos

| Archivo              | Uso                                      |
|----------------------|------------------------------------------|
| `rtl/tb/mac_int8_tb.sv`       | Clocking + program → EDA Playground / ModelSim / VCS |
| `rtl/tb/mac_int8_tb_iv.sv`   | Sin clocking → Icarus local              |
| `rtl/tb/leaky_relu_tb.sv`    | Clocking → EDA Playground / ModelSim / VCS |
| `rtl/tb/leaky_relu_tb_iv.sv` | Icarus local                             |
| `rtl/tb/mult_shift_add_tb.sv`| Clocking → EDA Playground / ModelSim / VCS |
| `rtl/tb/mult_shift_add_tb_iv.sv` | Icarus local                          |

---

## 5. Si tienes ModelSim/Questa/VCS local

Puedes simular los `.sv` **con clocking** (sin `_iv`) con tu herramienta habitual; el flujo depende del simulador. Los mismos archivos que en EDA Playground sirven para ModelSim/Questa/VCS.

---

**Resumen:** Para **clocking blocks** y depuración sin carreras, usa **EDA Playground** (o ModelSim/VCS) con los TB `*.sv` (sin `_iv`). Para **Icarus** en local, usa los TB `*_iv.sv` y `run_dpu_sim.ps1` o los comandos de la sección 3.
