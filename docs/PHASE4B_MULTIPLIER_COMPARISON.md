# PHASE 4B: Multiplier Architecture Exploration

## Resultado Sorprendente

**El enfoque LUT-based (shift-and-add) es MAS RAPIDO que DSP en ZedBoard!**

Esto ocurre porque:
1. DSPs son limitados (220 en Zynq-7020)
2. LUTs permiten arrays mas grandes
3. El mayor paralelismo compensa la menor frecuencia

---

## Target: ZedBoard (Zynq-7020)

| Recurso | Disponible |
|---------|------------|
| DSP48E1 | 220 |
| LUTs | 53,200 |
| FFs | 106,400 |
| BRAM | 630 KB |

---

## Tipos de Multiplicadores Analizados

### DSP-Based

| Tipo | DSPs | LUTs | Latencia | Freq |
|------|------|------|----------|------|
| DSP Native | 1.0 | 20 | 3 cyc | 200 MHz |
| DSP Packed 2x | 0.5 | 40 | 4 cyc | 180 MHz |

### LUT-Based (Shift-and-Add)

| Tipo | DSPs | LUTs | Latencia | Freq |
|------|------|------|----------|------|
| LUT Parallel | 0 | 64 | 4 cyc | 150 MHz |
| LUT Serial | 0 | 24 | 8 cyc | 200 MHz |
| LUT Booth R4 | 0 | 40 | 4 cyc | 180 MHz |
| **LUT Pipelined** | **0** | **48** | **2 cyc** | **160 MHz** |

---

## Configuraciones MAC Array en ZedBoard

| Configuracion | Array | MACs/cyc | Freq | DSPs | LUTs |
|--------------|-------|----------|------|------|------|
| DSP Native | 12x12 | 144 | 200 MHz | 144 | 4K |
| DSP Packed | 20x20 | 400 | 180 MHz | 200 | 20K |
| **LUT Pipelined** | **24x24** | **576** | **160 MHz** | **0** | **34K** |
| LUT Parallel | 20x20 | 400 | 150 MHz | 0 | 32K |

---

## Rendimiento YOLOv4-tiny

| Configuracion | FPS | DSPs | LUTs | Ranking |
|--------------|-----|------|------|---------|
| **LUT Pipelined 24x24** | **24.26** | **0** | **34K** | **#1** |
| DSP Packed 20x20 | 18.95 | 200 | 20K | #2 |
| Hybrid 50/50 20x20 | 18.95 | 200 | 17K | #3 |
| LUT Parallel 20x20 | 15.79 | 0 | 32K | #4 |
| DSP Native 12x12 | 7.58 | 144 | 4K | #5 |

---

## Analisis

### Por que LUT gana en ZedBoard?

```
DSP Packed:     400 MACs × 180 MHz = 72 GMACs/s
LUT Pipelined:  576 MACs × 160 MHz = 92 GMACs/s  (+28%!)
```

El array LUT es **44% mas grande** (576 vs 400), lo que compensa la frecuencia 11% menor.

### Comparacion Visual

```
DSP-Based (20x20 = 400 MACs):
+--------------------+
|  ■ ■ ■ ■ ■ ■ ■ ■ ■ |
|  ■ ■ ■ ■ ■ ■ ■ ■ ■ |
|  ■ ■ ■ ■ ■ ■ ■ ■ ■ |
|  ...               |  <- Limitado por 220 DSPs
+--------------------+

LUT-Based (24x24 = 576 MACs):
+------------------------+
|  □ □ □ □ □ □ □ □ □ □ □ |
|  □ □ □ □ □ □ □ □ □ □ □ |
|  □ □ □ □ □ □ □ □ □ □ □ |
|  □ □ □ □ □ □ □ □ □ □ □ |
|  ...                   |  <- Sin limite de DSP
+------------------------+
```

---

## Multiplicador Shift-and-Add

### Principio

```
A × B = Σ (A × B[i]) << i   para i = 0 a 7
```

### Hardware (Pipelined)

```
        B[7:0]
          |
    +-----+-----+
    |           |
    v           v
+-------+   +-------+
| B[3:0]|   | B[7:4]|
+---+---+   +---+---+
    |           |
    v           v
+-------+   +-------+
|Partial|   |Partial|   Stage 1: Partial Products
|Prods 0|   |Prods 1|
+---+---+   +---+---+
    |           |
    v           v
+-------+   +-------+
| Adder |   | Adder |   Stage 2: Add Trees
| Tree  |   | Tree  |
+---+---+   +---+---+
    |           |
    +-----+-----+
          |
          v
    +-----------+
    | Final Add |       Stage 3: Combine
    +-----------+
          |
          v
      Product
```

### Recursos por Multiplicador

| Componente | LUTs | FFs |
|------------|------|-----|
| Partial Products (AND gates) | 16 | 0 |
| Adder Tree (4-level) | 24 | 32 |
| Pipeline Registers | 0 | 16 |
| Control | 8 | 16 |
| **TOTAL** | **48** | **64** |

---

## Ventajas para ASIC

| Aspecto | DSP-Based | LUT-Based |
|---------|-----------|-----------|
| Portabilidad | Solo FPGA | FPGA + ASIC |
| Escalabilidad | Limitado | Ilimitado |
| Personalizacion | Fija | Flexible |
| Sintesis ASIC | Requiere remap | Directo |
| Verificacion | Diferente en ASIC | Identico |

---

## Recomendacion Final

### Para ZedBoard (y FPGAs similares)

**Usar LUT Pipelined (24x24)**
- 24.26 FPS en YOLOv4-tiny
- 0 DSPs utilizados
- 34K LUTs (64% utilizacion)
- Mejor rendimiento que DSP!

### Para camino hacia ASIC

**LUT-based es ideal porque:**
1. Mismo RTL funciona en FPGA y ASIC
2. Sin bloques propietarios que reemplazar
3. Estructura regular, facil de sintetizar
4. Verificacion en FPGA valida para ASIC

---

## Configuracion Seleccionada

```
+--------------------------------------------------+
|        DPU ARCHITECTURE (LUT-BASED)              |
+--------------------------------------------------+
|  Multiplicador:    Shift-and-Add Pipelined       |
|  MAC Array:        24 x 24 = 576 MACs/cycle      |
|  Frequency:        160 MHz                        |
|  Peak Performance: 0.092 TOPS                     |
+--------------------------------------------------+
|  RESOURCES (ZedBoard):                            |
|    DSPs:           0 / 220 (0%)                   |
|    LUTs:           34,377 / 53,200 (65%)          |
|    FFs:            45,004 / 106,400 (42%)         |
+--------------------------------------------------+
|  PERFORMANCE:                                     |
|    YOLOv4-tiny:    24.26 FPS                      |
|    vs DSP-based:   +28% mas rapido!               |
+--------------------------------------------------+
```

---

## Archivos Generados

| Archivo | Descripcion |
|---------|-------------|
| `phase4b_multiplier_exploration.py` | Script de analisis |
| `phase4b_multiplier_results.json` | Resultados JSON |
| `docs/PHASE4B_MULTIPLIER_COMPARISON.md` | Este documento |

---

**Phase 4B Status: COMPLETE**
**Seleccionado: LUT Pipelined (Shift-and-Add)**
**FPS: 24.26 (28% mejor que DSP!)**
