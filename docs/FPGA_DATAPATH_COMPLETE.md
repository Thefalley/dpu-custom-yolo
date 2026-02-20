# FPGA DATAPATH COMPLETO — YOLOv4-tiny (Capas 0-36)

## Documento de referencia: Todos los calculos, anchos de bits, operaciones y pipeline

---

## 1. SISTEMA DE NUMEROS

**Todo el datapath es ENTERO PURO en complemento a dos. No hay coma flotante ni punto fijo con bits fraccionarios explicitos.**

| Tipo  | Bits | Rango                          | Donde se usa                          |
|-------|------|--------------------------------|---------------------------------------|
| INT8  | 8    | [-128, +127]                   | Pesos, activaciones, salidas de capa  |
| INT16 | 16   | [-32768, +32767]               | Producto de multiplicacion 8x8        |
| INT32 | 32   | [-2,147,483,648, +2,147,483,647] | Acumulador, bias, resultado LeakyReLU |
| INT48 | 48   | [-140,737,488,355,328, ...]    | Producto de requantizacion 32x16      |
| UINT16| 16   | [0, 65535]                     | Scale (factor de requantizacion)      |

**La unica "coma" es implicita:** cuando se multiplica por `scale` y se hace shift right 16, eso equivale a multiplicar por `scale/65536`. Pero el hardware nunca ve una coma; solo ve enteros y un desplazamiento.

---

## 2. TIPOS DE OPERACION

El DPU implementa 6 tipos de operacion:

| Codigo | Tipo              | Operacion                                     | Calculo | Bits in→out |
|--------|-------------------|-----------------------------------------------|---------|-------------|
| 0      | LT_CONV3X3        | Conv 3x3 + Bias + LeakyReLU + Requant         | SI      | INT8→INT8   |
| 1      | LT_CONV1X1        | Conv 1x1 + Bias + LeakyReLU + Requant         | SI      | INT8→INT8   |
| 2      | LT_ROUTE_SPLIT    | Tomar mitad de canales (group_id=1)            | NO      | INT8→INT8   |
| 3      | LT_ROUTE_CONCAT   | Concatenar 2 feature maps en eje de canales   | NO      | INT8→INT8   |
| 4      | LT_MAXPOOL        | Max de ventana 2x2 con stride 2               | MIN     | INT8→INT8   |
| 5      | LT_CONV1X1_LINEAR | Conv 1x1 + Bias + Requant (SIN LeakyReLU)     | SI      | INT8→INT8   |
| 6      | LT_UPSAMPLE       | Nearest neighbor 2x (duplicar cada pixel)      | NO      | INT8→INT8   |

---

## 3. PIPELINE DE CONVOLUCION (Conv 3x3 / Conv 1x1)

Este pipeline se usa en las capas: 0, 1, 2, 4, 5, 7, 10, 12, 13, 15, 18, 20, 21, 23, 26, 27, 28, 32, 35.

### 3.1 ETAPA 1: Multiplicacion (MAC Array)

**Modulo:** `mac_array_32x32.sv`
**Arquitectura:** 32 filas x 32 columnas = 1024 multiplicadores en paralelo

```
Entrada peso:        w[r][c]    signed [7:0]     8 bits
Entrada activacion:  a[c]       signed [7:0]     8 bits

Paso 1a — Multiplicacion:
  prod16 = w[r][c] * a[c]

  Tipo resultado: signed [15:0]    16 bits

  Mapa de bits del producto:
    Bit 15:     signo (complemento a 2)
    Bits 14-0:  magnitud

  Rango: min = -128 * 127 = -16,256
         max = +127 * 127 = +16,129
         Ambos caben en 16 bits signed [-32768, +32767]

  Ejemplo binario:
    w = -15 = 8'b1111_0001
    a = +42 = 8'b0010_1010
    prod16 = -630 = 16'b1111_1101_1000_1010 = 16'hFD8A

Paso 1b — Extension de signo (16 → 32 bits):
  product_sext = {{16{prod16[15]}}, prod16}

  Si prod16 es negativo (bit 15 = 1):
    Se rellena con 16 unos por la izquierda
    Ejemplo: 16'hFD8A → 32'hFFFF_FD8A = -630

  Si prod16 es positivo (bit 15 = 0):
    Se rellena con 16 ceros por la izquierda
    Ejemplo: 16'h0126 → 32'h0000_0126 = +294

  Mapa de bits:
    [31:16] = copia de bit 15 (extension de signo)
    [15:0]  = producto original

Paso 1c — Suma parcial de 32 productos (1 fila del array):
  partial[r] = SUM_{c=0}^{31} sext(w[r][c] * a[c])

  Tipo: signed [31:0]    32 bits
  Rango peor caso: 32 * 16,129 = 516,128 (cabe holgadamente en INT32)

Paso 1d — Acumulacion inter-ciclo:
  Si es el primer MAC del tile Cout (clear_acc=1):
    acc[r] = partial[r]
  Si no (clear_acc=0):
    acc[r] = acc[r] + partial[r]

  Tipo: signed [31:0]    32 bits
  Se acumula sobre multiples ciclos: ceil(Cin/32) tiles * K^2 posiciones kernel

  Rango peor caso por capa (todos pesos y activaciones al maximo):
    Layer 0:  Cin*K^2 = 3*9 = 27 MACs     → max |acc| = 27 * 16,129 = 435,483
    Layer 10: Cin*K^2 = 128*9 = 1152 MACs  → max |acc| = 1,152 * 16,129 = 18,580,608
    Layer 26: Cin*K^2 = 512*9 = 4608 MACs  → max |acc| = 4,608 * 16,129 = 74,258,432
    Layer 28: Cin*K^2 = 256*9 = 2304 MACs  → max |acc| = 2,304 * 16,129 = 37,161,216
    Limite INT32: 2,147,483,647 → NINGUN caso desborda

Latencia: 1 ciclo de reloj (registrado en always_ff @posedge clk)
  Ciclo T:   valid=1, datos en w_in y act_in
  Ciclo T+1: acc actualizado, done=1
```

### 3.2 ETAPA 2: Suma de Bias

**Modulo:** `post_process_array.sv`, Stage 1 (linea 77-79)

```
Entrada acc[i]:    signed [31:0]    32 bits (del MAC array)
Entrada bias[i]:   signed [31:0]    32 bits (precargado de bias_buf)
Salida biased[i]:  signed [31:0]    32 bits (registrado)

Operacion:
  biased[i] = acc[i] + bias[i]

No hay extension de bits — ambos son 32 bits, resultado 32 bits.
Puede haber overflow en teoria, pero en practica los bias son pequenos
comparados con el rango INT32.

El bias incluye el BatchNorm fusionado. En inferencia cuantizada INT8:
  bias_fused = round(beta - gamma * mean / sqrt(var + epsilon))
  Donde beta, gamma, mean, var son parametros del BatchNorm original.
  Se calcula offline y se graba como INT32.

Mapa de bits:
  [31]   signo
  [30:0] magnitud (complemento a 2)
  No hay coma. Entero puro.

Latencia: 1 ciclo de reloj
  Ciclo T:   valid llega al post_process
  Ciclo T+1: biased_r[i] registrado, v1=1
```

### 3.3 ETAPA 3: LeakyReLU

**Modulo:** `post_process_array.sv`, Stage 2 (linea 83-85)
**Modulo standalone:** `leaky_relu.sv`

```
Entrada biased[i]:  signed [31:0]    32 bits
Salida relu[i]:     signed [31:0]    32 bits (registrado)

Operacion:
  Si biased >= 0 (bit 31 = 0):
    relu = biased                    // pasa directo, sin cambio

  Si biased < 0 (bit 31 = 1):
    relu = biased >>> 3              // arithmetic right shift por 3
                                     // equivale a dividir por 8
                                     // aproximacion de alpha=0.1 con alpha=0.125

Arithmetic right shift (>>>) vs logical right shift (>>):
  >>> preserva el signo: rellena por la izquierda con el bit de signo
  >>  rellena con ceros

  Ejemplo >>> 3 con numero negativo:
    biased = -4050 = 32'hFFFFF02E
    Binario: 1111_1111_1111_1111_1111_0000_0010_1110

    >>> 3:   1111_1111_1111_1111_1111_1110_0000_0101
             ^^^                                 ^^^
             3 copias del bit de signo           3 bits perdidos
             insertadas por la izquierda         por la derecha

    Resultado: -507 (trunca hacia -infinito: -4050/8 = -506.25 → -507)

  Ejemplo >>> 3 con numero positivo:
    No aplica — los positivos pasan directo sin shift.

Mapa de bits despues del shift:
  ANTES:   [31] [30] [29] [28] [27] ... [3] [2] [1] [0]
  DESPUES: [31] [31] [31] [31] [30] ... [6] [5] [4] [3]
           ←── 3 bits nuevos ──→              ←── 3 perdidos

Por que 1/8 y no 0.1:
  Division por 8 = shift right 3 bits = 0 LUTs, 0 ciclos extra
  Division por 10 = multiplicador real = cientos de LUTs
  Error: |0.125 - 0.1| / 0.1 = 25%, pero irrelevante en INT8 cuantizado

Latencia: 1 ciclo de reloj
  Ciclo T+1: v1=1 (biased disponible)
  Ciclo T+2: relu_r[i] registrado, v2=1
```

### 3.4 ETAPA 4: Requantizacion (INT32 → INT8)

**Modulo:** `post_process_array.sv`, Stage 3 (lineas 45-53)
**Modulo standalone:** `requantize.sv`

**Este es el paso donde 32 bits se comprimen a 8 bits.**

```
Entrada relu[i]:     signed [31:0]    32 bits
Entrada scale:       [15:0]           16 bits (unsigned, tratado como signed via $signed)
Parametro SCALE_Q:   16               (constante, bits de shift)
Salida result[i]:    signed [7:0]     8 bits (registrado)

═══ Paso 4a: Multiplicacion 32 x 16 → 48 bits ═══

  prod_w[i] = relu_r[i] * $signed(scale)

  Tipo: signed [47:0]    48 bits

  $signed(scale) interpreta los 16 bits como signed.
  Como scale siempre es positivo (0 a 65535), y bit 15 = 0 para valores utiles,
  rango efectivo del scale: [0, 32767].

  Valor tipico: scale = 655 = 16'h028F = 16'b0000_0010_1000_1111

  Ejemplo:
    relu = +4050 = 32'h00000FD2
    scale = 655  = 16'h028F
    prod_w = 4050 * 655 = 2,652,750 = 48'h000000287B5E

  Ejemplo negativo:
    relu = -507 = 32'hFFFFFE05
    scale = 655
    prod_w = -507 * 655 = -332,085 = 48'hFFFFFFFAEE4B

  Bits necesarios: max(|relu|) * max(|scale|) = 2^31 * 2^15 = 2^46
  Cabe en 48 bits signed (rango 2^47). NO hay overflow posible.

═══ Paso 4b: Shift right por SCALE_Q=16 → 32 bits ═══

  rnd_w[i] = prod_w[i] >>> 16

  Tipo: signed [31:0]    32 bits

  Esto descarta los 16 bits menos significativos.
  Es equivalente a dividir por 2^16 = 65536.

  La "coma" implicita:
    scale = 655 representa el factor 655 / 65536 = 0.009994...
    La operacion completa:  resultado = (relu * 655) >> 16
    Es identica a:          resultado = relu * 0.009994
    Pero sin usar punto flotante.

  Ejemplo:
    prod_w = 2,652,750

    En binario (48 bits):
    0000_0000_0000_0000 | 0010_1000_0111_1011_0101_1110
    ←── estos 16 MSB ──→  ←── estos 32 bits quedan ──────→
    se insertan arriba     (pero solo los de arriba importan)

    Tras >>> 16:
    rnd_w = 2,652,750 / 65,536 = 40 (truncamiento entero)
    rnd_w = 32'h00000028 = 40

  Ejemplo negativo:
    prod_w = -332,085
    rnd_w = -332,085 >>> 16 = -6 (arithmetic shift, preserva signo)
    (En realidad: -332,085 / 65,536 = -5.07, trunca hacia -inf → -6)

═══ Paso 4c: Saturacion/Clamp → 8 bits ═══

  Si rnd_w > +127:   clamp = +127 = 8'b0111_1111  (saturacion positiva)
  Si rnd_w < -128:   clamp = -128 = 8'b1000_0000  (saturacion negativa)
  Si -128 <= rnd_w <= +127:  clamp = rnd_w[7:0]   (tomar 8 bits bajos)

  Deteccion de overflow (como sabe el HW si saturar):
    Si rnd_w[31:7] != {25{rnd_w[7]}} → overflow → saturar
    Es decir: si los bits 31 a 8 NO son extension de signo del bit 7,
    el valor no cabe en 8 bits.

    Ejemplo que satura:
      rnd_w = 499 = 32'h000001F3
      Bits [31:8] = 24'h000001 ≠ 24'h000000 (extension de bit 7=1)
      → satura a +127

    Ejemplo que no satura:
      rnd_w = 40 = 32'h00000028
      Bits [31:8] = 24'h000000, bit 7 = 0, extension = 24'h000000 ✓
      → clamp = 40 = 8'h28

    Ejemplo negativo que no satura:
      rnd_w = -6 = 32'hFFFFFFFA
      Bits [31:8] = 24'hFFFFFF, bit 7 = 1, extension = 24'hFFFFFF ✓
      → clamp = -6 = 8'hFA

Latencia:
  Los assign (prod_w, rnd_w, clamp_w) son COMBINACIONALES (0 ciclos).
  El resultado se registra en el if(v2):
    Ciclo T+2: v2=1, combinacional calcula clamp_w
    Ciclo T+3: result_int[i] = clamp_w[i] registrado, v3=1

SALIDA: signed [7:0], 8 bits, entero en complemento a 2. Rango [-128, +127].
```

### 3.5 ETAPA 5: Salida

```
Ciclo T+4: done=1, result_int[0:31] contiene 32 valores INT8

La senal done viaja al conv_engine_array, que copia los resultados
al bus de salida out_data_flat[255:0] (32 bytes empaquetados).
```

### 3.6 RESUMEN TEMPORAL DEL POST-PROCESS

```
Ciclo   Senal       Etapa                Bits
──────  ──────────  ───────────────────  ──────────
T       valid=1     Bias add calcula     32+32→32
T+1     v1=1        LeakyReLU calcula    32→32
T+2     v2=1        Requant calcula      32*16→48→32→8
T+3     v3=1        Result registrado    8 bits listo
T+4     done=1      Senal de completado

Pipeline total: 4 ciclos de latencia
Throughput: 32 canales cada 4 ciclos (1 disparo del post-process)
```

---

## 4. PIPELINE DE CONV LINEAR (SIN LeakyReLU)

Se usa en capas: 29, 36.

Identico al pipeline de la Seccion 3, excepto que la Etapa 3 (LeakyReLU) se salta:

```
Etapa 1: Multiplicacion MAC    → igual (INT8*INT8→INT16→INT32 acumulado)
Etapa 2: Suma de Bias          → igual (INT32+INT32→INT32)
Etapa 3: LeakyReLU             → BYPASS: relu = biased (pasa directo)
Etapa 4: Requantizacion        → igual (INT32*INT16→INT48→INT32→INT8)
Etapa 5: Salida                → igual

Cambio en hardware: 1 bit de configuracion skip_relu.
  Si skip_relu=0: relu_r = (biased >= 0) ? biased : (biased >>> 3)
  Si skip_relu=1: relu_r = biased

Esto se activa para LT_CONV1X1_LINEAR (tipo 5).
La activacion LINEAR en darknet significa: output = input (identidad).
No hay funcion de activacion, el valor pasa tal cual al requantizador.
```

---

## 5. OPERACION MAXPOOL 2x2

Se usa en capas: 9, 17, 25.

**Modulo:** `maxpool_unit.sv`

```
Entrada:  4 valores INT8 signed [7:0] (ventana 2x2)
Salida:   1 valor INT8 signed [7:0] (el maximo)

Operacion:
  Dado un pixel de salida en posicion (c, oh, ow):

  a = input[c][oh*2    ][ow*2    ]     posicion arriba-izquierda
  b = input[c][oh*2    ][ow*2 + 1]     posicion arriba-derecha
  c = input[c][oh*2 + 1][ow*2    ]     posicion abajo-izquierda
  d = input[c][oh*2 + 1][ow*2 + 1]     posicion abajo-derecha

  output = max(a, b, c, d)

Implementacion hardware (arbol de comparadores):
  Nivel 1 (combinacional):
    max_ab = (a > b) ? a : b      // comparador signed 8 bits
    max_cd = (c > d) ? c : d      // comparador signed 8 bits

  Nivel 2 (combinacional):
    max_final = (max_ab > max_cd) ? max_ab : max_cd

  Registro (1 ciclo):
    max_out <= max_final

Mapa de bits:
  a, b, c, d:     signed [7:0]     8 bits cada uno
  max_ab, max_cd:  signed [7:0]     8 bits (comparacion, no hay crecimiento)
  max_final:       signed [7:0]     8 bits
  max_out:         signed [7:0]     8 bits registrado

NO HAY multiplicacion.
NO HAY cambio de ancho de bits.
NO HAY requantizacion.
Los datos INT8 entran y salen INT8 sin ninguna transformacion numerica.

Latencia: 1 ciclo (combinacional + registro)

FSM en dpu_top (ciclos por elemento):
  S_POOL_LOAD:    1 ciclo    lee 4 bytes de fmap
  S_POOL_COMPUTE: 1 ciclo    activa pool_valid
  S_POOL_WAIT:    1 ciclo    espera pool_done
  S_POOL_WRITE:   1 ciclo    escribe resultado en fmap
  S_POOL_NEXT:    1 ciclo    avanza al siguiente elemento
  Total: 5 ciclos por pixel de salida

Ejemplo numerico:
  Ventana 2x2:  [42, -15]    a=42, b=-15
                [88,  33]    c=88, d=33

  Nivel 1: max_ab = max(42, -15) = 42
           max_cd = max(88, 33)  = 88
  Nivel 2: max_final = max(42, 88) = 88

  Salida: 88 (INT8)
```

---

## 6. OPERACION ROUTE SPLIT

Se usa en capas: 3, 11, 19.

```
Entrada:  feature map [Cin, H, W] como INT8
Salida:   feature map [Cin/2, H, W] como INT8

Operacion:
  output[c][h][w] = input[c + Cin/2][h][w]

  Toma la SEGUNDA MITAD de los canales.
  (groups=2, group_id=1 en configuracion darknet)

No hay calculo. Solo copia de memoria con offset.

Mapa de bits: INT8 → INT8 (copia directa, 0 transformacion)

FSM en dpu_top:
  S_ROUTE_COPY_A: 1 byte por ciclo

  Direccion fuente:  src = (Cin/2) * H * W + idx
  Direccion destino: dst = idx

  Para idx = 0 hasta (Cout * H * W - 1):
    fmap_out[idx] = fmap_in[Cin/2 * H * W + idx]

Ciclos totales: Cout * H * W (1 byte/ciclo)
```

---

## 7. OPERACION ROUTE CONCAT

Se usa en capas: 6, 8, 14, 16, 22, 24, 34.

```
Entrada A:  feature map [Ca, H, W] como INT8 (source A)
Entrada B:  feature map [Cb, H, W] como INT8 (source B)
Salida:     feature map [Ca+Cb, H, W] como INT8

Operacion:
  output[0     .. Ca-1][h][w] = source_A[0..Ca-1][h][w]    // primera mitad
  output[Ca .. Ca+Cb-1][h][w] = source_B[0..Cb-1][h][w]    // segunda mitad

No hay calculo. Solo copia secuencial de dos bloques de memoria.

Mapa de bits: INT8 → INT8 (copia directa)

Sources pueden ser:
  - fmap actual (ping-pong buffer)
  - fmap_save_l2   (skip connection desde capa 2)
  - fmap_save_l4   (skip connection desde capa 4)
  - fmap_save_l10  (skip connection desde capa 10)
  - fmap_save_l12  (skip connection desde capa 12)
  - fmap_save_l23  (skip connection desde capa 23)  ← NUEVO
  - fmap_save_l27  (skip connection desde capa 27)  ← NUEVO

FSM en dpu_top:
  S_ROUTE_COPY_A: copia source A, 1 byte/ciclo, total = Ca * H * W bytes
  S_ROUTE_COPY_B: copia source B, 1 byte/ciclo, total = Cb * H * W bytes

Ciclos totales: (Ca + Cb) * H * W
```

---

## 8. OPERACION UPSAMPLE 2x (NEAREST NEIGHBOR)

Se usa en capa: 33.

```
Entrada:  feature map [C, Hin, Win] como INT8
Salida:   feature map [C, Hin*2, Win*2] como INT8

Operacion:
  output[c][oh][ow] = input[c][oh >> 1][ow >> 1]

  Donde >> 1 es shift right 1 bit = division entera por 2.

  Cada pixel de entrada se copia a un bloque de 2x2 en la salida.

Ejemplo visual:
  Entrada (1 canal, 2x2):
    A B
    C D

  Salida (1 canal, 4x4):
    A A B B
    A A B B
    C C D D
    C C D D

NO hay calculo. Cero multiplicaciones. Cero sumas.
Solo lectura de memoria con remapeo de direcciones.

Mapa de bits: INT8 → INT8 (copia directa, identico valor)

Calculo de direcciones:
  Para cada posicion de salida (c, oh, ow):
    addr_in  = c * Hin * Win + (oh >> 1) * Win + (ow >> 1)
    addr_out = c * Hout * Wout + oh * Wout + ow

  La division por 2 (>> 1) en hardware es GRATIS:
  simplemente ignora el bit LSB de oh y ow.
  No necesita sumador ni multiplicador.

FSM propuesta:
  Para cada (c, oh, ow) en el espacio de salida:
    1. Calcula addr_in = c * Hin * Win + (oh>>1) * Win + (ow>>1)
    2. Lee fmap_in[addr_in]
    3. Escribe fmap_out[addr_out]
  Total: 2 ciclos por pixel de salida (1 lectura + 1 escritura)
  O si se optimiza: lectura y escritura en el mismo ciclo = 1 ciclo/pixel

Ciclos totales: C * Hout * Wout = C * (2*Hin) * (2*Win) = 4 * C * Hin * Win
  Es 4x mas ciclos que la entrada porque la salida es 4x mas grande.
```

---

## 9. TABLA COMPLETA DE TODAS LAS 37 CAPAS FPGA

### 9.1 Dimensiones con entrada estandar 416x416

| Capa | Tipo            | Cin  | Cout | Hin  | Win  | Hout | Wout | Stride | K   |
|------|-----------------|------|------|------|------|------|------|--------|-----|
| 0    | Conv3x3+Leaky   | 3    | 32   | 416  | 416  | 208  | 208  | 2      | 3x3 |
| 1    | Conv3x3+Leaky   | 32   | 64   | 208  | 208  | 104  | 104  | 2      | 3x3 |
| 2    | Conv3x3+Leaky   | 64   | 64   | 104  | 104  | 104  | 104  | 1      | 3x3 |
| 3    | Route Split     | 64   | 32   | 104  | 104  | 104  | 104  | -      | -   |
| 4    | Conv3x3+Leaky   | 32   | 32   | 104  | 104  | 104  | 104  | 1      | 3x3 |
| 5    | Conv3x3+Leaky   | 32   | 32   | 104  | 104  | 104  | 104  | 1      | 3x3 |
| 6    | Route Concat    | 32+32| 64   | 104  | 104  | 104  | 104  | -      | -   |
| 7    | Conv1x1+Leaky   | 64   | 64   | 104  | 104  | 104  | 104  | 1      | 1x1 |
| 8    | Route Concat    | 64+64| 128  | 104  | 104  | 104  | 104  | -      | -   |
| 9    | MaxPool 2x2     | 128  | 128  | 104  | 104  | 52   | 52   | 2      | 2x2 |
| 10   | Conv3x3+Leaky   | 128  | 128  | 52   | 52   | 52   | 52   | 1      | 3x3 |
| 11   | Route Split     | 128  | 64   | 52   | 52   | 52   | 52   | -      | -   |
| 12   | Conv3x3+Leaky   | 64   | 64   | 52   | 52   | 52   | 52   | 1      | 3x3 |
| 13   | Conv3x3+Leaky   | 64   | 64   | 52   | 52   | 52   | 52   | 1      | 3x3 |
| 14   | Route Concat    | 64+64| 128  | 52   | 52   | 52   | 52   | -      | -   |
| 15   | Conv1x1+Leaky   | 128  | 128  | 52   | 52   | 52   | 52   | 1      | 1x1 |
| 16   | Route Concat    | 128+128| 256| 52   | 52   | 52   | 52   | -      | -   |
| 17   | MaxPool 2x2     | 256  | 256  | 52   | 52   | 26   | 26   | 2      | 2x2 |
| 18   | Conv3x3+Leaky   | 256  | 256  | 26   | 26   | 26   | 26   | 1      | 3x3 |
| 19   | Route Split     | 256  | 128  | 26   | 26   | 26   | 26   | -      | -   |
| 20   | Conv3x3+Leaky   | 128  | 128  | 26   | 26   | 26   | 26   | 1      | 3x3 |
| 21   | Conv3x3+Leaky   | 128  | 128  | 26   | 26   | 26   | 26   | 1      | 3x3 |
| 22   | Route Concat    | 128+128| 256| 26   | 26   | 26   | 26   | -      | -   |
| 23   | Conv1x1+Leaky   | 256  | 256  | 26   | 26   | 26   | 26   | 1      | 1x1 |
| 24   | Route Concat    | 256+256| 512| 26   | 26   | 26   | 26   | -      | -   |
| 25   | MaxPool 2x2     | 512  | 512  | 26   | 26   | 13   | 13   | 2      | 2x2 |
| 26   | Conv3x3+Leaky   | 512  | 512  | 13   | 13   | 13   | 13   | 1      | 3x3 |
| 27   | Conv1x1+Leaky   | 512  | 256  | 13   | 13   | 13   | 13   | 1      | 1x1 |
| 28   | Conv3x3+Leaky   | 256  | 512  | 13   | 13   | 13   | 13   | 1      | 3x3 |
| 29   | Conv1x1 LINEAR  | 512  | 255  | 13   | 13   | 13   | 13   | 1      | 1x1 |
| 31   | Route           | -    | 256  | 13   | 13   | 13   | 13   | -      | -   |
| 32   | Conv1x1+Leaky   | 256  | 128  | 13   | 13   | 13   | 13   | 1      | 1x1 |
| 33   | Upsample 2x     | 128  | 128  | 13   | 13   | 26   | 26   | -      | -   |
| 34   | Route Concat    | 128+256| 384| 26   | 26   | 26   | 26   | -      | -   |
| 35   | Conv3x3+Leaky   | 384  | 256  | 26   | 26   | 26   | 26   | 1      | 3x3 |
| 36   | Conv1x1 LINEAR  | 256  | 255  | 26   | 26   | 26   | 26   | 1      | 1x1 |

**Nota:** Capa 30 y 37 (YOLO decode) se ejecutan en CPU, no en FPGA.

### 9.2 Operaciones MAC por capa (entrada 416x416)

| Capa | Tipo         | Formula MACs                      | Total MACs     |
|------|--------------|-----------------------------------|----------------|
| 0    | Conv3x3 s=2  | 32 * 3 * 9 * 208 * 208           | 37,380,096     |
| 1    | Conv3x3 s=2  | 64 * 32 * 9 * 104 * 104          | 199,229,440    |
| 2    | Conv3x3      | 64 * 64 * 9 * 104 * 104          | 398,458,880    |
| 4    | Conv3x3      | 32 * 32 * 9 * 104 * 104          | 99,614,720     |
| 5    | Conv3x3      | 32 * 32 * 9 * 104 * 104          | 99,614,720     |
| 7    | Conv1x1      | 64 * 64 * 1 * 104 * 104          | 44,236,800     |
| 10   | Conv3x3      | 128 * 128 * 9 * 52 * 52          | 398,458,880    |
| 12   | Conv3x3      | 64 * 64 * 9 * 52 * 52            | 99,614,720     |
| 13   | Conv3x3      | 64 * 64 * 9 * 52 * 52            | 99,614,720     |
| 15   | Conv1x1      | 128 * 128 * 1 * 52 * 52          | 44,236,800     |
| 18   | Conv3x3      | 256 * 256 * 9 * 26 * 26          | 398,458,880    |
| 20   | Conv3x3      | 128 * 128 * 9 * 26 * 26          | 99,614,720     |
| 21   | Conv3x3      | 128 * 128 * 9 * 26 * 26          | 99,614,720     |
| 23   | Conv1x1      | 256 * 256 * 1 * 26 * 26          | 44,236,800     |
| 26   | Conv3x3      | 512 * 512 * 9 * 13 * 13          | 398,458,880    |
| 27   | Conv1x1      | 256 * 512 * 1 * 13 * 13          | 22,118,400     |
| 28   | Conv3x3      | 512 * 256 * 9 * 13 * 13          | 199,229,440    |
| 29   | Conv1x1 LIN  | 255 * 512 * 1 * 13 * 13          | 22,057,200     |
| 32   | Conv1x1      | 128 * 256 * 1 * 13 * 13          | 5,529,600      |
| 35   | Conv3x3      | 256 * 384 * 9 * 26 * 26          | 598,376,448    |
| 36   | Conv1x1 LIN  | 255 * 256 * 1 * 26 * 26          | 44,124,480     |
|      |              |                                   |                |
|      | **TOTAL**    |                                   | **2,896,729,344** |

**~2.9 GOPS (giga-operaciones) por imagen.**

### 9.3 Save buffers necesarios (skip connections)

| Buffer       | Grabado en capa | Leido en capa | Tamano (bytes)           |
|--------------|-----------------|---------------|--------------------------|
| save_l2      | 2               | 8             | 64 * 104 * 104 = 692,224 |
| save_l4      | 4               | 6             | 32 * 104 * 104 = 346,112 |
| save_l10     | 10              | 16            | 128 * 52 * 52 = 346,112  |
| save_l12     | 12              | 14            | 64 * 52 * 52 = 173,056   |
| save_l18     | 18              | 24            | 256 * 26 * 26 = 173,056  |
| save_l20     | 20              | 22            | 128 * 26 * 26 = 86,528   |
| save_l23     | 23              | 34            | 256 * 26 * 26 = 173,056  |
| save_l27     | 27              | 31            | 256 * 13 * 13 = 43,264   |

**Total save buffers: ~2.03 MB**

### 9.4 Weight buffer tamaño por capa (la mas grande define MAX_WBUF)

| Capa | Cout * Cin * K^2      | Bytes      |
|------|------------------------|------------|
| 26   | 512 * 512 * 9          | 2,359,296  |
| 28   | 512 * 256 * 9          | 1,179,648  |
| 35   | 256 * 384 * 9          | 884,736    |
| 18   | 256 * 256 * 9          | 589,824    |

**MAX_WBUF debe ser al menos 2,359,296** (para capa 26: 512x512x3x3).

---

## 10. DIAGRAMA MAESTRO DE ANCHOS DE BITS

```
                     8b         8b
                   +-----+   +-----+
                   |Peso |   | Act |
                   |INT8 |   |INT8 |
                   +--+--+   +--+--+
                      |         |
              +-------+---------+-------+
              |  MULT 8x8 -> 16         |  Shift-and-add (sin DSP)
              |  signed [15:0]          |  8 partial products + adder tree
              +------------+------------+
                           | 16b
              +------------+------------+
              |  SEXT 16 -> 32          |  {{16{bit15}}, prod16}
              |  signed [31:0]          |  Solo cableado, 0 LUTs
              +------------+------------+
                           | 32b
              +------------+------------+
              |  ACUMULADOR += (x32)    |  32 sumas en paralelo (1 por fila)
              |  signed [31:0]          |  Se repite N ciclos:
              |  N = ceil(Cin/32) * K^2 |  N=1 (L0) a N=144 (L26)
              |  REGISTRADO: 1 ciclo    |
              +------------+------------+
                           | 32b
              +------------+------------+
              |  BIAS ADD               |  32b + 32b = 32b
              |  signed [31:0]          |  REGISTRADO: 1 ciclo
              +------------+------------+
                           | 32b
              +------------+------------+
              |  LeakyReLU              |  x>=0 ? x : x>>>3
              |  signed [31:0]          |  O BYPASS si LINEAR
              |  REGISTRADO: 1 ciclo    |
              +------------+------------+
                           | 32b          16b
                           |            +-----+
                           |            |Scale|
                           |            |[15:0]|
              +------------+------+     +--+--+
              |  MULT 32x16 -> 48 +--------+
              |  signed [47:0]    |  COMBINACIONAL
              +------------+------+
                           | 48b
              +------------+------------+
              |  SHIFT >>> 16           |  Descarta 16 bits LSB
              |  signed [31:0]          |  COMBINACIONAL
              +------------+------------+
                           | 32b
              +------------+------------+
              |  CLAMP [-128, +127]     |  COMBINACIONAL
              |  signed [7:0]           |
              |  REGISTRADO: 1 ciclo    |
              +------------+------------+
                           | 8b
                     +-----+-----+
                     |  SALIDA   |
                     |   INT8    |
                     +-----------+

LATENCIA TOTAL POST-PROCESS: 4 ciclos de reloj
LATENCIA TOTAL POR PIXEL:    variable (depende de carga de pesos y activaciones)
```

---

## 11. FLUJO DE DATOS ENTRE CAPAS

```
              +-------------+
              | Imagen RGB  |  3 canales, 416x416, INT8
              | [-128, 127] |
              +------+------+
                     |
    +================+=================+
    | BLOQUE CSP 1 (capas 0-9)        |
    |                                  |
    | L0: Conv3x3 s=2 → 32x208x208   |
    | L1: Conv3x3 s=2 → 64x104x104   |
    | L2: Conv3x3     → 64x104x104   | ←── SAVE para L8
    | L3: Split        → 32x104x104   |
    | L4: Conv3x3     → 32x104x104   | ←── SAVE para L6
    | L5: Conv3x3     → 32x104x104   |
    | L6: Concat(L5,L4)→ 64x104x104  |
    | L7: Conv1x1     → 64x104x104   |
    | L8: Concat(L2,L7)→128x104x104  |
    | L9: MaxPool      →128x52x52    |
    +================+=================+
                     |
    +================+=================+
    | BLOQUE CSP 2 (capas 10-17)       |
    | (misma estructura, 2x canales)   |
    |                                  |
    | L10: Conv3x3    →128x52x52     | ←── SAVE para L16
    | L11: Split       → 64x52x52    |
    | L12: Conv3x3    → 64x52x52     | ←── SAVE para L14
    | L13: Conv3x3    → 64x52x52     |
    | L14: Concat(L13,L12)→128x52x52 |
    | L15: Conv1x1    →128x52x52     |
    | L16: Concat(L10,L15)→256x52x52 |
    | L17: MaxPool     →256x26x26    |
    +================+=================+
                     |
    +================+=================+
    | BLOQUE CSP 3 (capas 18-25)       |  ←── NUEVO
    | (misma estructura, 2x canales)   |
    |                                  |
    | L18: Conv3x3    →256x26x26     | ←── SAVE para L24
    | L19: Split       →128x26x26    |
    | L20: Conv3x3    →128x26x26     | ←── SAVE para L22
    | L21: Conv3x3    →128x26x26     |
    | L22: Concat(L21,L20)→256x26x26 |
    | L23: Conv1x1    →256x26x26     | ←── SAVE para L34
    | L24: Concat(L18,L23)→512x26x26 |
    | L25: MaxPool     →512x13x13    |
    +================+=================+
                     |
    +================+=================+
    | CABEZA DETECCION 1 (capas 26-29) |  ←── NUEVO
    |                                  |
    | L26: Conv3x3    →512x13x13     |
    | L27: Conv1x1    →256x13x13     | ←── SAVE para L31
    | L28: Conv3x3    →512x13x13     |
    | L29: Conv1x1 LIN→255x13x13    | ──── SALIDA 1 → CPU
    +================+=================+

    +================+=================+
    | PUENTE + CABEZA 2 (capas 31-36)  |  ←── NUEVO
    |                                  |
    | L31: Route(L27)  →256x13x13    | (lee save_l27)
    | L32: Conv1x1     →128x13x13    |
    | L33: Upsample 2x →128x26x26   | ←── OPERACION NUEVA
    | L34: Concat(L33,L23)→384x26x26 | (lee save_l23)
    | L35: Conv3x3     →256x26x26    |
    | L36: Conv1x1 LIN →255x26x26   | ──── SALIDA 2 → CPU
    +================+=================+

    CPU recibe:
      Tensor 1: [255, 13, 13] INT8  (objetos grandes, stride 32)
      Tensor 2: [255, 26, 26] INT8  (objetos pequenos, stride 16)
      → sigmoid, exp, anchor decode, NMS → bounding boxes finales
```

---

## 12. FORMATO DE LOS 255 CANALES DE SALIDA

Las capas 29 y 36 producen 255 canales. Estos codifican:

```
255 = 3 anchors * (4 coordenadas + 1 objectness + 80 clases)
    = 3 * 85
    = 255

Para cada celda del grid y cada anchor:
  Canal  0: tx  (offset x del centro)       ← requiere sigmoid en CPU
  Canal  1: ty  (offset y del centro)       ← requiere sigmoid en CPU
  Canal  2: tw  (ancho relativo al anchor)  ← requiere exp en CPU
  Canal  3: th  (alto relativo al anchor)   ← requiere exp en CPU
  Canal  4: objectness (confianza)          ← requiere sigmoid en CPU
  Canal  5-84: probabilidad de cada clase   ← requiere sigmoid en CPU

Repetido para 3 anchors por celda:
  Anchor 0: canales 0-84
  Anchor 1: canales 85-169
  Anchor 2: canales 170-254

Estos valores salen como INT8 [-128, 127] del DPU.
El CPU debe:
  1. Convertir INT8 a float: val_float = val_int8 / scale_factor
  2. Aplicar sigmoid: sigma(x) = 1 / (1 + exp(-x))
  3. Aplicar exp para tw, th
  4. Multiplicar por tamano del anchor
  5. Filtrar por threshold de objectness
  6. Non-Maximum Suppression
```

---

## 13. RESUMEN DE TODOS LOS ANCHOS DE BITS EN EL DPU

| Senal              | Ancho  | Tipo    | Donde aparece                        |
|--------------------|--------|---------|--------------------------------------|
| Activacion entrada | 8 bit  | signed  | fmap_a/fmap_b, patch_buf             |
| Peso               | 8 bit  | signed  | weight_buf, w_reg                    |
| Producto MAC       | 16 bit | signed  | prod16 en mac_array                  |
| Producto extendido | 32 bit | signed  | product_sext en mac_array            |
| Parcial (32 sumas) | 32 bit | signed  | partial en mac_array                 |
| Acumulador         | 32 bit | signed  | acc en mac_array                     |
| Bias               | 32 bit | signed  | bias_buf, bias_reg                   |
| Biased (acc+bias)  | 32 bit | signed  | biased_r en post_process             |
| LeakyReLU out      | 32 bit | signed  | relu_r en post_process               |
| Scale              | 16 bit | unsigned| scale_reg, ld_scale                  |
| Producto requant   | 48 bit | signed  | prod_w en post_process               |
| Shifted            | 32 bit | signed  | rnd_w en post_process                |
| Clamped            | 8 bit  | signed  | clamp_w / result_int en post_process |
| Salida capa        | 8 bit  | signed  | out_data_int, fmap_a/fmap_b          |
| MaxPool entrada    | 8 bit  | signed  | pool_a/b/c/d                         |
| MaxPool salida     | 8 bit  | signed  | pool_max                             |
| Upsample entrada   | 8 bit  | signed  | lectura de fmap                      |
| Upsample salida    | 8 bit  | signed  | escritura a fmap                     |
| Route entrada      | 8 bit  | signed  | lectura de fmap / save_buf           |
| Route salida       | 8 bit  | signed  | escritura a fmap                     |
| Direccion fmap     | 24 bit | unsigned| ADDR_BITS                            |
| Direccion pesos    | 18 bit | unsigned| weight_rd_addr (21 para nuevo)       |
| Contador capa      | 6 bit  | unsigned| current_layer_reg (0-36)             |
| Cin, Cout, H, W    | 11 bit | unsigned| ld_c_in, ld_c_out, etc.              |
| Bus wide memoria   | 256 bit| packed  | weight_rd_data_wide, patch_rd_data_wide |
