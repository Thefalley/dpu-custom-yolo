# DPU Control Software Architecture

**Role:** Senior embedded software architect.  
**Target:** Custom INT8 AI DPU for YOLO; control software for ZedBoard (Zynq) + Vitis bare-metal.  
**Current:** Design and validation only; final code in a later phase. PC (gcc) + simulation for now.

This document defines and validates the **software architecture** that will control the DPU via memory-mapped registers and DMA. It is aligned with the existing RTL command interface (`dpu_layer0_top`: write_byte, run_layer0, read_byte).

**Naming:** Software design phases are labelled **Phase W1** … **Phase W6** (W = software / control wave) to distinguish them from hardware or project phases (e.g. Phase 1–7 in CONTEXTO_PROYECTO.md).

---

# Phase W1 — DPU COMMAND & CONTROL ANALYSIS

## Goal

Identify the minimal set of DPU commands and controls needed to operate the accelerator correctly.

---

## 1. Minimal Set of DPU Commands

| Command       | Mnemonic / Role   | Blocking? | Justification |
|---------------|-------------------|-----------|----------------|
| **RESET**     | Reset DPU state   | Blocking  | Required before any run; clears FSM and optional internal state. |
| **CONFIGURE** | Set shape/addresses | Non-blocking | Set input/output/weight base addresses, dimensions; no execution. |
| **LOAD_INPUT** | Load input image  | Non-blocking (DMA) or Blocking (PIO) | Fill input buffer; can be byte-write loop (current RTL) or future DMA. |
| **LOAD_WEIGHTS** | Load weights + bias | Non-blocking (DMA) or Blocking (PIO) | Fill weight and bias regions; same as above. |
| **START**     | Start inference   | Non-blocking | Fire the run; software returns; completion via STATUS/IRQ. |
| **STOP**      | Abort run         | Non-blocking | Optional; halt FSM cleanly for debug or timeout. |
| **STATUS**    | Poll busy/done/error | Non-blocking | Read busy, done, error flags; used for polling or to confirm IRQ. |

**Optional (for flexibility):**

- **READ_OUTPUT** — Not a separate “command” in the current RTL: reading is done via **read_byte** (address in output region). So “read output” is a **usage pattern** of the same memory-mapped read interface, not a distinct opcode. We keep it as a **logical operation** in the software API (e.g. `dpu_read_output()`).

---

## 2. Per-Command Explanation

### RESET

- **What it does:** Asserts hardware reset (or a soft-reset control bit) so the DPU FSM and internal state (e.g. command decoder, engine) return to a known idle state.
- **When used:** Once at init; optionally before each inference if the driver guarantees a clean state.
- **Hardware state:** FSM → IDLE; command interface ready; internal buffers/pointers in defined state (implementation-dependent).

**Blocking:** Yes. Software waits for reset release and optionally for a “reset done” or “idle” status before proceeding.

---

### CONFIGURE

- **What it does:** Writes registers that define (a) input base address, (b) output base address, (c) weight/bias base (or use fixed map), (d) dimensions (e.g. H_OUT, W_OUT, NUM_CH) if the hardware supports variable shape.
- **When used:** Once per inference (or once at init if shape never changes). Current RTL has fixed shape; “configure” still sets base addresses for DMA or for a wrapper that uses the fixed map.
- **Hardware state:** Only register values; no execution started.

**Blocking:** No. Pure register writes.

---

### LOAD_INPUT

- **What it does:** Fills the DPU input buffer with the preprocessed image (e.g. INT8 418×418×3 padded, or 416×416×3 then HW pads). Today: software writes bytes via write_byte (PIO). Later: program DMA to write to the same logical region.
- **When used:** Once per inference, after CONFIGURE (if addresses are used) and before START.
- **Hardware state:** Input memory / buffer contents; no FSM run.

**Blocking:** With PIO (current RTL): yes, until last byte is written. With DMA: non-blocking (fire DMA, wait for DMA complete elsewhere).

---

### LOAD_WEIGHTS

- **What it does:** Fills weight and bias regions (e.g. 864 bytes weights + 128 bytes bias for layer0). Same as LOAD_INPUT: PIO write_byte today, DMA later.
- **When used:** Once per model load (or per layer if HW is layer-by-layer); typically before first START or after RESET when changing model.
- **Hardware state:** Weight and bias memory; no FSM run.

**Blocking:** Same as LOAD_INPUT (PIO vs DMA).

---

### START

- **What it does:** Starts the inference run (e.g. run_layer0: full 208×208×32). In RTL, this is “cmd_type = run_layer0”; FSM runs until done.
- **When used:** After LOAD_INPUT and LOAD_WEIGHTS (and CONFIGURE if used); once per inference.
- **Hardware state:** FSM leaves IDLE, runs computation; **busy** = 1, then **done** = 1 when finished.

**Blocking:** No. Software writes START and returns; completion is observed via STATUS (polling) or IRQ.

---

### STOP

- **What it does:** Requests abort of the current run (FSM returns to IDLE or a safe state). Optional if RTL does not support mid-run abort.
- **When used:** On timeout, user cancel, or error recovery.
- **Hardware state:** FSM aborted; **busy** = 0; **done** may be 0 (aborted).

**Blocking:** No. Software writes STOP and can then poll STATUS until not busy.

---

### STATUS

- **What it does:** Returns current state: **busy**, **done**, and optionally **error** (if RTL exposes it). Used for polling or to confirm interrupt.
- **When used:** After START to wait for completion; after STOP to confirm idle; at init to confirm reset.
- **Hardware state:** None (read-only).

**Blocking:** No. Single register read.

---

## 3. Blocking vs Non-Blocking Summary

| Command       | Blocking? | Reason |
|---------------|-----------|--------|
| RESET         | Yes       | Must wait until HW is idle. |
| CONFIGURE     | No        | Register writes only. |
| LOAD_INPUT    | Yes (PIO) / No (DMA) | PIO: wait last write. DMA: async. |
| LOAD_WEIGHTS  | Yes (PIO) / No (DMA) | Same as LOAD_INPUT. |
| START         | No        | Async execution; completion via STATUS/IRQ. |
| STOP          | No        | Request only; then poll STATUS. |
| STATUS        | No        | Single read. |

---

## Deliverable (Phase W1)

- **Commands:** RESET, CONFIGURE, LOAD_INPUT, LOAD_WEIGHTS, START, STOP, STATUS (and logical READ_OUTPUT via address range).
- **Explanation:** Per-command “what / when / which state” and blocking behaviour as above.
- **Justification:** Minimal set to load data, start run, observe completion, and optionally abort; aligns with current RTL (write_byte, run_layer0, read_byte) and future DMA + memory-mapped registers.

---

# Phase W2 — SOFTWARE HIERARCHY DESIGN

## Goal

Design the full software stack that will control the DPU and map cleanly to PC (gcc) now and Vitis bare-metal later.

---

## 1. Software Layers

```
+------------------------------------------+
|  APPLICATION LAYER                       |
|  (YOLO inference, image path, results)   |
+------------------------------------------+
                    |
                    v
+------------------------------------------+
|  DPU DRIVER / LIBRARY                    |
|  (dpu_init, dpu_run_inference, etc.)     |
+------------------------------------------+
                    |
                    v
+------------------------------------------+
|  HARDWARE ABSTRACTION LAYER (HAL)        |
|  (register read/write, DMA if present)   |
+------------------------------------------+
                    |
                    v
+------------------------------------------+
|  PLATFORM (PC memory map / Zynq AXI)     |
+------------------------------------------+
```

---

## 2. Responsibility of Each Layer

### Application layer

- **Responsibility:** High-level flow: load image path or buffer, call “run one inference”, get back detections or layer outputs. No knowledge of registers or DMA.
- **Depends on:** DPU driver API only (e.g. `dpu_run_inference(in_buf, out_buf)` or file-based wrapper).

### DPU driver / library

- **Responsibility:** Implements the command sequence: RESET (if needed), CONFIGURE, LOAD_INPUT, LOAD_WEIGHTS, START, wait for completion via STATUS (or IRQ), READ_OUTPUT into host buffer. Exposes a small API: init, load_weights (optional), run_inference, read_output, deinit.
- **Depends on:** HAL only (register write/read, optional DMA submit/wait). No platform-specific addresses; those come from HAL or a config table.

### Hardware abstraction layer (HAL)

- **Responsibility:** Abstract register and memory access. For PC: reads/writes go to a simulated “DPU” backend (e.g. a model that mimics register/memory behaviour, or a test harness that talks to RTL sim). For Vitis: same function names, but implementation uses volatile pointers to the real AXI base address (and optionally DMA driver). Optionally abstracts DMA (submit transfer, wait for done).
- **Depends on:** Platform (PC vs Zynq) and build configuration; provides a uniform interface to the driver.

### Platform

- **PC:** “Memory map” is a struct or array in process memory used by the DPU model/sim; no real AXI.  
- **Zynq/Vitis:** Memory map is the actual AXI-Lite (and AXI-DMA if used) base address; HAL translates register/memory accesses to pointer dereferences or DMA calls.

---

## 3. Mapping to PC vs Vitis

| Aspect            | PC (gcc, simulation)              | Vitis bare-metal                    |
|------------------|------------------------------------|-------------------------------------|
| HAL impl         | Mock or sim backend                | Volatile pointers to AXI base       |
| Register access  | Write to sim buffer / IPC / file   | `*(volatile uint32_t*)(base+off)`   |
| “Memory” (input/weights/output) | Host arrays; “load” = copy into sim or send to sim | DMA to/from DPU buffers or shared memory |
| DPU driver       | **Same code**                     | **Same code**                       |
| Application      | **Same code** (e.g. one inference) | **Same code**                       |

The hierarchy is chosen so that **only the HAL implementation** changes between PC and Vitis; the driver and application stay unchanged.

---

## Deliverable (Phase W2)

- **Architecture:** Application → DPU driver → HAL → Platform, as above.
- **Responsibilities:** Application = inference flow; Driver = command sequence and API; HAL = register/memory (and optional DMA) abstraction; Platform = real or simulated HW.
- **Mapping:** Same driver and app on PC and Vitis; only HAL and platform differ.

---

# Phase W3 — MEMORY & IMAGE LOADING FLOW

## Goal

Define image format, memory layout, and the step-by-step flow from “image ready” to “DPU can run,” and how this maps to DMA later.

---

## 1. Image Format and Memory Layout

### Image format (input to DPU)

- **Resolution:** 416×416 (input to network); after padding (e.g. 1 pixel): 418×418.
- **Channels:** 3 (e.g. BGR or RGB; model-specific).
- **Data type:** INT8, range typically [-128, 127].
- **Layout:** **CHW** (channel-major) or **HWC** (channel-last) as agreed with RTL; current RTL uses a single linear buffer indexed as `[c][h][w]` (e.g. 3×418×418 = 523,254 bytes for layer0 input).

### Memory layout (inside DPU or “DPU view”)

- **Linear, byte-addressed:**  
  - Region 0: input (e.g. 0 .. 523253).  
  - Region 1: weights (e.g. 523254 .. 524117).  
  - Region 2: bias (e.g. 524118 .. 524245).  
  - Region 3: output (e.g. 524246 .. 1907157).  
- **No tiling** in the current design; tiling can be added later if the hardware supports it and the HAL/driver expose it.

---

## 2. Step-by-Step Image Loading Flow

1. **Generate or load image**
   - **Synthetic:** App or test generates INT8 array 416×416×3 (or 3×418×418 padded) in host memory.
   - **Real image:** Load image, resize to 416×416, convert to INT8 (e.g. subtract 128, clamp), store in host buffer (same shape).

2. **Optional: pad for layer0**
   - If padding is done in software: produce 3×418×418 buffer. If done in HW, feed 416×416×3 and configure padding in CONFIGURE (when supported).

3. **Write into “DPU input” memory**
   - **PC/sim:** HAL “write” copies host buffer into the simulated DPU input region (e.g. shared memory, file, or model process). This can be a loop of write_byte(addr, data) or a bulk write to the sim.
   - **Vitis/DMA:** HAL programs DMA to transfer host buffer to the physical base address of the DPU input region; no per-byte PIO.

4. **Inform DPU of location**
   - **Fixed map (current RTL):** No address register; input is always at base 0 in the DPU’s view. So “inform” = “data is already in the right place after step 3.”
   - **Configurable (future):** CONFIGURE writes “input_base” (and optionally size); DPU uses that for the run. Same flow: after DMA (or PIO) to that base, CONFIGURE + START.

5. **Weights and bias**
   - Same idea: host holds weight and bias arrays; HAL writes them into DPU weight/bias regions (PIO today, DMA later). No “image” here, but the flow is identical (prepare buffer → write to DPU region).

---

## 3. Mapping to DMA

- **Today (PIO):** “Write into DPU memory” = driver calls HAL write_byte (or write_block) in a loop; HAL talks to sim or to a future AXI-Lite “data port” that fills internal DPU RAM.
- **Later (DMA):** “Write into DPU memory” = driver calls HAL `dma_send(host_src, dpu_dest, size)`. HAL programs the DMA engine (source = host buffer, destination = AXI address of DPU input/weight/bias region), then either blocks until DMA done or returns and completion is checked via STATUS/IRQ. The rest of the command sequence (START, STATUS, READ_OUTPUT) is unchanged; only the data path changes from PIO to DMA.

---

## Deliverable (Phase W3)

- **Image format:** INT8, 416×416×3 (or 3×418×418 padded); layout CHW linear.
- **Memory layout:** Linear byte regions for input, weights, bias, output (address map as in RTL or derived from it).
- **Flow:** Generate/load image → (optional pad) → write to DPU input (PIO/DMA) → (same for weights/bias) → DPU already “informed” by fixed map or CONFIGURE.
- **DMA:** Same logical “load” step; HAL swaps PIO loop for DMA transfer; driver API unchanged.

---

# Phase W4 — REGISTER MAP & DATA INJECTION

## Goal

Design the register-level interface used to control the DPU (memory-mapped view that the HAL will use).

---

## 1. Register Map (Target for Vitis; Simulated on PC)

Base address `DPU_BASE` (e.g. 0x43C0_0000 on Zynq). All registers 32-bit; offsets in bytes.

| Offset | Name           | R/W | Bit fields        | Meaning |
|--------|----------------|-----|-------------------|---------|
| 0x00   | CONTROL        | W   | [1:0] = CMD_TYPE  | 0 = NOP, 1 = WRITE_BYTE, 2 = RUN_LAYER0, 3 = READ_BYTE (or RESET if bit 2 = soft_reset). |
|        |                |     | [2] = SOFT_RESET  | 1 = assert soft reset. |
|        |                |     | [3] = START      | 1 = start run (run_layer0). |
| 0x04   | STATUS         | R   | [0] = CMD_READY  | 1 = last command accepted / response ready. |
|        |                |     | [1] = BUSY       | 1 = DPU running. |
|        |                |     | [2] = DONE       | 1 = run finished. |
|        |                |     | [3] = RSP_VALID  | 1 = read data valid (for READ_BYTE). |
| 0x08   | ADDR           | W   | [23:0] ADDR      | Byte address for WRITE_BYTE or READ_BYTE. |
| 0x0C   | DATA_WR        | W   | [7:0] DATA       | Byte to write (for WRITE_BYTE). |
| 0x10   | DATA_RD        | R   | [7:0] DATA       | Byte read (for READ_BYTE); valid when RSP_VALID. |

**Alternative (closer to current RTL):** One “command” register that encodes valid + type + addr + data in one or two cycles; STATUS holds cmd_ready, busy, done, rsp_data. The table above is a logical view that the HAL can present even if the actual RTL packs signals differently.

**Optional later:** INPUT_BASE, OUTPUT_BASE, WEIGHT_BASE, SIZE_H, SIZE_W (for configurable shapes); DMA source/dest and length (if DMA is in the DPU block).

---

## 2. Address Map (DPU Logical Memory — Byte)

Used in ADDR when CMD_TYPE = WRITE_BYTE or READ_BYTE.

| Region    | Byte range (example) | Size (layer0) |
|-----------|------------------------|---------------|
| INPUT     | 0 .. PAD_SIZE-1        | 523254        |
| WEIGHTS   | PAD_SIZE .. PAD_SIZE+863 | 864          |
| BIAS      | PAD_SIZE+864 .. PAD_SIZE+864+127 | 128   |
| OUTPUT    | PAD_SIZE+864+128 .. ... | 1382912      |

(Exact constants match `dpu_layer0_top.sv`: INPUT_BASE=0, WEIGHT_BASE=PAD_SIZE, etc.)

---

## 3. Register Usage Sequence

- **WRITE_BYTE (load input/weights/bias):** Write ADDR, write DATA_WR, write CONTROL with CMD_TYPE=WRITE_BYTE (and valid/trigger). Poll STATUS until CMD_READY; repeat for next byte (or batch if RTL supports burst).
- **RUN:** Write CONTROL with START=1 (or CMD_TYPE=RUN_LAYER0). Return; poll STATUS until DONE=1 (and BUSY=0).
- **READ_BYTE (read output):** Write ADDR (output region), write CONTROL with CMD_TYPE=READ_BYTE. Poll STATUS until RSP_VALID/CMD_READY; read DATA_RD. Repeat for each byte (or burst).
- **RESET:** Write CONTROL with SOFT_RESET=1; then 0; wait until STATUS indicates idle.

---

## Deliverable (Phase W4)

- **Register map:** CONTROL, STATUS, ADDR, DATA_WR, DATA_RD with offsets and bit fields as above.
- **Address map:** Input, weights, bias, output regions and sizes.
- **Usage:** Write ADDR/DATA_WR + command for loads; write START for run; poll STATUS; write ADDR + READ for readback.

---

# Phase W5 — FULL EXECUTION SEQUENCE

## Goal

Describe the complete software-driven execution of one YOLO inference on the DPU (one layer0 run as the current HW unit), in plain English and pseudo-code, with error handling.

---

## 1. Full Execution Flow (Plain English)

1. **System init:** HAL initializes (e.g. map DPU_BASE for Zynq; or open sim connection on PC). Driver calls RESET (optional if HW comes out of reset idle); wait until not busy.
2. **Image load:** Application provides input buffer (INT8, 3×418×418). Driver calls HAL to write this buffer to DPU input region (byte-by-byte today, or DMA later). Same for weights and bias if not already loaded.
3. **DPU configuration:** If using CONFIGURE, driver writes input/output/weight base and dimensions; for fixed map, skip or no-op.
4. **Start run:** Driver writes START (RUN_LAYER0). Returns immediately.
5. **Completion:** Driver polls STATUS until DONE=1 (and BUSY=0). Timeout after N ms; if timeout, optionally call STOP and return error.
6. **Result readback:** Driver writes output base address and issues READ_BYTE for each byte of the output region (or burst); HAL returns bytes into host output buffer. Application uses this buffer (e.g. for next layer or for detection post-processing).
7. **Errors:** Invalid ADDR, timeout, or RTL error flag (if any) → driver returns error code; application can retry or abort.

---

## 2. Pseudo-Code

```text
// Application
output_buf = dpu_run_inference(input_buf, weights_buf, bias_buf);
if (output_buf == NULL)  // error
    handle_error();

// Driver: dpu_run_inference
dpu_run_inference(in_buf, w_buf, b_buf, out_buf):
    hal_reset_dpu()                    // optional
    hal_load_region(INPUT_BASE,  in_buf,  INPUT_SIZE)
    hal_load_region(WEIGHT_BASE, w_buf,   WEIGHT_SIZE)
    hal_load_region(BIAS_BASE,   b_buf,   BIAS_SIZE)
    hal_start()
    if not hal_wait_done(timeout_ms):
        hal_stop()
        return ERROR_TIMEOUT
    hal_read_region(OUTPUT_BASE, out_buf, OUTPUT_SIZE)
    return OK

// HAL: hal_load_region (PIO version)
hal_load_region(base, host_buf, size):
    for i = 0 to size-1:
        hal_write_reg(ADDR, base + i)
        hal_write_reg(DATA_WR, host_buf[i])
        hal_write_reg(CONTROL, CMD_WRITE_BYTE)
        while not (hal_read_reg(STATUS) & CMD_READY): spin

// HAL: hal_start
hal_start():
    hal_write_reg(CONTROL, CMD_RUN_LAYER0)  // or START bit

// HAL: hal_wait_done
hal_wait_done(timeout_ms):
    t0 = now()
    while (hal_read_reg(STATUS) & BUSY):
        if (now() - t0 > timeout_ms) return false
    return (hal_read_reg(STATUS) & DONE)

// HAL: hal_read_region (PIO)
hal_read_region(base, host_buf, size):
    for i = 0 to size-1:
        hal_write_reg(ADDR, base + i)
        hal_write_reg(CONTROL, CMD_READ_BYTE)
        while not (hal_read_reg(STATUS) & RSP_VALID): spin
        host_buf[i] = hal_read_reg(DATA_RD)
```

---

## 3. Where Errors Can Occur and How Software Detects Them

| Point              | Failure mode           | Detection / handling                          |
|--------------------|------------------------|-----------------------------------------------|
| Init               | HAL cannot map HW      | HAL init returns error; driver exits or retries. |
| Load (PIO/DMA)     | Wrong size, bad addr   | HAL checks bounds; RTL may ignore bad addr → garbage run. |
| Start              | DPU already busy       | Driver can check STATUS before START; if busy, return BUSY. |
| Wait done          | FSM stuck, HW hang     | Timeout in hal_wait_done; then STOP and return TIMEOUT. |
| Readback           | RSP_VALID never        | Timeout per byte or per transfer; return IO_ERROR. |
| RTL error flag     | Internal error         | If STATUS has an error bit, driver returns DPU_ERROR. |

---

## Deliverable (Phase W5)

- **Flow:** Init → load input/weights/bias → start → wait done → read output; RESET/STOP where needed.
- **Pseudo-code:** Driver and HAL as above; application calls single “run inference.”
- **Error strategy:** Timeout on wait_done and on read; STATUS checks; HAL/driver return codes; application handles errors.

---

# Phase W6 — VALIDATION STRATEGY

## Goal

Ensure the software design is verifiable before real hardware exists and that it reuses unchanged on Vitis.

---

## 1. Testing on PC

- **HAL mock:** Implement HAL so that “register” writes/reads go to an in-process model (e.g. a C struct that mimics the DPU state machine and memory). The model can:
  - For WRITE_BYTE: store bytes in the right region.
  - For RUN: either compute a dummy result or run a small reference (e.g. one row) and fill “output” so that the driver sees a deterministic result.
- **Simulated DPU:** Replace the mock with a process or thread that talks to the RTL simulator (e.g. DPI, socket, or file-based protocol). Same HAL interface; backend sends commands to the sim and returns responses. Driver and app code unchanged.
- **Success at this stage:** The same driver and application code run on PC; with the mock, they complete without crash and produce a defined result; with the sim backend, they produce the same result as the RTL testbench (e.g. layer0 output matches reference).

---

## 2. What “Success” Means Now

- **Design phase:**  
  - Command set and blocking behaviour are defined and justified.  
  - Layers (App / Driver / HAL / Platform) are defined and responsibilities clear.  
  - Memory and image flow are specified; register map and execution sequence are specified.  
  - Validation plan is in place: PC with mock/sim, same code path as future Vitis.
- **Later (implementation):**  
  - Driver + app run on PC against mock/sim and pass one inference (e.g. layer0).  
  - Same binary (or same source with Vitis HAL) runs on ZedBoard and completes one inference with real DPU.

---

## 3. Reuse in Vitis Unchanged

- **Same APIs:** `dpu_run_inference`, `hal_load_region`, `hal_start`, `hal_wait_done`, `hal_read_region` keep the same signatures and semantics.
- **Only HAL implementation changes:** On Vitis, HAL uses `DPU_BASE` and volatile accesses (and optionally DMA APIs) instead of mock/sim. Compile the same driver and app with the Vitis HAL and link with the Vitis runtime; no changes to the command sequence or to the application.
- **Optional:** A small platform header (e.g. `DPU_BASE`, `USE_DMA`) is the only configuration switch between “PC sim” and “Vitis bare-metal.”

---

## Deliverable (Phase W6)

- **Validation plan:** PC test with HAL mock and with simulated DPU; success = same flow completes and (with sim) matches RTL reference.
- **Success criteria:** Design complete and validated in simulation; code reuse on Vitis with only HAL and platform config changed.
- **Mapping to Vitis:** Same driver and application; swap HAL and base address for bare-metal.

---

# Summary of the Entire Software Design

- **Commands:** RESET, CONFIGURE, LOAD_INPUT, LOAD_WEIGHTS, START, STOP, STATUS; read output via address range. Blocking where noted (RESET, PIO loads); START and STOP non-blocking; completion via STATUS/IRQ.
- **Stack:** Application → DPU driver → HAL → Platform; only HAL and platform differ between PC and Vitis.
- **Memory:** Linear INT8 input (e.g. 3×418×418), weights, bias, output; fixed address map aligned with current RTL.
- **Registers:** CONTROL, STATUS, ADDR, DATA_WR, DATA_RD; usage sequence for load / run / readback.
- **Execution:** Init → load regions → start → wait done (with timeout) → read output; pseudo-code and error handling defined.
- **Validation:** PC with mock or sim backend; same code path; reuse in Vitis by swapping HAL and platform only.

This gives a single, consistent software architecture that can be implemented and validated on PC and then moved to ZedBoard + Vitis bare-metal without redesigning the control flow.
