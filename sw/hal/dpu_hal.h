/**
 * DPU Hardware Abstraction Layer
 * Low-level register access for DPU Custom YOLOv4-tiny accelerator.
 *
 * Supports two backends:
 *   - PLATFORM_ZYNQ: Memory-mapped I/O on Zynq (Vitis bare-metal)
 *   - PLATFORM_SIM:  PC simulation via file I/O (for golden model validation)
 */

#ifndef DPU_HAL_H
#define DPU_HAL_H

#include <stdint.h>
#include <stddef.h>

/* --------------------------------------------------------------------------
 * Register offsets (match dpu_system_top AXI4-Lite register map)
 * -------------------------------------------------------------------------- */
#define DPU_REG_CMD         0x00    /* [W]   [2:0] cmd_type                   */
#define DPU_REG_ADDR        0x04    /* [W]   [23:0] PIO address               */
#define DPU_REG_WDATA       0x08    /* [W]   [7:0] write data (triggers PIO)  */
#define DPU_REG_RDATA       0x0C    /* [R]   [7:0] read result                */
#define DPU_REG_STATUS      0x10    /* [R]   busy/done/layer/reload/ready     */
#define DPU_REG_PERF        0x14    /* [R]   total cycles                     */
#define DPU_REG_IRQ_EN      0x18    /* [RW]  interrupt enables                */
#define DPU_REG_IRQ_STAT    0x1C    /* [R/W1C] interrupt status               */
#define DPU_REG_DMA_TARGET  0x20    /* [W]   DMA target buffer                */
#define DPU_REG_DMA_BASE    0x24    /* [W]   DMA base address                 */
#define DPU_REG_DMA_LENGTH  0x28    /* [W]   DMA byte count                   */
#define DPU_REG_DMA_CTRL    0x2C    /* [W]   [0]=start [1]=dir                */
#define DPU_REG_DMA_STATUS  0x30    /* [R]   [0]=busy [1]=done                */
#define DPU_REG_VERSION     0x34    /* [R]   version (0x00010000)             */

/* STATUS register bit fields */
#define DPU_STATUS_BUSY         (1 << 0)
#define DPU_STATUS_DONE         (1 << 1)
#define DPU_STATUS_LAYER_MASK   (0x1F << 8)
#define DPU_STATUS_LAYER_SHIFT  8
#define DPU_STATUS_RELOAD_REQ   (1 << 16)
#define DPU_STATUS_CMD_READY    (1 << 17)

/* IRQ bits */
#define DPU_IRQ_DONE        (1 << 0)
#define DPU_IRQ_RELOAD      (1 << 1)

/* CMD types (match dpu_top cmd_type encoding) */
#define DPU_CMD_WRITE_BYTE      0
#define DPU_CMD_RUN_LAYER       1
#define DPU_CMD_READ_BYTE       2
#define DPU_CMD_SET_LAYER       3
#define DPU_CMD_RUN_ALL         4
#define DPU_CMD_WRITE_SCALE     5
#define DPU_CMD_WRITE_LAYER_DESC 6

/* DMA targets */
#define DPU_DMA_TARGET_WEIGHT   0
#define DPU_DMA_TARGET_FMAP     1
#define DPU_DMA_TARGET_BIAS     2
#define DPU_DMA_TARGET_SCALE    3
#define DPU_DMA_TARGET_LDESC    4

/* Memory layout constants */
#define DPU_WEIGHT_BUF_SIZE     147456
#define DPU_BIAS_BUF_OFFSET     147456
#define DPU_BIAS_BUF_SIZE       1024    /* 256 x 4 bytes */
#define DPU_FMAP_SIZE           65536
#define DPU_NUM_LAYERS          18

/* --------------------------------------------------------------------------
 * Platform abstraction
 * -------------------------------------------------------------------------- */
typedef struct {
    uintptr_t base_addr;    /* Base address of DPU register space */
} dpu_hal_t;

/**
 * Initialize HAL with the DPU base address.
 * For Zynq: typically 0x43C00000 (from block design).
 */
void dpu_hal_init(dpu_hal_t *hal, uintptr_t base_addr);

/** Write a 32-bit value to a DPU register. */
void dpu_hal_write32(const dpu_hal_t *hal, uint32_t offset, uint32_t value);

/** Read a 32-bit value from a DPU register. */
uint32_t dpu_hal_read32(const dpu_hal_t *hal, uint32_t offset);

/** Issue a PIO command: set CMD + ADDR, then write WDATA to trigger. */
void dpu_hal_pio_cmd(const dpu_hal_t *hal, uint8_t cmd_type,
                     uint32_t addr, uint8_t data);

/** Read DPU status register. */
uint32_t dpu_hal_status(const dpu_hal_t *hal);

/** Check if DPU is busy. */
int dpu_hal_is_busy(const dpu_hal_t *hal);

/** Check if DPU is done. */
int dpu_hal_is_done(const dpu_hal_t *hal);

/** Check if DPU requests weight reload. */
int dpu_hal_reload_requested(const dpu_hal_t *hal);

/** Read performance cycle counter. */
uint32_t dpu_hal_perf_cycles(const dpu_hal_t *hal);

/** Read version register. */
uint32_t dpu_hal_version(const dpu_hal_t *hal);

#endif /* DPU_HAL_H */
