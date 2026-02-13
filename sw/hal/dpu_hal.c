/**
 * DPU Hardware Abstraction Layer — Implementation
 *
 * Two platform backends:
 *   PLATFORM_ZYNQ (default): volatile MMIO reads/writes
 *   PLATFORM_SIM:            printf-based trace (for PC testing)
 */

#include "dpu_hal.h"

#ifdef PLATFORM_SIM
#include <stdio.h>
static uint32_t sim_regs[64];  /* simulated register file */
#else
#include <stddef.h>
/* Volatile MMIO access */
#define MMIO_WRITE(addr, val) (*(volatile uint32_t *)(addr) = (val))
#define MMIO_READ(addr)       (*(volatile uint32_t *)(addr))
#endif

void dpu_hal_init(dpu_hal_t *hal, uintptr_t base_addr)
{
    hal->base_addr = base_addr;
#ifdef PLATFORM_SIM
    for (int i = 0; i < 64; i++) sim_regs[i] = 0;
    sim_regs[DPU_REG_VERSION / 4] = 0x00010000;
    printf("[HAL] init base=0x%08lx\n", (unsigned long)base_addr);
#endif
}

void dpu_hal_write32(const dpu_hal_t *hal, uint32_t offset, uint32_t value)
{
#ifdef PLATFORM_SIM
    sim_regs[offset / 4] = value;
    printf("[HAL] W reg[0x%02x] = 0x%08x\n", offset, value);
#else
    MMIO_WRITE(hal->base_addr + offset, value);
#endif
}

uint32_t dpu_hal_read32(const dpu_hal_t *hal, uint32_t offset)
{
#ifdef PLATFORM_SIM
    uint32_t val = sim_regs[offset / 4];
    printf("[HAL] R reg[0x%02x] = 0x%08x\n", offset, val);
    return val;
#else
    return MMIO_READ(hal->base_addr + offset);
#endif
}

void dpu_hal_pio_cmd(const dpu_hal_t *hal, uint8_t cmd_type,
                     uint32_t addr, uint8_t data)
{
    dpu_hal_write32(hal, DPU_REG_CMD,   (uint32_t)cmd_type);
    dpu_hal_write32(hal, DPU_REG_ADDR,  addr);
    dpu_hal_write32(hal, DPU_REG_WDATA, (uint32_t)data);  /* triggers PIO */
}

uint32_t dpu_hal_status(const dpu_hal_t *hal)
{
    return dpu_hal_read32(hal, DPU_REG_STATUS);
}

int dpu_hal_is_busy(const dpu_hal_t *hal)
{
    return (dpu_hal_status(hal) & DPU_STATUS_BUSY) != 0;
}

int dpu_hal_is_done(const dpu_hal_t *hal)
{
    return (dpu_hal_status(hal) & DPU_STATUS_DONE) != 0;
}

int dpu_hal_reload_requested(const dpu_hal_t *hal)
{
    return (dpu_hal_status(hal) & DPU_STATUS_RELOAD_REQ) != 0;
}

uint32_t dpu_hal_perf_cycles(const dpu_hal_t *hal)
{
    return dpu_hal_read32(hal, DPU_REG_PERF);
}

uint32_t dpu_hal_version(const dpu_hal_t *hal)
{
    return dpu_hal_read32(hal, DPU_REG_VERSION);
}
