/**
 * DPU Driver — Implementation
 */

#include "dpu_driver.h"
#include <string.h>

/* --------------------------------------------------------------------------
 * YOLOv4-tiny layer table (matches golden model, H0=W0 parameterized)
 * -------------------------------------------------------------------------- */
static const dpu_layer_desc_t yolov4_tiny_layers[DPU_NUM_LAYERS] = {
    /*  0 */ { 0,   3,  32, 16, 16,  8,  8, 2, 655 },  /* Conv3x3 s2 */
    /*  1 */ { 0,  32,  64,  8,  8,  4,  4, 2, 655 },  /* Conv3x3 s2 */
    /*  2 */ { 0,  64,  64,  4,  4,  4,  4, 1, 655 },  /* Conv3x3    */
    /*  3 */ { 2,  64,  32,  4,  4,  4,  4, 1, 655 },  /* RouteSplit  */
    /*  4 */ { 0,  32,  32,  4,  4,  4,  4, 1, 655 },  /* Conv3x3    */
    /*  5 */ { 0,  32,  32,  4,  4,  4,  4, 1, 655 },  /* Conv3x3    */
    /*  6 */ { 3,  32,  64,  4,  4,  4,  4, 1, 655 },  /* RouteConcat */
    /*  7 */ { 1,  64,  64,  4,  4,  4,  4, 1, 655 },  /* Conv1x1    */
    /*  8 */ { 3,  64, 128,  4,  4,  4,  4, 1, 655 },  /* RouteConcat */
    /*  9 */ { 4, 128, 128,  4,  4,  2,  2, 2, 655 },  /* MaxPool s2 */
    /* 10 */ { 0, 128, 128,  2,  2,  2,  2, 1, 655 },  /* Conv3x3    */
    /* 11 */ { 2, 128,  64,  2,  2,  2,  2, 1, 655 },  /* RouteSplit  */
    /* 12 */ { 0,  64,  64,  2,  2,  2,  2, 1, 655 },  /* Conv3x3    */
    /* 13 */ { 0,  64,  64,  2,  2,  2,  2, 1, 655 },  /* Conv3x3    */
    /* 14 */ { 3,  64, 128,  2,  2,  2,  2, 1, 655 },  /* RouteConcat */
    /* 15 */ { 1, 128, 128,  2,  2,  2,  2, 1, 655 },  /* Conv1x1    */
    /* 16 */ { 3, 128, 256,  2,  2,  2,  2, 1, 655 },  /* RouteConcat */
    /* 17 */ { 4, 256, 256,  2,  2,  1,  1, 2, 655 },  /* MaxPool s2 */
};

/* --------------------------------------------------------------------------
 * Init
 * -------------------------------------------------------------------------- */
void dpu_driver_init(dpu_driver_t *drv, uintptr_t base)
{
    dpu_hal_init(&drv->hal, base);
    memcpy(drv->layers, yolov4_tiny_layers, sizeof(yolov4_tiny_layers));

    /* Compute weight/bias offsets per layer */
    uint32_t w_off = 0;
    uint32_t b_off = 0;
    for (int i = 0; i < DPU_NUM_LAYERS; i++) {
        drv->weight_offsets[i] = w_off;
        drv->bias_offsets[i]   = b_off;

        uint8_t type = drv->layers[i].type;
        if (type == 0) {
            /* Conv3x3: c_out * 9 * c_in weights, c_out biases */
            drv->weight_sizes[i] = drv->layers[i].c_out * 9 * drv->layers[i].c_in;
            drv->bias_sizes[i]   = drv->layers[i].c_out;
        } else if (type == 1) {
            /* Conv1x1: c_out * c_in weights, c_out biases */
            drv->weight_sizes[i] = drv->layers[i].c_out * drv->layers[i].c_in;
            drv->bias_sizes[i]   = drv->layers[i].c_out;
        } else {
            /* Route/MaxPool: no weights */
            drv->weight_sizes[i] = 0;
            drv->bias_sizes[i]   = 0;
        }
        w_off += drv->weight_sizes[i];
        b_off += drv->bias_sizes[i];
    }
}

/* --------------------------------------------------------------------------
 * Load layer descriptors via PIO cmd_type 6
 * -------------------------------------------------------------------------- */
static void write_layer_field(dpu_driver_t *drv, uint8_t layer, uint8_t field, uint8_t val)
{
    uint32_t addr = ((uint32_t)layer << 4) | field;
    dpu_hal_pio_cmd(&drv->hal, DPU_CMD_WRITE_LAYER_DESC, addr, val);
}

void dpu_driver_load_descriptors(dpu_driver_t *drv)
{
    for (int i = 0; i < DPU_NUM_LAYERS; i++) {
        const dpu_layer_desc_t *ld = &drv->layers[i];

        write_layer_field(drv, i,  0, ld->type);
        write_layer_field(drv, i,  1, (uint8_t)(ld->c_in  & 0xFF));
        write_layer_field(drv, i,  2, (uint8_t)(ld->c_in  >> 8));
        write_layer_field(drv, i,  3, (uint8_t)(ld->c_out & 0xFF));
        write_layer_field(drv, i,  4, (uint8_t)(ld->c_out >> 8));
        write_layer_field(drv, i,  5, (uint8_t)(ld->h_in  & 0xFF));
        write_layer_field(drv, i,  6, (uint8_t)(ld->h_in  >> 8));
        write_layer_field(drv, i,  7, (uint8_t)(ld->w_in  & 0xFF));
        write_layer_field(drv, i,  8, (uint8_t)(ld->w_in  >> 8));
        write_layer_field(drv, i,  9, (uint8_t)(ld->h_out & 0xFF));
        write_layer_field(drv, i, 10, (uint8_t)(ld->h_out >> 8));
        write_layer_field(drv, i, 11, (uint8_t)(ld->w_out & 0xFF));
        write_layer_field(drv, i, 12, (uint8_t)(ld->w_out >> 8));
        write_layer_field(drv, i, 13, ld->stride);
        write_layer_field(drv, i, 14, (uint8_t)(ld->scale & 0xFF));
        write_layer_field(drv, i, 15, (uint8_t)(ld->scale >> 8));
    }
}

/* --------------------------------------------------------------------------
 * Load weights via PIO
 * -------------------------------------------------------------------------- */
void dpu_driver_load_weights(dpu_driver_t *drv,
                             const uint8_t *weights, uint32_t total_bytes)
{
    for (uint32_t i = 0; i < total_bytes; i++) {
        dpu_hal_pio_cmd(&drv->hal, DPU_CMD_WRITE_BYTE, i, weights[i]);
    }
}

/* --------------------------------------------------------------------------
 * Load biases via PIO
 * -------------------------------------------------------------------------- */
void dpu_driver_load_biases(dpu_driver_t *drv,
                            const int32_t *biases, uint32_t total_count)
{
    uint32_t base = DPU_BIAS_BUF_OFFSET;
    for (uint32_t i = 0; i < total_count; i++) {
        uint32_t val = (uint32_t)biases[i];
        /* Write 4 bytes per bias (little-endian) */
        dpu_hal_pio_cmd(&drv->hal, DPU_CMD_WRITE_BYTE, base + i * 4 + 0, (val >>  0) & 0xFF);
        dpu_hal_pio_cmd(&drv->hal, DPU_CMD_WRITE_BYTE, base + i * 4 + 1, (val >>  8) & 0xFF);
        dpu_hal_pio_cmd(&drv->hal, DPU_CMD_WRITE_BYTE, base + i * 4 + 2, (val >> 16) & 0xFF);
        dpu_hal_pio_cmd(&drv->hal, DPU_CMD_WRITE_BYTE, base + i * 4 + 3, (val >> 24) & 0xFF);
    }
}

/* --------------------------------------------------------------------------
 * Load input image
 * -------------------------------------------------------------------------- */
void dpu_driver_load_input(dpu_driver_t *drv,
                           const int8_t *input, uint32_t size)
{
    /* Input goes to fmap region: offset after weight_buf + bias_buf */
    uint32_t fmap_base = DPU_WEIGHT_BUF_SIZE + DPU_BIAS_BUF_SIZE;
    for (uint32_t i = 0; i < size; i++) {
        dpu_hal_pio_cmd(&drv->hal, DPU_CMD_WRITE_BYTE,
                        fmap_base + i, (uint8_t)input[i]);
    }
}

/* --------------------------------------------------------------------------
 * Run full inference
 * -------------------------------------------------------------------------- */
uint32_t dpu_driver_run_inference(dpu_driver_t *drv)
{
    /* Start run_all mode */
    dpu_hal_pio_cmd(&drv->hal, DPU_CMD_RUN_ALL, 0, 0);

    /* Wait for completion, handling reload requests */
    while (1) {
        uint32_t status = dpu_hal_status(&drv->hal);

        if (status & DPU_STATUS_DONE)
            break;

        if (status & DPU_STATUS_RELOAD_REQ) {
            /* DPU paused before a conv layer needing weights.
             * In a real system, reload weights for the current layer.
             * For now, weights are pre-loaded; just issue continue (run_layer). */
            dpu_hal_pio_cmd(&drv->hal, DPU_CMD_RUN_LAYER, 0, 0);
        }
    }

    return dpu_hal_perf_cycles(&drv->hal);
}

/* --------------------------------------------------------------------------
 * Read output
 * -------------------------------------------------------------------------- */
void dpu_driver_read_output(dpu_driver_t *drv,
                            int8_t *output, uint32_t size)
{
    for (uint32_t i = 0; i < size; i++) {
        dpu_hal_pio_cmd(&drv->hal, DPU_CMD_READ_BYTE, i, 0);
        uint32_t rdata = dpu_hal_read32(&drv->hal, DPU_REG_RDATA);
        output[i] = (int8_t)(rdata & 0xFF);
    }
}

/* --------------------------------------------------------------------------
 * Single layer execution
 * -------------------------------------------------------------------------- */
void dpu_driver_run_layer(dpu_driver_t *drv, uint8_t layer_id)
{
    dpu_hal_pio_cmd(&drv->hal, DPU_CMD_SET_LAYER, 0, layer_id);
    dpu_hal_pio_cmd(&drv->hal, DPU_CMD_RUN_LAYER, 0, 0);

    while (dpu_hal_is_busy(&drv->hal)) {
        /* spin */
    }
}

uint8_t dpu_driver_current_layer(const dpu_driver_t *drv)
{
    uint32_t st = dpu_hal_status(&drv->hal);
    return (uint8_t)((st & DPU_STATUS_LAYER_MASK) >> DPU_STATUS_LAYER_SHIFT);
}

uint32_t dpu_driver_perf_cycles(const dpu_driver_t *drv)
{
    return dpu_hal_perf_cycles(&drv->hal);
}
