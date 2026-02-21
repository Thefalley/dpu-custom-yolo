/**
 * DPU Driver — Implementation
 * Full 36-layer YOLOv4-tiny (H0=32, W0=32)
 */

#include "dpu_driver.h"
#include <string.h>

/* --------------------------------------------------------------------------
 * YOLOv4-tiny 36-layer table (matches golden model, H0=32 W0=32)
 *
 * Format: { type, c_in, c_out, h_in, w_in, h_out, w_out, stride, scale }
 *
 * Layer mapping (internal -> darknet):
 *   0-29:  same as darknet 0-29
 *   30:    darknet 31 (route_save)
 *   31:    darknet 32 (conv1x1)
 *   32:    darknet 33 (upsample)
 *   33:    darknet 34 (route_concat)
 *   34:    darknet 35 (conv3x3)
 *   35:    darknet 36 (conv1x1_linear, detection head 2)
 * -------------------------------------------------------------------------- */
static const dpu_layer_desc_t yolov4_tiny_36layer[DPU_NUM_LAYERS] = {
    /*  0 */ { DPU_LAYER_CONV3X3,        3,  32, 32, 32, 16, 16, 2, 655 },
    /*  1 */ { DPU_LAYER_CONV3X3,       32,  64, 16, 16,  8,  8, 2, 655 },
    /*  2 */ { DPU_LAYER_CONV3X3,       64,  64,  8,  8,  8,  8, 1, 655 },
    /*  3 */ { DPU_LAYER_ROUTE_SPLIT,   64,  32,  8,  8,  8,  8, 1, 655 },
    /*  4 */ { DPU_LAYER_CONV3X3,       32,  32,  8,  8,  8,  8, 1, 655 },
    /*  5 */ { DPU_LAYER_CONV3X3,       32,  32,  8,  8,  8,  8, 1, 655 },
    /*  6 */ { DPU_LAYER_ROUTE_CONCAT,  32,  64,  8,  8,  8,  8, 1, 655 },
    /*  7 */ { DPU_LAYER_CONV1X1,       64,  64,  8,  8,  8,  8, 1, 655 },
    /*  8 */ { DPU_LAYER_ROUTE_CONCAT,  64, 128,  8,  8,  8,  8, 1, 655 },
    /*  9 */ { DPU_LAYER_MAXPOOL,      128, 128,  8,  8,  4,  4, 2, 655 },
    /* 10 */ { DPU_LAYER_CONV3X3,      128, 128,  4,  4,  4,  4, 1, 655 },
    /* 11 */ { DPU_LAYER_ROUTE_SPLIT,  128,  64,  4,  4,  4,  4, 1, 655 },
    /* 12 */ { DPU_LAYER_CONV3X3,       64,  64,  4,  4,  4,  4, 1, 655 },
    /* 13 */ { DPU_LAYER_CONV3X3,       64,  64,  4,  4,  4,  4, 1, 655 },
    /* 14 */ { DPU_LAYER_ROUTE_CONCAT,  64, 128,  4,  4,  4,  4, 1, 655 },
    /* 15 */ { DPU_LAYER_CONV1X1,      128, 128,  4,  4,  4,  4, 1, 655 },
    /* 16 */ { DPU_LAYER_ROUTE_CONCAT, 128, 256,  4,  4,  4,  4, 1, 655 },
    /* 17 */ { DPU_LAYER_MAXPOOL,      256, 256,  4,  4,  2,  2, 2, 655 },
    /* 18 */ { DPU_LAYER_CONV3X3,      256, 256,  2,  2,  2,  2, 1, 655 },
    /* 19 */ { DPU_LAYER_ROUTE_SPLIT,  256, 128,  2,  2,  2,  2, 1, 655 },
    /* 20 */ { DPU_LAYER_CONV3X3,      128, 128,  2,  2,  2,  2, 1, 655 },
    /* 21 */ { DPU_LAYER_CONV3X3,      128, 128,  2,  2,  2,  2, 1, 655 },
    /* 22 */ { DPU_LAYER_ROUTE_CONCAT, 128, 256,  2,  2,  2,  2, 1, 655 },
    /* 23 */ { DPU_LAYER_CONV1X1,      256, 256,  2,  2,  2,  2, 1, 655 },
    /* 24 */ { DPU_LAYER_ROUTE_CONCAT, 256, 512,  2,  2,  2,  2, 1, 655 },
    /* 25 */ { DPU_LAYER_MAXPOOL,      512, 512,  2,  2,  1,  1, 2, 655 },
    /* 26 */ { DPU_LAYER_CONV3X3,      512, 512,  1,  1,  1,  1, 1, 655 },
    /* 27 */ { DPU_LAYER_CONV1X1,      512, 256,  1,  1,  1,  1, 1, 655 },
    /* 28 */ { DPU_LAYER_CONV3X3,      256, 512,  1,  1,  1,  1, 1, 655 },
    /* 29 */ { DPU_LAYER_CONV1X1_LIN,  512, 255,  1,  1,  1,  1, 1, 655 }, /* Det head 1 */
    /* 30 */ { DPU_LAYER_ROUTE_SAVE,   256, 256,  1,  1,  1,  1, 1, 655 },
    /* 31 */ { DPU_LAYER_CONV1X1,      256, 128,  1,  1,  1,  1, 1, 655 },
    /* 32 */ { DPU_LAYER_UPSAMPLE,     128, 128,  1,  1,  2,  2, 1, 655 },
    /* 33 */ { DPU_LAYER_ROUTE_CONCAT, 128, 384,  2,  2,  2,  2, 1, 655 },
    /* 34 */ { DPU_LAYER_CONV3X3,      384, 256,  2,  2,  2,  2, 1, 655 },
    /* 35 */ { DPU_LAYER_CONV1X1_LIN,  256, 255,  2,  2,  2,  2, 1, 655 }, /* Det head 2 */
};

/* --------------------------------------------------------------------------
 * Init
 * -------------------------------------------------------------------------- */
void dpu_driver_init(dpu_driver_t *drv, uintptr_t base)
{
    dpu_hal_init(&drv->hal, base);
    memcpy(drv->layers, yolov4_tiny_36layer, sizeof(yolov4_tiny_36layer));

    /* Detection heads */
    drv->det_heads[0] = (dpu_detection_head_t){ 29, 255, 1, 1 };
    drv->det_heads[1] = (dpu_detection_head_t){ 35, 255, 2, 2 };

    /* Compute weight/bias offsets per layer */
    uint32_t w_off = 0;
    uint32_t b_off = 0;
    for (int i = 0; i < DPU_NUM_LAYERS; i++) {
        drv->weight_offsets[i] = w_off;
        drv->bias_offsets[i]   = b_off;

        uint8_t type = drv->layers[i].type;
        uint32_t cin  = drv->layers[i].c_in;
        uint32_t cout = drv->layers[i].c_out;

        if (type == DPU_LAYER_CONV3X3) {
            drv->weight_sizes[i] = cout * 9 * cin;
            drv->bias_sizes[i]   = cout;
        } else if (type == DPU_LAYER_CONV1X1 || type == DPU_LAYER_CONV1X1_LIN) {
            drv->weight_sizes[i] = cout * cin;
            drv->bias_sizes[i]   = cout;
        } else {
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
 * Run full inference (36 layers with reload support)
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
             * Weights are pre-loaded in weight_buf; just continue. */
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
