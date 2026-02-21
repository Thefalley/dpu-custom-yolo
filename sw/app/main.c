/**
 * DPU YOLOv4-tiny Inference Application
 *
 * Full 36-layer inference pipeline:
 *   1. Initialize DPU
 *   2. Load layer descriptors and scales
 *   3. Load weights + biases (from binary files or embedded arrays)
 *   4. Load input image (INT8 CHW, 3x32x32)
 *   5. Run 36-layer inference
 *   6. Read dual detection outputs (layer 29: 255x1x1, layer 35: 255x2x2)
 *   7. Report performance
 *
 * Build:
 *   PC sim:  gcc -DPLATFORM_SIM -I../hal -I../driver \
 *            ../hal/dpu_hal.c ../driver/dpu_driver.c main.c -o dpu_app
 *   Zynq:   Vitis bare-metal project (create_app.tcl)
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../driver/dpu_driver.h"

/* --------------------------------------------------------------------------
 * Configuration
 * -------------------------------------------------------------------------- */
#define DPU_BASE_ADDR   0x43C00000   /* From Vivado block design */
#define INPUT_C         3
#define INPUT_H         32
#define INPUT_W         32
#define INPUT_SIZE      (INPUT_C * INPUT_H * INPUT_W)

/* Detection outputs */
#define DET1_C          255     /* Layer 29 output */
#define DET1_H          1
#define DET1_W          1
#define DET1_SIZE       (DET1_C * DET1_H * DET1_W)

#define DET2_C          255     /* Layer 35 output (final layer) */
#define DET2_H          2
#define DET2_W          2
#define DET2_SIZE       (DET2_C * DET2_H * DET2_W)

/* --------------------------------------------------------------------------
 * Helper: load binary file into buffer
 * -------------------------------------------------------------------------- */
static uint8_t *load_file(const char *path, uint32_t *out_size)
{
    FILE *f = fopen(path, "rb");
    if (!f) {
        printf("ERROR: cannot open %s\n", path);
        return NULL;
    }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    uint8_t *buf = (uint8_t *)malloc(sz);
    if (!buf) {
        fclose(f);
        return NULL;
    }
    fread(buf, 1, sz, f);
    fclose(f);
    *out_size = (uint32_t)sz;
    return buf;
}

/* --------------------------------------------------------------------------
 * Print detection output summary
 * -------------------------------------------------------------------------- */
static void print_detections(const char *name, const int8_t *data,
                             int channels, int grid_h, int grid_w)
{
    int num_anchors = 3;
    int num_classes = 80;
    int box_attrs = 5 + num_classes;  /* tx,ty,tw,th,conf + 80 classes */

    printf("\n  %s (grid %dx%d, %d channels):\n", name, grid_h, grid_w, channels);

    for (int gy = 0; gy < grid_h; gy++) {
        for (int gx = 0; gx < grid_w; gx++) {
            for (int a = 0; a < num_anchors; a++) {
                int base = (a * box_attrs + 4) * grid_h * grid_w + gy * grid_w + gx;
                int8_t conf_raw = data[base];
                /* conf_raw is INT8; positive values indicate higher confidence */
                if (conf_raw > 10) {  /* threshold: ~0.6 after sigmoid */
                    printf("    grid(%d,%d) anchor %d: conf_raw=%d\n",
                           gy, gx, a, conf_raw);
                }
            }
        }
    }
}

/* --------------------------------------------------------------------------
 * Main
 * -------------------------------------------------------------------------- */
int main(int argc, char *argv[])
{
    const char *weight_file = "weights.bin";
    const char *bias_file   = "biases.bin";
    const char *input_file  = "input.bin";
    const char *scale_file  = "scales.bin";

    if (argc >= 5) {
        weight_file = argv[1];
        bias_file   = argv[2];
        input_file  = argv[3];
        scale_file  = argv[4];
    }

    printf("=== DPU YOLOv4-tiny — 36-Layer Inference ===\n");
    printf("Weights: %s\n", weight_file);
    printf("Biases:  %s\n", bias_file);
    printf("Input:   %s\n", input_file);
    printf("Scales:  %s\n", scale_file);
    printf("\n");

    /* ---- 1. Initialize ---- */
    dpu_driver_t drv;
    dpu_driver_init(&drv, DPU_BASE_ADDR);

    uint32_t ver = dpu_hal_version(&drv.hal);
    printf("DPU version: 0x%08x\n", ver);

    /* ---- 2. Load descriptors ---- */
    printf("Loading %d layer descriptors... ", DPU_NUM_LAYERS);
    dpu_driver_load_descriptors(&drv);
    printf("done.\n");

    /* ---- 3. Load weights ---- */
    uint32_t w_size = 0;
    uint8_t *w_data = load_file(weight_file, &w_size);
    if (!w_data) return 1;
    printf("Loading weights (%u bytes)... ", w_size);
    dpu_driver_load_weights(&drv, w_data, w_size);
    printf("done.\n");
    free(w_data);

    /* ---- 4. Load biases ---- */
    uint32_t b_size = 0;
    uint8_t *b_data = load_file(bias_file, &b_size);
    if (!b_data) return 1;
    uint32_t b_count = b_size / 4;
    printf("Loading biases (%u values)... ", b_count);
    dpu_driver_load_biases(&drv, (const int32_t *)b_data, b_count);
    printf("done.\n");
    free(b_data);

    /* ---- 5. Load input ---- */
    uint32_t in_size = 0;
    uint8_t *in_data = load_file(input_file, &in_size);
    if (!in_data) return 1;
    printf("Loading input (%u bytes, %dx%dx%d)... ",
           in_size, INPUT_C, INPUT_H, INPUT_W);
    dpu_driver_load_input(&drv, (const int8_t *)in_data, in_size);
    printf("done.\n");
    free(in_data);

    /* ---- 6. Run inference ---- */
    printf("\nRunning 36-layer inference...\n");
    uint32_t cycles = dpu_driver_run_inference(&drv);
    printf("Inference complete!\n");
    printf("Total cycles: %u\n", cycles);
    printf("At 100 MHz: %.3f ms\n", (double)cycles / 100000.0);

    /* ---- 7. Read final output (layer 35 = last layer) ---- */
    int8_t *output = (int8_t *)malloc(DET2_SIZE);
    printf("Reading output (%u bytes, %dx%dx%d)... ",
           DET2_SIZE, DET2_C, DET2_H, DET2_W);
    dpu_driver_read_output(&drv, output, DET2_SIZE);
    printf("done.\n");

    /* Print first 32 output values */
    printf("\nFinal output (first 32 values):\n  ");
    for (uint32_t i = 0; i < 32 && i < DET2_SIZE; i++) {
        printf("%4d ", (int)output[i]);
        if ((i + 1) % 16 == 0) printf("\n  ");
    }
    printf("\n");

    /* Print detection summary */
    print_detections("Detection Head 2 (layer 35)", output, DET2_C, DET2_H, DET2_W);

    free(output);

    printf("\n=== Done ===\n");
    return 0;
}
