/**
 * DPU YOLOv4-tiny Inference Application
 *
 * Demonstrates full inference pipeline:
 *   1. Initialize DPU
 *   2. Load layer descriptors
 *   3. Load weights + biases
 *   4. Load input image
 *   5. Run 18-layer inference
 *   6. Read output feature maps
 *   7. Report performance
 *
 * Build:
 *   PC sim:  gcc -DPLATFORM_SIM -I../hal -I../driver \
 *            ../hal/dpu_hal.c ../driver/dpu_driver.c main.c -o dpu_app
 *   Zynq:   arm-none-eabi-gcc -I../hal -I../driver \
 *            ../hal/dpu_hal.c ../driver/dpu_driver.c main.c -o dpu_app.elf
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
#define INPUT_H         16
#define INPUT_W         16
#define OUTPUT_C        256
#define OUTPUT_H        1
#define OUTPUT_W        1

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
 * Main
 * -------------------------------------------------------------------------- */
int main(int argc, char *argv[])
{
    const char *weight_file = "weights.bin";
    const char *bias_file   = "biases.bin";
    const char *input_file  = "input.bin";

    if (argc >= 4) {
        weight_file = argv[1];
        bias_file   = argv[2];
        input_file  = argv[3];
    }

    printf("=== DPU YOLOv4-tiny Inference ===\n");
    printf("Weights: %s\n", weight_file);
    printf("Biases:  %s\n", bias_file);
    printf("Input:   %s\n", input_file);
    printf("\n");

    /* ---- 1. Initialize ---- */
    dpu_driver_t drv;
    dpu_driver_init(&drv, DPU_BASE_ADDR);

    uint32_t ver = dpu_hal_version(&drv.hal);
    printf("DPU version: 0x%08x\n", ver);

    /* ---- 2. Load descriptors ---- */
    printf("Loading layer descriptors... ");
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
    printf("\nRunning 18-layer inference...\n");
    uint32_t cycles = dpu_driver_run_inference(&drv);
    printf("Inference complete!\n");
    printf("Total cycles: %u\n", cycles);
    printf("At 100 MHz: %.3f ms\n", (double)cycles / 100000.0);

    /* ---- 7. Read output ---- */
    uint32_t out_size = OUTPUT_C * OUTPUT_H * OUTPUT_W;
    int8_t *output = (int8_t *)malloc(out_size);
    printf("Reading output (%u bytes)... ", out_size);
    dpu_driver_read_output(&drv, output, out_size);
    printf("done.\n");

    /* Print first 32 output values */
    printf("\nOutput (first 32 values):\n  ");
    for (uint32_t i = 0; i < 32 && i < out_size; i++) {
        printf("%4d ", (int)output[i]);
        if ((i + 1) % 16 == 0) printf("\n  ");
    }
    printf("\n");

    free(output);

    printf("\n=== Done ===\n");
    return 0;
}
