// End-to-end TB for dpu_top: 36-layer YOLOv4-tiny (H0=32, W0=32).
// Full network (darknet 0-29, 31-36; skipping YOLO decode 30, 37).
// Uses hierarchical references for fast weight loading, PIO for control.
//
// PER-LAYER BIT-EXACT VERIFICATION:
//   After each layer completes (layer_done_pulse), this TB loads the
//   golden expected output and compares every byte against the RTL output.
//   This ensures no error accumulation — each layer must match exactly.
`timescale 1ns/1ps

module dpu_top_36layer_tb;
    timeunit 1ns;
    timeprecision 1ps;

    localparam int H0 = 32;
    localparam int W0 = 32;
    localparam int MAX_CH   = 512;
    localparam int MAX_FMAP = 65536;
    localparam int MAX_WBUF = 2400000;
    localparam int NUM_LAYERS = 36;

    localparam int INPUT_SIZE = 3 * H0 * W0;  // 3072

    logic clk, rst_n;
    logic        cmd_valid, cmd_ready;
    logic [2:0]  cmd_type;
    logic [23:0] cmd_addr;
    logic [7:0]  cmd_data;
    logic        rsp_valid;
    logic [7:0]  rsp_data;
    logic        busy, dut_done;
    logic [5:0]  current_layer;
    logic        reload_req;
    logic [31:0] perf_total_cycles;
    logic        layer_done_pulse;
    logic [5:0]  done_layer_idx;

    dpu_top #(
        .H0(H0), .W0(W0),
        .MAX_CH(MAX_CH), .MAX_FMAP(MAX_FMAP), .MAX_WBUF(MAX_WBUF),
        .NUM_LAYERS(NUM_LAYERS)
    ) u_dut (
        .clk(clk), .rst_n(rst_n),
        .cmd_valid(cmd_valid), .cmd_ready(cmd_ready),
        .cmd_type(cmd_type), .cmd_addr(cmd_addr), .cmd_data(cmd_data),
        .rsp_valid(rsp_valid), .rsp_data(rsp_data),
        .busy(busy), .done(dut_done), .current_layer(current_layer),
        .reload_req(reload_req),
        .perf_total_cycles(perf_total_cycles),
        .layer_done_pulse(layer_done_pulse),
        .done_layer_idx(done_layer_idx)
    );

    initial begin clk = 0; forever #5 clk = ~clk; end
    initial begin #10000000000; $display("TIMEOUT"); $finish; end

    // ---- PIO Tasks (control only) ----
    task send_cmd(input [2:0] ctype, input [23:0] addr, input [7:0] data);
        @(posedge clk);
        cmd_valid <= 1;
        cmd_type  <= ctype;
        cmd_addr  <= addr;
        cmd_data  <= data;
        @(posedge clk);
        while (!cmd_ready) @(posedge clk);
        cmd_valid <= 0;
        @(posedge clk);
    endtask

    task run_all_cmd();
        send_cmd(3'd4, 24'd0, 8'd0);
    endtask

    task continue_cmd();
        send_cmd(3'd1, 24'd0, 8'd0);
    endtask

    task wait_reload_or_done();
        @(posedge clk);
        while (!reload_req && !dut_done) @(posedge clk);
    endtask

    // ---- Buffers ----
    reg [7:0] bias_hex  [0:2047];   // max 512 ch * 4 bytes
    reg [7:0] scale_buf [0:71];     // 36 layers * 2 bytes LE
    reg [7:0] exp_buf   [0:MAX_FMAP-1];

    // ---- Per-layer verification state ----
    integer layer_errs     [0:NUM_LAYERS-1];
    integer layer_sizes    [0:NUM_LAYERS-1];
    integer total_layer_fails;

    // Output sizes for all 36 layers (precomputed from layer descriptors)
    // Format: c_out * h_out * w_out
    initial begin
        // 1st CSP block
        layer_sizes[ 0] =  32 * (H0/2)   * (W0/2);      // 32*16*16 = 8192
        layer_sizes[ 1] =  64 * (H0/4)   * (W0/4);      // 64*8*8   = 4096
        layer_sizes[ 2] =  64 * (H0/4)   * (W0/4);      // 64*8*8
        layer_sizes[ 3] =  32 * (H0/4)   * (W0/4);      // 32*8*8   = 2048
        layer_sizes[ 4] =  32 * (H0/4)   * (W0/4);      // 32*8*8
        layer_sizes[ 5] =  32 * (H0/4)   * (W0/4);      // 32*8*8
        layer_sizes[ 6] =  64 * (H0/4)   * (W0/4);      // 64*8*8
        layer_sizes[ 7] =  64 * (H0/4)   * (W0/4);      // 64*8*8
        layer_sizes[ 8] = 128 * (H0/4)   * (W0/4);      // 128*8*8  = 8192
        layer_sizes[ 9] = 128 * (H0/8)   * (W0/8);      // 128*4*4  = 2048
        // 2nd CSP block
        layer_sizes[10] = 128 * (H0/8)   * (W0/8);      // 128*4*4
        layer_sizes[11] =  64 * (H0/8)   * (W0/8);      // 64*4*4   = 1024
        layer_sizes[12] =  64 * (H0/8)   * (W0/8);      // 64*4*4
        layer_sizes[13] =  64 * (H0/8)   * (W0/8);      // 64*4*4
        layer_sizes[14] = 128 * (H0/8)   * (W0/8);      // 128*4*4
        layer_sizes[15] = 128 * (H0/8)   * (W0/8);      // 128*4*4
        layer_sizes[16] = 256 * (H0/8)   * (W0/8);      // 256*4*4  = 4096
        layer_sizes[17] = 256 * (H0/16)  * (W0/16);     // 256*2*2  = 1024
        // 3rd CSP block
        layer_sizes[18] = 256 * (H0/16)  * (W0/16);     // 256*2*2
        layer_sizes[19] = 128 * (H0/16)  * (W0/16);     // 128*2*2  = 512
        layer_sizes[20] = 128 * (H0/16)  * (W0/16);     // 128*2*2
        layer_sizes[21] = 128 * (H0/16)  * (W0/16);     // 128*2*2
        layer_sizes[22] = 256 * (H0/16)  * (W0/16);     // 256*2*2
        layer_sizes[23] = 256 * (H0/16)  * (W0/16);     // 256*2*2
        layer_sizes[24] = 512 * (H0/16)  * (W0/16);     // 512*2*2  = 2048
        layer_sizes[25] = 512 * (H0/32)  * (W0/32);     // 512*1*1  = 512
        // Detection head 1
        layer_sizes[26] = 512 * (H0/32)  * (W0/32);     // 512*1*1
        layer_sizes[27] = 256 * (H0/32)  * (W0/32);     // 256*1*1  = 256
        layer_sizes[28] = 512 * (H0/32)  * (W0/32);     // 512*1*1
        layer_sizes[29] = 255 * (H0/32)  * (W0/32) * 4;  // 255*1*1 * 4 bytes (INT32)
        // Bridge + detection head 2
        layer_sizes[30] = 256 * (H0/32)  * (W0/32);     // 256*1*1  = 256
        layer_sizes[31] = 128 * (H0/32)  * (W0/32);     // 128*1*1  = 128
        layer_sizes[32] = 128 * (H0/16)  * (W0/16);     // 128*2*2  = 512
        layer_sizes[33] = 384 * (H0/16)  * (W0/16);     // 384*2*2  = 1536
        layer_sizes[34] = 256 * (H0/16)  * (W0/16);     // 256*2*2  = 1024
        layer_sizes[35] = 255 * (H0/16)  * (W0/16) * 4;  // 255*2*2 * 4 bytes (INT32)
    end

    // ---- Bias packing: 4 LE bytes -> 32-bit signed ----
    task pack_bias_to_dut(input integer cout);
        integer j;
        begin
            for (j = 0; j < cout; j = j + 1)
                u_dut.bias_buf[j] = {bias_hex[j*4+3], bias_hex[j*4+2],
                                     bias_hex[j*4+1], bias_hex[j*4]};
        end
    endtask

    // ---- Weight/bias loading via hierarchical reference (fast) ----
    task load_conv_layer(input integer layer);
        begin
            case (layer)
                0: begin
                    $readmemh("image_sim_out/dpu_top_37/layer0_weights.hex", u_dut.weight_buf);
                    $readmemh("image_sim_out/dpu_top_37/layer0_bias.hex", bias_hex);
                    pack_bias_to_dut(32);
                end
                1: begin
                    $readmemh("image_sim_out/dpu_top_37/layer1_weights.hex", u_dut.weight_buf);
                    $readmemh("image_sim_out/dpu_top_37/layer1_bias.hex", bias_hex);
                    pack_bias_to_dut(64);
                end
                2: begin
                    $readmemh("image_sim_out/dpu_top_37/layer2_weights.hex", u_dut.weight_buf);
                    $readmemh("image_sim_out/dpu_top_37/layer2_bias.hex", bias_hex);
                    pack_bias_to_dut(64);
                end
                4: begin
                    $readmemh("image_sim_out/dpu_top_37/layer4_weights.hex", u_dut.weight_buf);
                    $readmemh("image_sim_out/dpu_top_37/layer4_bias.hex", bias_hex);
                    pack_bias_to_dut(32);
                end
                5: begin
                    $readmemh("image_sim_out/dpu_top_37/layer5_weights.hex", u_dut.weight_buf);
                    $readmemh("image_sim_out/dpu_top_37/layer5_bias.hex", bias_hex);
                    pack_bias_to_dut(32);
                end
                7: begin
                    $readmemh("image_sim_out/dpu_top_37/layer7_weights.hex", u_dut.weight_buf);
                    $readmemh("image_sim_out/dpu_top_37/layer7_bias.hex", bias_hex);
                    pack_bias_to_dut(64);
                end
                10: begin
                    $readmemh("image_sim_out/dpu_top_37/layer10_weights.hex", u_dut.weight_buf);
                    $readmemh("image_sim_out/dpu_top_37/layer10_bias.hex", bias_hex);
                    pack_bias_to_dut(128);
                end
                12: begin
                    $readmemh("image_sim_out/dpu_top_37/layer12_weights.hex", u_dut.weight_buf);
                    $readmemh("image_sim_out/dpu_top_37/layer12_bias.hex", bias_hex);
                    pack_bias_to_dut(64);
                end
                13: begin
                    $readmemh("image_sim_out/dpu_top_37/layer13_weights.hex", u_dut.weight_buf);
                    $readmemh("image_sim_out/dpu_top_37/layer13_bias.hex", bias_hex);
                    pack_bias_to_dut(64);
                end
                15: begin
                    $readmemh("image_sim_out/dpu_top_37/layer15_weights.hex", u_dut.weight_buf);
                    $readmemh("image_sim_out/dpu_top_37/layer15_bias.hex", bias_hex);
                    pack_bias_to_dut(128);
                end
                18: begin
                    $readmemh("image_sim_out/dpu_top_37/layer18_weights.hex", u_dut.weight_buf);
                    $readmemh("image_sim_out/dpu_top_37/layer18_bias.hex", bias_hex);
                    pack_bias_to_dut(256);
                end
                20: begin
                    $readmemh("image_sim_out/dpu_top_37/layer20_weights.hex", u_dut.weight_buf);
                    $readmemh("image_sim_out/dpu_top_37/layer20_bias.hex", bias_hex);
                    pack_bias_to_dut(128);
                end
                21: begin
                    $readmemh("image_sim_out/dpu_top_37/layer21_weights.hex", u_dut.weight_buf);
                    $readmemh("image_sim_out/dpu_top_37/layer21_bias.hex", bias_hex);
                    pack_bias_to_dut(128);
                end
                23: begin
                    $readmemh("image_sim_out/dpu_top_37/layer23_weights.hex", u_dut.weight_buf);
                    $readmemh("image_sim_out/dpu_top_37/layer23_bias.hex", bias_hex);
                    pack_bias_to_dut(256);
                end
                26: begin
                    $readmemh("image_sim_out/dpu_top_37/layer26_weights.hex", u_dut.weight_buf);
                    $readmemh("image_sim_out/dpu_top_37/layer26_bias.hex", bias_hex);
                    pack_bias_to_dut(512);
                end
                27: begin
                    $readmemh("image_sim_out/dpu_top_37/layer27_weights.hex", u_dut.weight_buf);
                    $readmemh("image_sim_out/dpu_top_37/layer27_bias.hex", bias_hex);
                    pack_bias_to_dut(256);
                end
                28: begin
                    $readmemh("image_sim_out/dpu_top_37/layer28_weights.hex", u_dut.weight_buf);
                    $readmemh("image_sim_out/dpu_top_37/layer28_bias.hex", bias_hex);
                    pack_bias_to_dut(512);
                end
                29: begin
                    $readmemh("image_sim_out/dpu_top_37/layer29_weights.hex", u_dut.weight_buf);
                    $readmemh("image_sim_out/dpu_top_37/layer29_bias.hex", bias_hex);
                    pack_bias_to_dut(255);
                end
                31: begin
                    $readmemh("image_sim_out/dpu_top_37/layer31_weights.hex", u_dut.weight_buf);
                    $readmemh("image_sim_out/dpu_top_37/layer31_bias.hex", bias_hex);
                    pack_bias_to_dut(128);
                end
                34: begin
                    $readmemh("image_sim_out/dpu_top_37/layer34_weights.hex", u_dut.weight_buf);
                    $readmemh("image_sim_out/dpu_top_37/layer34_bias.hex", bias_hex);
                    pack_bias_to_dut(256);
                end
                35: begin
                    $readmemh("image_sim_out/dpu_top_37/layer35_weights.hex", u_dut.weight_buf);
                    $readmemh("image_sim_out/dpu_top_37/layer35_bias.hex", bias_hex);
                    pack_bias_to_dut(255);
                end
                default: $display("WARNING: load_conv_layer called for non-conv layer %0d", layer);
            endcase
        end
    endtask

    // ---- Per-layer verification task ----
    // Called from the concurrent always block when layer_done_pulse fires.
    // Loads expected output hex, compares against RTL fmap, reports errors.
    task verify_layer_output(input integer lidx, input logic pp_new);
        integer vi, sz, errcnt;
        reg signed [7:0] rtl_val;
        reg signed [7:0] exp_val;
        begin
            sz = layer_sizes[lidx];
            // Load expected output for this layer
            case (lidx)
                 0: $readmemh("image_sim_out/dpu_top_37/layer0_expected.hex", exp_buf);
                 1: $readmemh("image_sim_out/dpu_top_37/layer1_expected.hex", exp_buf);
                 2: $readmemh("image_sim_out/dpu_top_37/layer2_expected.hex", exp_buf);
                 3: $readmemh("image_sim_out/dpu_top_37/layer3_expected.hex", exp_buf);
                 4: $readmemh("image_sim_out/dpu_top_37/layer4_expected.hex", exp_buf);
                 5: $readmemh("image_sim_out/dpu_top_37/layer5_expected.hex", exp_buf);
                 6: $readmemh("image_sim_out/dpu_top_37/layer6_expected.hex", exp_buf);
                 7: $readmemh("image_sim_out/dpu_top_37/layer7_expected.hex", exp_buf);
                 8: $readmemh("image_sim_out/dpu_top_37/layer8_expected.hex", exp_buf);
                 9: $readmemh("image_sim_out/dpu_top_37/layer9_expected.hex", exp_buf);
                10: $readmemh("image_sim_out/dpu_top_37/layer10_expected.hex", exp_buf);
                11: $readmemh("image_sim_out/dpu_top_37/layer11_expected.hex", exp_buf);
                12: $readmemh("image_sim_out/dpu_top_37/layer12_expected.hex", exp_buf);
                13: $readmemh("image_sim_out/dpu_top_37/layer13_expected.hex", exp_buf);
                14: $readmemh("image_sim_out/dpu_top_37/layer14_expected.hex", exp_buf);
                15: $readmemh("image_sim_out/dpu_top_37/layer15_expected.hex", exp_buf);
                16: $readmemh("image_sim_out/dpu_top_37/layer16_expected.hex", exp_buf);
                17: $readmemh("image_sim_out/dpu_top_37/layer17_expected.hex", exp_buf);
                18: $readmemh("image_sim_out/dpu_top_37/layer18_expected.hex", exp_buf);
                19: $readmemh("image_sim_out/dpu_top_37/layer19_expected.hex", exp_buf);
                20: $readmemh("image_sim_out/dpu_top_37/layer20_expected.hex", exp_buf);
                21: $readmemh("image_sim_out/dpu_top_37/layer21_expected.hex", exp_buf);
                22: $readmemh("image_sim_out/dpu_top_37/layer22_expected.hex", exp_buf);
                23: $readmemh("image_sim_out/dpu_top_37/layer23_expected.hex", exp_buf);
                24: $readmemh("image_sim_out/dpu_top_37/layer24_expected.hex", exp_buf);
                25: $readmemh("image_sim_out/dpu_top_37/layer25_expected.hex", exp_buf);
                26: $readmemh("image_sim_out/dpu_top_37/layer26_expected.hex", exp_buf);
                27: $readmemh("image_sim_out/dpu_top_37/layer27_expected.hex", exp_buf);
                28: $readmemh("image_sim_out/dpu_top_37/layer28_expected.hex", exp_buf);
                29: $readmemh("image_sim_out/dpu_top_37/layer29_expected_int32.hex", exp_buf);
                30: $readmemh("image_sim_out/dpu_top_37/layer30_expected.hex", exp_buf);
                31: $readmemh("image_sim_out/dpu_top_37/layer31_expected.hex", exp_buf);
                32: $readmemh("image_sim_out/dpu_top_37/layer32_expected.hex", exp_buf);
                33: $readmemh("image_sim_out/dpu_top_37/layer33_expected.hex", exp_buf);
                34: $readmemh("image_sim_out/dpu_top_37/layer34_expected.hex", exp_buf);
                35: $readmemh("image_sim_out/dpu_top_37/layer35_expected_int32.hex", exp_buf);
            endcase

            // Compare: after S_LAYER_DONE, pp is TOGGLED.
            // pp_new==1 means layer ran with old pp==0, wrote to fmap_b
            // pp_new==0 means layer ran with old pp==1, wrote to fmap_a
            errcnt = 0;
            for (vi = 0; vi < sz; vi = vi + 1) begin
                if (pp_new == 1'b1)
                    rtl_val = u_dut.fmap_b[vi];
                else
                    rtl_val = u_dut.fmap_a[vi];
                exp_val = $signed(exp_buf[vi]);
                if (rtl_val !== exp_val) begin
                    if (errcnt < 5)
                        $display("    MISMATCH L%0d idx %0d: rtl=%02x exp=%02x",
                                 lidx, vi, rtl_val & 8'hff, exp_val & 8'hff);
                    errcnt = errcnt + 1;
                end
            end

            layer_errs[lidx] = errcnt;
            if (errcnt == 0)
                $display("  Layer %2d: PASS  (%0d bytes verified)", lidx, sz);
            else begin
                $display("  Layer %2d: FAIL  (%0d errors / %0d bytes)", lidx, errcnt, sz);
                total_layer_fails = total_layer_fails + 1;
            end

            // Export RTL output for detection layers (29, 35) to hex files
            // These layers output INT32 (4 bytes per element in fmap)
            if (lidx == 29 || lidx == 35) begin : export_rtl_output
                integer fd, ei;
                reg [7:0] rv;
                if (lidx == 29)
                    fd = $fopen("image_sim_out/dpu_top_37/rtl_output_layer29.hex", "w");
                else
                    fd = $fopen("image_sim_out/dpu_top_37/rtl_output_layer35.hex", "w");
                if (fd != 0) begin
                    for (ei = 0; ei < sz; ei = ei + 1) begin
                        if (pp_new == 1'b1)
                            rv = u_dut.fmap_b[ei];
                        else
                            rv = u_dut.fmap_a[ei];
                        $fwrite(fd, "%02x\n", rv & 8'hff);
                    end
                    $fclose(fd);
                    $display("  -> RTL output exported: rtl_output_layer%0d.hex (%0d bytes, INT32)", lidx, sz);
                end
            end
        end
    endtask

    // ---- Per-layer verification: concurrent monitor ----
    // Fires when layer_done_pulse goes high, verifies the completed layer.
    always @(posedge clk) begin
        if (layer_done_pulse) begin
            // done_layer_idx = layer that just completed
            // ping_pong = new value (already toggled)
            verify_layer_output(done_layer_idx, u_dut.ping_pong);
        end
    end

    // ---- Main Test ----
    integer i;
    reg all_done_flag;

    initial begin
        cmd_valid = 0; cmd_type = 0; cmd_addr = 0; cmd_data = 0;
        rst_n = 0;
        total_layer_fails = 0;
        for (i = 0; i < NUM_LAYERS; i = i + 1)
            layer_errs[i] = -1;  // -1 = not checked yet
        #100;
        rst_n = 1;
        #20;

        $display("=== DPU Top 36-Layer TB (H0=%0d, W0=%0d, NUM_LAYERS=%0d) ===", H0, W0, NUM_LAYERS);
        $display("=== PER-LAYER BIT-EXACT VERIFICATION ENABLED ===");

        // [1] Load input image directly into DUT fmap_a
        $readmemh("image_sim_out/dpu_top_37/input_image.hex", u_dut.fmap_a);
        $display("[1] Input image loaded (%0d bytes)", INPUT_SIZE);

        // [2] Load per-layer scales
        $readmemh("image_sim_out/dpu_top_37/scales.hex", scale_buf);
        for (i = 0; i < NUM_LAYERS; i = i + 1)
            u_dut.ld_scale[i] = {scale_buf[i*2+1], scale_buf[i*2]};
        $display("[2] Per-layer scales loaded (%0d layers)", NUM_LAYERS);

        // [3] Load initial weights (layer 0)
        load_conv_layer(0);
        $display("[3] Layer 0 weights loaded");

        // [4] Start run_all
        $display("[4] Starting run_all (36 layers) with per-layer verification");
        run_all_cmd();

        // [5] Reload loop
        all_done_flag = 0;
        while (!all_done_flag) begin
            wait_reload_or_done();
            if (dut_done) begin
                all_done_flag = 1;
            end else begin
                $display("  reload_req for layer %0d (t=%0t)", current_layer, $time);
                load_conv_layer(current_layer);
                continue_cmd();
            end
        end

        $display("[5] run_all complete (t=%0t)", $time);

        // Wait a few cycles for the last layer verification to complete
        #100;

        // ---- Layer-by-layer Summary ----
        $display("");
        $display("=== PER-LAYER VERIFICATION SUMMARY ===");
        for (i = 0; i < NUM_LAYERS; i = i + 1) begin
            if (layer_errs[i] == 0)
                $display("  Layer %2d: PASS  (%0d bytes)", i, layer_sizes[i]);
            else if (layer_errs[i] > 0)
                $display("  Layer %2d: FAIL  (%0d errors)", i, layer_errs[i]);
            else
                $display("  Layer %2d: NOT CHECKED", i);
        end

        // ---- Overall Result ----
        $display("");
        if (total_layer_fails == 0)
            $display("RESULT: ALL 36 LAYERS PASS (bit-exact match on every layer)");
        else
            $display("RESULT: FAILED (%0d layers have mismatches)", total_layer_fails);

        // ---- Performance Report ----
        $display("");
        $display("=== PERFORMANCE (cycles) ===");
        for (i = 0; i < NUM_LAYERS; i = i + 1)
            $display("  Layer %2d: %0d cycles", i, u_dut.layer_cycles[i]);
        $display("  TOTAL compute: %0d cycles", perf_total_cycles);

        $finish;
    end

endmodule
