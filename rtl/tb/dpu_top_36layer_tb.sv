// End-to-end TB for dpu_top: 36-layer YOLOv4-tiny (H0=32, W0=32).
// Full network (darknet 0-29, 31-36; skipping YOLO decode 30, 37).
// Uses hierarchical references for fast weight loading, PIO for control.
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
    // Layer 35 output: 255 * 2 * 2 = 1020
    localparam int FINAL_OSIZE = 255 * (H0/16) * (W0/16);

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
        .perf_total_cycles(perf_total_cycles)
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

    // ---- Main Test ----
    integer i, errs;
    reg all_done_flag;

    initial begin
        cmd_valid = 0; cmd_type = 0; cmd_addr = 0; cmd_data = 0;
        rst_n = 0;
        #100;
        rst_n = 1;
        #20;

        $display("=== DPU Top 36-Layer TB (H0=%0d, W0=%0d, NUM_LAYERS=%0d) ===", H0, W0, NUM_LAYERS);

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
        $display("[4] Starting run_all (36 layers)");
        run_all_cmd();

        // [5] Reload loop: wait for reload_req or dut_done
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

        // [6] Compare final output (layer 35)
        // After 36 layers, ping_pong=0, output is in fmap_a
        $readmemh("image_sim_out/dpu_top_37/layer35_expected.hex", exp_buf);
        errs = 0;
        for (i = 0; i < FINAL_OSIZE; i = i + 1) begin
            if (u_dut.fmap_a[i] !== $signed(exp_buf[i])) begin
                if (errs < 20)
                    $display("  MISMATCH L35 idx %0d: got=%02x exp=%02x",
                             i, u_dut.fmap_a[i] & 8'hff, exp_buf[i]);
                errs = errs + 1;
            end
        end

        $display("  Final output (layer 35): %s (%0d bytes, %0d errors)",
                 errs==0 ? "PASS" : "FAIL", FINAL_OSIZE, errs);

        // ---- Summary ----
        $display("");
        if (errs == 0)
            $display("RESULT: ALL 36 LAYERS PASS");
        else
            $display("RESULT: FAILED (%0d errors in final output)", errs);

        // ---- Performance Report ----
        $display("");
        $display("=== PERFORMANCE (cycles) ===");
        for (i = 0; i < NUM_LAYERS; i = i + 1)
            $display("  Layer %2d: %0d cycles", i, u_dut.layer_cycles[i]);
        $display("  TOTAL compute: %0d cycles", perf_total_cycles);

        $finish;
    end

endmodule
