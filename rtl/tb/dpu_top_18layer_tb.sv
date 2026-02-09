// End-to-end TB for dpu_top: 18-layer YOLOv4-tiny (H0=16, W0=16).
// Loads image via PIO, runs all 18 layers, compares with golden per-layer.
`timescale 1ns/1ps

module dpu_top_18layer_tb;
    timeunit 1ns;
    timeprecision 1ps;

    localparam int H0 = 16;
    localparam int W0 = 16;
    localparam int MAX_CH   = 256;
    localparam int MAX_FMAP = 65536;
    localparam int MAX_WBUF = 147456;
    localparam int FMAP_BASE = MAX_WBUF + MAX_CH*4;

    logic clk, rst_n;
    logic        cmd_valid, cmd_ready;
    logic [2:0]  cmd_type;
    logic [23:0] cmd_addr;
    logic [7:0]  cmd_data;
    logic        rsp_valid;
    logic [7:0]  rsp_data;
    logic        busy, dut_done;
    logic [4:0]  current_layer;

    dpu_top #(
        .H0(H0), .W0(W0),
        .MAX_CH(MAX_CH), .MAX_FMAP(MAX_FMAP), .MAX_WBUF(MAX_WBUF)
    ) u_dut (
        .clk(clk), .rst_n(rst_n),
        .cmd_valid(cmd_valid), .cmd_ready(cmd_ready),
        .cmd_type(cmd_type), .cmd_addr(cmd_addr), .cmd_data(cmd_data),
        .rsp_valid(rsp_valid), .rsp_data(rsp_data),
        .busy(busy), .done(dut_done), .current_layer(current_layer)
    );

    initial begin clk = 0; forever #5 clk = ~clk; end
    initial begin #2000000000; $display("TIMEOUT"); $finish; end

    // ---- PIO Tasks ----
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

    task write_byte(input [23:0] addr, input [7:0] data);
        send_cmd(3'd0, addr, data);
    endtask

    task set_layer(input [4:0] layer_idx);
        send_cmd(3'd3, 24'd0, {3'd0, layer_idx});
    endtask

    task run_layer_cmd();
        send_cmd(3'd1, 24'd0, 8'd0);
        @(posedge clk);
        while (!dut_done) @(posedge clk);
    endtask

    task write_scale(input [15:0] scale_val);
        send_cmd(3'd5, 24'd0, scale_val[7:0]);
        send_cmd(3'd5, 24'd1, scale_val[15:8]);
    endtask

    task read_byte(input [23:0] addr, output [7:0] data);
        @(posedge clk);
        cmd_valid <= 1;
        cmd_type  <= 3'd2;
        cmd_addr  <= addr;
        cmd_data  <= 8'd0;
        @(posedge clk);
        while (!cmd_ready) @(posedge clk);
        cmd_valid <= 0;
        @(posedge clk);
        while (!rsp_valid) @(posedge clk);
        data = rsp_data;
        @(posedge clk);
    endtask

    // ---- Buffers ----
    reg [7:0] hex_buf [0:MAX_WBUF-1];
    reg [7:0] exp_buf [0:MAX_FMAP-1];

    // Layer output sizes
    localparam int OSIZE_0  = 32 * (H0/2) * (W0/2);      // 2048
    localparam int OSIZE_1  = 64 * (H0/4) * (W0/4);      // 1024
    localparam int OSIZE_2  = 64 * (H0/4) * (W0/4);      // 1024
    localparam int OSIZE_3  = 32 * (H0/4) * (W0/4);      // 512
    localparam int OSIZE_4  = 32 * (H0/4) * (W0/4);      // 512
    localparam int OSIZE_5  = 32 * (H0/4) * (W0/4);      // 512
    localparam int OSIZE_6  = 64 * (H0/4) * (W0/4);      // 1024
    localparam int OSIZE_7  = 64 * (H0/4) * (W0/4);      // 1024
    localparam int OSIZE_8  = 128 * (H0/4) * (W0/4);     // 2048
    localparam int OSIZE_9  = 128 * (H0/8) * (W0/8);     // 512
    localparam int OSIZE_10 = 128 * (H0/8) * (W0/8);     // 512
    localparam int OSIZE_11 = 64 * (H0/8) * (W0/8);      // 256
    localparam int OSIZE_12 = 64 * (H0/8) * (W0/8);      // 256
    localparam int OSIZE_13 = 64 * (H0/8) * (W0/8);      // 256
    localparam int OSIZE_14 = 128 * (H0/8) * (W0/8);     // 512
    localparam int OSIZE_15 = 128 * (H0/8) * (W0/8);     // 512
    localparam int OSIZE_16 = 256 * (H0/8) * (W0/8);     // 1024
    localparam int OSIZE_17 = 256 * (H0/16) * (W0/16);   // 256

    // Weight sizes
    localparam int WSIZE_0  = 32 * 3 * 9;       // 864
    localparam int WSIZE_1  = 64 * 32 * 9;      // 18432
    localparam int WSIZE_2  = 64 * 64 * 9;      // 36864
    localparam int WSIZE_4  = 32 * 32 * 9;      // 9216
    localparam int WSIZE_5  = 32 * 32 * 9;      // 9216
    localparam int WSIZE_7  = 64 * 64 * 1;      // 4096
    localparam int WSIZE_10 = 128 * 128 * 9;    // 147456
    localparam int WSIZE_12 = 64 * 64 * 9;      // 36864
    localparam int WSIZE_13 = 64 * 64 * 9;      // 36864
    localparam int WSIZE_15 = 128 * 128 * 1;    // 16384

    // C_out per conv layer
    localparam int COUT_0  = 32;
    localparam int COUT_1  = 64;
    localparam int COUT_2  = 64;
    localparam int COUT_4  = 32;
    localparam int COUT_5  = 32;
    localparam int COUT_7  = 64;
    localparam int COUT_10 = 128;
    localparam int COUT_12 = 64;
    localparam int COUT_13 = 64;
    localparam int COUT_15 = 128;

    // ---- Helpers ----
    task load_weights_and_bias(input int wsize, input int cout);
        integer j;
        begin
            for (j = 0; j < wsize; j = j + 1)
                write_byte(j, hex_buf[j]);
            for (j = 0; j < cout * 4; j = j + 1)
                write_byte(MAX_WBUF + j, hex_buf[j]);  // hex_buf reloaded with bias before call
        end
    endtask

    task compare_output(input int layer_idx, input int osize, output int errors);
        integer j;
        reg [7:0] got;
        begin
            errors = 0;
            for (j = 0; j < osize; j = j + 1) begin
                read_byte(j[23:0], got);
                if (got !== exp_buf[j]) begin
                    if (errors < 5)
                        $display("  MISMATCH layer %0d idx %0d: got=%02x exp=%02x", layer_idx, j, got, exp_buf[j]);
                    errors = errors + 1;
                end
            end
        end
    endtask

    // ---- Main ----
    integer i, j, errs, total_errs;
    integer layer_pass [0:17];
    reg [7:0] got;

    initial begin
        cmd_valid = 0; cmd_type = 0; cmd_addr = 0; cmd_data = 0;
        rst_n = 0;
        #100;
        rst_n = 1;
        #20;

        total_errs = 0;
        for (i = 0; i < 18; i = i + 1) layer_pass[i] = 0;

        $display("=== DPU Top 18-Layer TB (H0=%0d, W0=%0d) ===", H0, W0);

        // Load input image
        $display("[1] Loading input image");
        $readmemh("image_sim_out/dpu_top/input_image.hex", hex_buf);
        for (i = 0; i < 768; i = i + 1)
            write_byte(FMAP_BASE + i, hex_buf[i]);

        // Set scale
        $display("[2] Setting scale");
        write_scale(16'h028F);

        // ==== Layer 0 ====
        $display("[Layer 0] Conv3x3 3->32 stride2");
        $readmemh("image_sim_out/dpu_top/layer0_weights.hex", hex_buf);
        for (j = 0; j < WSIZE_0; j = j + 1) write_byte(j, hex_buf[j]);
        $readmemh("image_sim_out/dpu_top/layer0_bias.hex", hex_buf);
        for (j = 0; j < COUT_0 * 4; j = j + 1) write_byte(MAX_WBUF + j, hex_buf[j]);
        set_layer(5'd0); run_layer_cmd();
        $readmemh("image_sim_out/dpu_top/layer0_expected.hex", exp_buf);
        compare_output(0, OSIZE_0, errs);
        layer_pass[0] = (errs == 0); total_errs = total_errs + errs;
        $display("  Layer 0: %s (%0d bytes, %0d err)", errs==0 ? "PASS" : "FAIL", OSIZE_0, errs);

        // ==== Layer 1 ====
        $display("[Layer 1] Conv3x3 32->64 stride2");
        $readmemh("image_sim_out/dpu_top/layer1_weights.hex", hex_buf);
        for (j = 0; j < WSIZE_1; j = j + 1) write_byte(j, hex_buf[j]);
        $readmemh("image_sim_out/dpu_top/layer1_bias.hex", hex_buf);
        for (j = 0; j < COUT_1 * 4; j = j + 1) write_byte(MAX_WBUF + j, hex_buf[j]);
        set_layer(5'd1); run_layer_cmd();
        $readmemh("image_sim_out/dpu_top/layer1_expected.hex", exp_buf);
        compare_output(1, OSIZE_1, errs);
        layer_pass[1] = (errs == 0); total_errs = total_errs + errs;
        $display("  Layer 1: %s (%0d bytes, %0d err)", errs==0 ? "PASS" : "FAIL", OSIZE_1, errs);

        // ==== Layer 2 ====
        $display("[Layer 2] Conv3x3 64->64 stride1");
        $readmemh("image_sim_out/dpu_top/layer2_weights.hex", hex_buf);
        for (j = 0; j < WSIZE_2; j = j + 1) write_byte(j, hex_buf[j]);
        $readmemh("image_sim_out/dpu_top/layer2_bias.hex", hex_buf);
        for (j = 0; j < COUT_2 * 4; j = j + 1) write_byte(MAX_WBUF + j, hex_buf[j]);
        set_layer(5'd2); run_layer_cmd();
        $readmemh("image_sim_out/dpu_top/layer2_expected.hex", exp_buf);
        compare_output(2, OSIZE_2, errs);
        layer_pass[2] = (errs == 0); total_errs = total_errs + errs;
        $display("  Layer 2: %s (%0d bytes, %0d err)", errs==0 ? "PASS" : "FAIL", OSIZE_2, errs);

        // ==== Layer 3 (Route Split) ====
        $display("[Layer 3] Route split");
        set_layer(5'd3); run_layer_cmd();
        $readmemh("image_sim_out/dpu_top/layer3_expected.hex", exp_buf);
        compare_output(3, OSIZE_3, errs);
        layer_pass[3] = (errs == 0); total_errs = total_errs + errs;
        $display("  Layer 3: %s (%0d bytes, %0d err)", errs==0 ? "PASS" : "FAIL", OSIZE_3, errs);

        // ==== Layer 4 ====
        $display("[Layer 4] Conv3x3 32->32 stride1");
        $readmemh("image_sim_out/dpu_top/layer4_weights.hex", hex_buf);
        for (j = 0; j < WSIZE_4; j = j + 1) write_byte(j, hex_buf[j]);
        $readmemh("image_sim_out/dpu_top/layer4_bias.hex", hex_buf);
        for (j = 0; j < COUT_4 * 4; j = j + 1) write_byte(MAX_WBUF + j, hex_buf[j]);
        set_layer(5'd4); run_layer_cmd();
        $readmemh("image_sim_out/dpu_top/layer4_expected.hex", exp_buf);
        compare_output(4, OSIZE_4, errs);
        layer_pass[4] = (errs == 0); total_errs = total_errs + errs;
        $display("  Layer 4: %s (%0d bytes, %0d err)", errs==0 ? "PASS" : "FAIL", OSIZE_4, errs);

        // ==== Layer 5 ====
        $display("[Layer 5] Conv3x3 32->32 stride1");
        $readmemh("image_sim_out/dpu_top/layer5_weights.hex", hex_buf);
        for (j = 0; j < WSIZE_5; j = j + 1) write_byte(j, hex_buf[j]);
        $readmemh("image_sim_out/dpu_top/layer5_bias.hex", hex_buf);
        for (j = 0; j < COUT_5 * 4; j = j + 1) write_byte(MAX_WBUF + j, hex_buf[j]);
        set_layer(5'd5); run_layer_cmd();
        $readmemh("image_sim_out/dpu_top/layer5_expected.hex", exp_buf);
        compare_output(5, OSIZE_5, errs);
        layer_pass[5] = (errs == 0); total_errs = total_errs + errs;
        $display("  Layer 5: %s (%0d bytes, %0d err)", errs==0 ? "PASS" : "FAIL", OSIZE_5, errs);

        // ==== Layer 6 (Route Concat) ====
        $display("[Layer 6] Route concat (L5+L4_save)");
        set_layer(5'd6); run_layer_cmd();
        $readmemh("image_sim_out/dpu_top/layer6_expected.hex", exp_buf);
        compare_output(6, OSIZE_6, errs);
        layer_pass[6] = (errs == 0); total_errs = total_errs + errs;
        $display("  Layer 6: %s (%0d bytes, %0d err)", errs==0 ? "PASS" : "FAIL", OSIZE_6, errs);

        // ==== Layer 7 ====
        $display("[Layer 7] Conv1x1 64->64");
        $readmemh("image_sim_out/dpu_top/layer7_weights.hex", hex_buf);
        for (j = 0; j < WSIZE_7; j = j + 1) write_byte(j, hex_buf[j]);
        $readmemh("image_sim_out/dpu_top/layer7_bias.hex", hex_buf);
        for (j = 0; j < COUT_7 * 4; j = j + 1) write_byte(MAX_WBUF + j, hex_buf[j]);
        set_layer(5'd7); run_layer_cmd();
        $readmemh("image_sim_out/dpu_top/layer7_expected.hex", exp_buf);
        compare_output(7, OSIZE_7, errs);
        layer_pass[7] = (errs == 0); total_errs = total_errs + errs;
        $display("  Layer 7: %s (%0d bytes, %0d err)", errs==0 ? "PASS" : "FAIL", OSIZE_7, errs);

        // ==== Layer 8 (Route Concat) ====
        $display("[Layer 8] Route concat (L2_save+L7)");
        set_layer(5'd8); run_layer_cmd();
        $readmemh("image_sim_out/dpu_top/layer8_expected.hex", exp_buf);
        compare_output(8, OSIZE_8, errs);
        layer_pass[8] = (errs == 0); total_errs = total_errs + errs;
        $display("  Layer 8: %s (%0d bytes, %0d err)", errs==0 ? "PASS" : "FAIL", OSIZE_8, errs);

        // ==== Layer 9 (MaxPool) ====
        $display("[Layer 9] MaxPool 2x2");
        set_layer(5'd9); run_layer_cmd();
        $readmemh("image_sim_out/dpu_top/layer9_expected.hex", exp_buf);
        compare_output(9, OSIZE_9, errs);
        layer_pass[9] = (errs == 0); total_errs = total_errs + errs;
        $display("  Layer 9: %s (%0d bytes, %0d err)", errs==0 ? "PASS" : "FAIL", OSIZE_9, errs);

        // ==== Layer 10 ====
        $display("[Layer 10] Conv3x3 128->128 stride1");
        $readmemh("image_sim_out/dpu_top/layer10_weights.hex", hex_buf);
        for (j = 0; j < WSIZE_10; j = j + 1) write_byte(j, hex_buf[j]);
        $readmemh("image_sim_out/dpu_top/layer10_bias.hex", hex_buf);
        for (j = 0; j < COUT_10 * 4; j = j + 1) write_byte(MAX_WBUF + j, hex_buf[j]);
        set_layer(5'd10); run_layer_cmd();
        $readmemh("image_sim_out/dpu_top/layer10_expected.hex", exp_buf);
        compare_output(10, OSIZE_10, errs);
        layer_pass[10] = (errs == 0); total_errs = total_errs + errs;
        $display("  Layer 10: %s (%0d bytes, %0d err)", errs==0 ? "PASS" : "FAIL", OSIZE_10, errs);

        // ==== Layer 11 (Route Split) ====
        $display("[Layer 11] Route split");
        set_layer(5'd11); run_layer_cmd();
        $readmemh("image_sim_out/dpu_top/layer11_expected.hex", exp_buf);
        compare_output(11, OSIZE_11, errs);
        layer_pass[11] = (errs == 0); total_errs = total_errs + errs;
        $display("  Layer 11: %s (%0d bytes, %0d err)", errs==0 ? "PASS" : "FAIL", OSIZE_11, errs);

        // ==== Layer 12 ====
        $display("[Layer 12] Conv3x3 64->64 stride1");
        $readmemh("image_sim_out/dpu_top/layer12_weights.hex", hex_buf);
        for (j = 0; j < WSIZE_12; j = j + 1) write_byte(j, hex_buf[j]);
        $readmemh("image_sim_out/dpu_top/layer12_bias.hex", hex_buf);
        for (j = 0; j < COUT_12 * 4; j = j + 1) write_byte(MAX_WBUF + j, hex_buf[j]);
        set_layer(5'd12); run_layer_cmd();
        $readmemh("image_sim_out/dpu_top/layer12_expected.hex", exp_buf);
        compare_output(12, OSIZE_12, errs);
        layer_pass[12] = (errs == 0); total_errs = total_errs + errs;
        $display("  Layer 12: %s (%0d bytes, %0d err)", errs==0 ? "PASS" : "FAIL", OSIZE_12, errs);

        // ==== Layer 13 ====
        $display("[Layer 13] Conv3x3 64->64 stride1");
        $readmemh("image_sim_out/dpu_top/layer13_weights.hex", hex_buf);
        for (j = 0; j < WSIZE_13; j = j + 1) write_byte(j, hex_buf[j]);
        $readmemh("image_sim_out/dpu_top/layer13_bias.hex", hex_buf);
        for (j = 0; j < COUT_13 * 4; j = j + 1) write_byte(MAX_WBUF + j, hex_buf[j]);
        set_layer(5'd13); run_layer_cmd();
        $readmemh("image_sim_out/dpu_top/layer13_expected.hex", exp_buf);
        compare_output(13, OSIZE_13, errs);
        layer_pass[13] = (errs == 0); total_errs = total_errs + errs;
        $display("  Layer 13: %s (%0d bytes, %0d err)", errs==0 ? "PASS" : "FAIL", OSIZE_13, errs);

        // ==== Layer 14 (Route Concat) ====
        $display("[Layer 14] Route concat (L13+L12_save)");
        set_layer(5'd14); run_layer_cmd();
        $readmemh("image_sim_out/dpu_top/layer14_expected.hex", exp_buf);
        compare_output(14, OSIZE_14, errs);
        layer_pass[14] = (errs == 0); total_errs = total_errs + errs;
        $display("  Layer 14: %s (%0d bytes, %0d err)", errs==0 ? "PASS" : "FAIL", OSIZE_14, errs);

        // ==== Layer 15 ====
        $display("[Layer 15] Conv1x1 128->128");
        $readmemh("image_sim_out/dpu_top/layer15_weights.hex", hex_buf);
        for (j = 0; j < WSIZE_15; j = j + 1) write_byte(j, hex_buf[j]);
        $readmemh("image_sim_out/dpu_top/layer15_bias.hex", hex_buf);
        for (j = 0; j < COUT_15 * 4; j = j + 1) write_byte(MAX_WBUF + j, hex_buf[j]);
        set_layer(5'd15); run_layer_cmd();
        $readmemh("image_sim_out/dpu_top/layer15_expected.hex", exp_buf);
        compare_output(15, OSIZE_15, errs);
        layer_pass[15] = (errs == 0); total_errs = total_errs + errs;
        $display("  Layer 15: %s (%0d bytes, %0d err)", errs==0 ? "PASS" : "FAIL", OSIZE_15, errs);

        // ==== Layer 16 (Route Concat) ====
        $display("[Layer 16] Route concat (L10_save+L15)");
        set_layer(5'd16); run_layer_cmd();
        $readmemh("image_sim_out/dpu_top/layer16_expected.hex", exp_buf);
        compare_output(16, OSIZE_16, errs);
        layer_pass[16] = (errs == 0); total_errs = total_errs + errs;
        $display("  Layer 16: %s (%0d bytes, %0d err)", errs==0 ? "PASS" : "FAIL", OSIZE_16, errs);

        // ==== Layer 17 (MaxPool) ====
        $display("[Layer 17] MaxPool 2x2");
        set_layer(5'd17); run_layer_cmd();
        $readmemh("image_sim_out/dpu_top/layer17_expected.hex", exp_buf);
        compare_output(17, OSIZE_17, errs);
        layer_pass[17] = (errs == 0); total_errs = total_errs + errs;
        $display("  Layer 17: %s (%0d bytes, %0d err)", errs==0 ? "PASS" : "FAIL", OSIZE_17, errs);

        // ---- Summary ----
        $display("");
        $display("=== SUMMARY ===");
        errs = 0;
        for (i = 0; i < 18; i = i + 1) begin
            if (layer_pass[i])
                $display("  Layer %2d: PASS", i);
            else begin
                $display("  Layer %2d: FAIL", i);
                errs = errs + 1;
            end
        end

        if (errs == 0)
            $display("RESULT: ALL 18 LAYERS PASS");
        else
            $display("RESULT: %0d LAYERS FAILED", errs);

        $finish;
    end

endmodule
