// Layer12 patch TB: one pixel (0,0), 4 output channels (576 MACs + bias + LeakyReLU + requantize each) vs Python golden.
// Layer 12: 64 input channels, 64 output channels, 3x3 kernel -> 64*9 = 576 MACs per output pixel
// Icarus-compatible. Run from project root; hex files in image_sim_out/.
// Generate golden: python run_image_to_detection.py --synthetic --layers 13; python tests/layer12_patch_golden.py
`timescale 1ns/1ps

module layer12_patch_tb;
    timeunit 1ns;
    timeprecision 1ps;

    localparam NUM_CH = 4;
    localparam MACS_PER_CH = 576;  // 64 input channels * 3 * 3

    logic clk, rst_n, valid;
    logic signed [7:0]  weight, activation;
    logic signed [31:0] acc_in, acc_out;
    logic done_mac;

    logic signed [31:0] leaky_x, leaky_y;
    logic done_leaky;
    logic signed [31:0] req_acc;
    logic [15:0] req_scale;
    logic signed [7:0] req_out;
    logic done_req;

    integer pass_count, fail_count;
    integer i, ch;
    logic signed [31:0] acc_feedback;
    logic signed [31:0] bias_val;
    logic signed [7:0] expected_int8;

    reg signed [7:0] w_mem0 [0:MACS_PER_CH-1];
    reg signed [7:0] w_mem1 [0:MACS_PER_CH-1];
    reg signed [7:0] w_mem2 [0:MACS_PER_CH-1];
    reg signed [7:0] w_mem3 [0:MACS_PER_CH-1];
    reg signed [7:0] a_mem [0:MACS_PER_CH-1];
    reg signed [31:0] bias_mem [0:NUM_CH-1];
    reg signed [7:0] exp_mem [0:NUM_CH-1];

    mac_int8 u_mac (
        .clk(clk), .rst_n(rst_n), .valid(valid),
        .weight(weight), .activation(activation), .acc_in(acc_in),
        .acc_out(acc_out), .done(done_mac)
    );
    leaky_relu u_leaky (
        .clk(clk), .rst_n(rst_n), .valid(leaky_x_valid),
        .x(leaky_x), .y(leaky_y), .done(done_leaky)
    );
    requantize #(.SCALE_Q(16)) u_req (
        .clk(clk), .rst_n(rst_n), .valid(req_valid),
        .acc(req_acc), .scale(req_scale),
        .out_int8(req_out), .done(done_req)
    );

    logic leaky_x_valid, req_valid;

    initial begin clk = 0; forever #5 clk = ~clk; end

    initial begin
        $dumpfile("waveform_layer12.vcd");
        $dumpvars(0, layer12_patch_tb);
    end
    initial begin #100000000; $display("TIMEOUT: forcing $finish"); $finish; end

    initial begin
        pass_count = 0;
        fail_count = 0;
        clk = 0;
        rst_n = 0;
        valid = 0;
        weight = 0;
        activation = 0;
        acc_in = 0;
        acc_feedback = 0;
        leaky_x = 0;
        leaky_x_valid = 0;
        req_acc = 0;
        req_scale = 16'd655;
        req_valid = 0;

        $readmemh("image_sim_out/layer12_patch_w0.hex", w_mem0);
        $readmemh("image_sim_out/layer12_patch_w1.hex", w_mem1);
        $readmemh("image_sim_out/layer12_patch_w2.hex", w_mem2);
        $readmemh("image_sim_out/layer12_patch_w3.hex", w_mem3);
        $readmemh("image_sim_out/layer12_patch_a.hex", a_mem);
        $readmemh("image_sim_out/layer12_patch_bias.hex", bias_mem);
        $readmemh("image_sim_out/layer12_patch_expected.hex", exp_mem);

        repeat(3) @(posedge clk);
        rst_n <= 1;
        repeat(1) @(posedge clk);

        $display("=== Layer12 patch TB (one pixel, %0d channels, %0d MACs each) ===", NUM_CH, MACS_PER_CH);

        for (ch = 0; ch < NUM_CH; ch++) begin
            bias_val = bias_mem[ch];
            expected_int8 = exp_mem[ch];
            acc_feedback = 0;

            for (i = 0; i < MACS_PER_CH; i++) begin
                @(posedge clk);
                case (ch)
                    0: weight <= w_mem0[i];
                    1: weight <= w_mem1[i];
                    2: weight <= w_mem2[i];
                    3: weight <= w_mem3[i];
                endcase
                activation <= a_mem[i];
                acc_in <= acc_feedback;
                valid <= 1;
                @(posedge clk);
                valid <= 0;
                @(posedge clk);
                acc_feedback <= acc_out;
            end
            @(posedge clk);

            leaky_x <= acc_feedback + bias_val;
            leaky_x_valid <= 1;
            @(posedge clk);
            leaky_x_valid <= 0;
            @(posedge clk);
            @(posedge clk);

            @(posedge clk);
            req_acc <= leaky_y;
            req_valid <= 1;
            @(posedge clk);
            req_valid <= 0;
            @(posedge clk);
            @(posedge clk);

            if (req_out == expected_int8) begin
                pass_count++;
                $display("[PASS] ch%0d expected=%0d got=%0d", ch, expected_int8, req_out);
            end else begin
                fail_count++;
                $display("[FAIL] ch%0d expected=%0d got=%0d (acc=%0d, leaky=%0d)",
                         ch, expected_int8, req_out, acc_feedback + bias_val, leaky_y);
            end
        end

        $display("TOTAL: %0d PASS, %0d FAIL", pass_count, fail_count);
        $display("RESULT: %s", fail_count ? "SOME FAIL" : "ALL PASS");
        repeat(2) @(posedge clk);
        $finish;
    end
endmodule
