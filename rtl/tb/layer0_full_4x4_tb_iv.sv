// Layer0 full 4x4 TB: first 4x4 output positions x 32 channels vs Python golden.
// Reads layer0_full4x4_*.hex. Generate: run_layer0_full_check.py (or layer0_full_4x4_export.py).
`timescale 1ns/1ps

module layer0_full_4x4_tb;
    timeunit 1ns;
    timeprecision 1ps;

    localparam H4 = 4;
    localparam W4 = 4;
    localparam NUM_CH = 32;
    localparam PAD_SIZE = 243;   // 3*9*9
    localparam EXP_SIZE = 512;    // 4*4*32

    logic clk, rst_n, valid;
    logic signed [7:0]  weight, activation;
    logic signed [31:0] acc_in, acc_out;
    logic signed [31:0] leaky_x, leaky_y;
    logic signed [31:0] req_acc;
    logic [15:0] req_scale;
    logic signed [7:0] req_out;
    logic leaky_x_valid, req_valid;

    integer pass_count, fail_count;
    integer i, ch, oh, ow;
    integer a_index;
    logic signed [31:0] acc_feedback;
    logic signed [31:0] bias_val;
    logic signed [7:0] expected_int8;

    reg signed [7:0] padded_mem [0:PAD_SIZE-1];
    reg signed [7:0] w_flat [0:NUM_CH*27-1];  // w_flat[ch*27+i] = weight ch,i
    reg signed [31:0] bias_mem [0:NUM_CH-1];
    reg signed [7:0] exp_mem [0:EXP_SIZE-1];

    mac_int8 u_mac (
        .clk(clk), .rst_n(rst_n), .valid(valid),
        .weight(weight), .activation(activation), .acc_in(acc_in),
        .acc_out(acc_out), .done()
    );
    leaky_relu u_leaky (
        .clk(clk), .rst_n(rst_n), .valid(leaky_x_valid),
        .x(leaky_x), .y(leaky_y), .done()
    );
    requantize #(.SCALE_Q(16)) u_req (
        .clk(clk), .rst_n(rst_n), .valid(req_valid),
        .acc(req_acc), .scale(req_scale),
        .out_int8(req_out), .done()
    );

    initial begin clk = 0; forever #5 clk = ~clk; end
    initial begin $dumpfile("waveform_4x4.vcd"); $dumpvars(0, layer0_full_4x4_tb); end
    initial begin #60000000; $display("TIMEOUT"); $finish; end

    initial begin
        pass_count = 0;
        fail_count = 0;
        rst_n = 0;
        valid = 0;
        leaky_x_valid = 0;
        req_valid = 0;
        req_scale = 16'd655;

        $readmemh("image_sim_out/layer0_full4x4_padded.hex", padded_mem);
        $readmemh("image_sim_out/layer0_full4x4_weights.hex", w_flat);
        $readmemh("image_sim_out/layer0_full4x4_bias.hex", bias_mem);
        $readmemh("image_sim_out/layer0_full4x4_expected.hex", exp_mem);

        repeat(3) @(posedge clk);
        rst_n <= 1;
        repeat(1) @(posedge clk);

        $display("=== Layer0 full 4x4 TB (4x4 x 32 ch) ===");

        for (oh = 0; oh < H4; oh++) begin
            for (ow = 0; ow < W4; ow++) begin
                for (ch = 0; ch < NUM_CH; ch++) begin
                    bias_val = bias_mem[ch];
                    expected_int8 = exp_mem[(oh * W4 + ow) * NUM_CH + ch];
                    acc_feedback = 0;

                    // 27 MACs: 3 cycles each
                    for (i = 0; i < 27; i++) begin
                        a_index = (i/9)*81 + (oh*2 + (i%9)/3)*9 + (ow*2) + (i%3);
                        @(posedge clk);
                        weight <= w_flat[ch*27 + i];
                        activation <= padded_mem[a_index];
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
                    repeat(2) @(posedge clk);

                    req_acc <= leaky_y;
                    req_valid <= 1;
                    @(posedge clk);
                    req_valid <= 0;
                    repeat(2) @(posedge clk);

                    if (req_out == expected_int8)
                        pass_count++;
                    else begin
                        fail_count++;
                        if (fail_count <= 8)
                            $display("[FAIL] oh=%0d ow=%0d ch=%0d exp=%0d got=%0d", oh, ow, ch, expected_int8, req_out);
                    end
                end
            end
        end

        $display("TOTAL: %0d PASS, %0d FAIL", pass_count, fail_count);
        $display("RESULT: %s", fail_count ? "SOME FAIL" : "ALL PASS");
        repeat(2) @(posedge clk);
        $finish;
    end
endmodule
