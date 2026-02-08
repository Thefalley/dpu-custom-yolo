// TB for layer0_engine: one channel, 27 MACs, compare with golden.
// Reads layer0_patch_w0.hex, layer0_patch_a.hex, layer0_patch_bias.hex, layer0_patch_expected.hex (ch0).
// Generate: run_layer0_patch_check.py (or layer0_patch_golden.py).
`timescale 1ns/1ps

module layer0_engine_tb;
    timeunit 1ns;
    timeprecision 1ps;

    localparam MACS = 27;

    logic clk, rst_n, start, done;
    logic signed [7:0] act_in, w_in;
    logic signed [31:0] bias;
    logic [15:0] scale;
    logic signed [7:0] result_int8;
    logic [4:0] mac_index;

    reg signed [7:0] a_mem [0:MACS-1];
    reg signed [7:0] w_mem [0:MACS-1];
    reg signed [31:0] bias_mem [0:0];
    reg signed [7:0] exp_mem [0:0];
    logic signed [31:0] bias_val;
    logic signed [7:0] expected;

    integer i;
    logic pass;

    layer0_engine #(.MACS(MACS), .SCALE_Q(16)) u_eng (
        .clk(clk), .rst_n(rst_n), .start(start),
        .act_in(act_in), .w_in(w_in), .bias(bias), .scale(scale),
        .done(done), .result_int8(result_int8), .mac_index(mac_index)
    );

    assign act_in = a_mem[mac_index];
    assign w_in   = w_mem[mac_index];
    assign bias   = bias_val;
    assign scale  = 16'd655;

    initial begin clk = 0; forever #5 clk = ~clk; end
    initial begin $dumpfile("waveform_engine.vcd"); $dumpvars(0, layer0_engine_tb); end
    initial begin #5000000; $display("TIMEOUT"); $finish; end

    initial begin
        rst_n = 0;
        start = 0;
        bias_val = 0;

        $readmemh("image_sim_out/layer0_patch_a.hex", a_mem);
        $readmemh("image_sim_out/layer0_patch_w0.hex", w_mem);
        $readmemh("image_sim_out/layer0_patch_bias.hex", bias_mem);
        $readmemh("image_sim_out/layer0_patch_expected.hex", exp_mem);
        bias_val = bias_mem[0];
        expected = exp_mem[0];

        repeat(3) @(posedge clk);
        rst_n <= 1;
        repeat(2) @(posedge clk);

        $display("=== Layer0 engine TB (one channel, 27 MACs) ===");
        start <= 1;
        @(posedge clk);
        start <= 0;
        while (!done) @(posedge clk);
        pass = (result_int8 == expected);
        if (pass)
            $display("[PASS] expected=%0d got=%0d", expected, result_int8);
        else
            $display("[FAIL] expected=%0d got=%0d", expected, result_int8);
        $display("RESULT: %s", pass ? "ALL PASS" : "SOME FAIL");
        repeat(2) @(posedge clk);
        $finish;
    end
endmodule
