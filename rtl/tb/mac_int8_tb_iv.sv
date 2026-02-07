// Testbench MAC INT8 - compatible con Icarus Verilog (sin program/clocking)
// Misma disciplina: drives con <= y @(posedge clk) para reducir carreras.
// Uso: python verilog-sim-py/sv_simulator.py --no-wave rtl/dpu/primitives/mac_int8.sv rtl/tb/mac_int8_tb_iv.sv --top mac_int8_tb
`timescale 1ns/1ps

module mac_int8_tb;
    timeunit 1ns;
    timeprecision 1ps;

    logic                clk, rst_n, valid;
    logic signed [7:0]   weight, activation;
    logic signed [31:0]  acc_in, acc_out;
    logic                done;
    integer pass_count, fail_count;

    mac_int8 dut (
        .clk(clk), .rst_n(rst_n), .valid(valid),
        .weight(weight), .activation(activation), .acc_in(acc_in),
        .acc_out(acc_out), .done(done)
    );

    initial forever #5 clk = ~clk;

    initial begin
        $dumpfile("waveform.vcd");
        $dumpvars(0, mac_int8_tb);
    end

    task run_mac(input logic signed [7:0] w, a, input logic signed [31:0] acc);
        @(posedge clk);
        weight <= w; activation <= a; acc_in <= acc; valid <= 1;
        @(posedge clk);
        valid <= 0;
        @(posedge clk);
        $display("[TB] t=%0t W=%0d A=%0d acc_in=%0d -> acc_out=%0d", $time, w, a, acc, acc_out);
    endtask

    task check_mac(input logic signed [31:0] expected);
        if (acc_out == expected) begin pass_count++; $display("  [PASS] expected %0d", expected); end
        else begin fail_count++; $display("  [FAIL] got %0d expected %0d", acc_out, expected); end
    endtask

    initial begin
        pass_count = 0; fail_count = 0;
        rst_n = 0; valid = 0; weight = 0; activation = 0; acc_in = 0;
        repeat(2) @(posedge clk);
        rst_n <= 1;
        repeat(1) @(posedge clk);

        $display("=== MAC INT8 (Icarus-compatible) ===");
        run_mac(8'sd10, 8'sd20, 32'sd0);   check_mac(32'sd200);
        run_mac(-8'sd10, 8'sd20, 32'sd0);  check_mac(-32'sd200);
        run_mac(8'sd127, 8'sd127, 32'sd0); check_mac(32'sd16129);
        run_mac(8'sd50, 8'sd50, 32'sd1000); check_mac(32'sd3500);

        $display("TOTAL: %0d PASS, %0d FAIL", pass_count, fail_count);
        $display("RESULT: %s", fail_count ? "SOME FAIL" : "ALL PASS");
        repeat(2) @(posedge clk);
        $finish;
    end
endmodule
