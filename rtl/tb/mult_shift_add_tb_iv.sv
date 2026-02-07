// Testbench mult_shift_add - compatible con Icarus (sin program/clocking)
`timescale 1ns/1ps

module mult_shift_add_tb;
    timeunit 1ns;
    timeprecision 1ps;

    logic clk, rst_n, valid;
    logic signed [7:0] a, b;
    logic signed [15:0] product;
    logic done;
    integer pass_count, fail_count;

    mult_shift_add dut (.clk(clk), .rst_n(rst_n), .valid(valid), .a(a), .b(b), .product(product), .done(done));

    initial forever #5 clk = ~clk;
    initial begin $dumpfile("waveform.vcd"); $dumpvars(0, mult_shift_add_tb); end

    task run_mult(input logic signed [7:0] a_in, b_in);
        @(posedge clk); a <= a_in; b <= b_in; valid <= 1;
        @(posedge clk); valid <= 0;
        repeat(2) @(posedge clk);
        $display("[TB] t=%0t a=%0d b=%0d -> product=%0d", $time, a_in, b_in, product);
    endtask
    task check_mult(input logic signed [15:0] expected);
        if (product == expected) begin pass_count++; $display("  [PASS] expected %0d", expected); end
        else begin fail_count++; $display("  [FAIL] got %0d expected %0d", product, expected); end
    endtask

    initial begin
        pass_count = 0; fail_count = 0;
        rst_n = 0; valid = 0; a = 0; b = 0;
        repeat(2) @(posedge clk); rst_n <= 1; repeat(1) @(posedge clk);
        $display("=== Mult Shift-Add (Icarus-compatible) ===");
        run_mult(8'sd10, 8'sd20);   check_mult(16'sd200);
        run_mult(8'sd127, 8'sd127); check_mult(16'sd16129);
        run_mult(-8'sd5, 8'sd4);   check_mult(-16'sd20);
        $display("TOTAL: %0d PASS, %0d FAIL", pass_count, fail_count);
        $display("RESULT: %s", fail_count ? "SOME FAIL" : "ALL PASS");
        repeat(2) @(posedge clk); $finish;
    end
endmodule
