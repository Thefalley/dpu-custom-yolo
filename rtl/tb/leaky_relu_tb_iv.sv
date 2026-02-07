// Testbench LeakyReLU - compatible con Icarus (sin program/clocking)
`timescale 1ns/1ps

module leaky_relu_tb;
    timeunit 1ns;
    timeprecision 1ps;

    logic clk, rst_n, valid;
    logic signed [31:0] x, y;
    logic done;
    integer pass_count, fail_count;

    leaky_relu dut (.clk(clk), .rst_n(rst_n), .valid(valid), .x(x), .y(y), .done(done));

    initial begin clk = 0; forever #5 clk = ~clk; end
    initial begin $dumpfile("waveform.vcd"); $dumpvars(0, leaky_relu_tb); end
    initial begin #100000; $display("TIMEOUT: forcing $finish"); $finish; end

    task run_relu(input logic signed [31:0] x_in);
        @(posedge clk); x <= x_in; valid <= 1;
        @(posedge clk); valid <= 0;
        repeat(2) @(posedge clk);
        $display("[TB] t=%0t x=%0d -> y=%0d", $time, x_in, y);
    endtask
    task check_relu(input logic signed [31:0] expected);
        if (y == expected) begin pass_count++; $display("  [PASS] expected %0d", expected); end
        else begin fail_count++; $display("  [FAIL] got %0d expected %0d", y, expected); end
    endtask

    initial begin
        pass_count = 0; fail_count = 0;
        rst_n = 0; valid = 0; x = 0;
        repeat(2) @(posedge clk); rst_n <= 1; repeat(1) @(posedge clk);
        $display("=== LeakyReLU (Icarus-compatible) ===");
        run_relu(-32'sd80); check_relu(-32'sd10);
        run_relu(32'sd40);  check_relu(32'sd40);
        run_relu(32'sd0);   check_relu(32'sd0);
        $display("TOTAL: %0d PASS, %0d FAIL", pass_count, fail_count);
        $display("RESULT: %s", fail_count ? "SOME FAIL" : "ALL PASS");
        repeat(2) @(posedge clk); $finish;
    end
endmodule
