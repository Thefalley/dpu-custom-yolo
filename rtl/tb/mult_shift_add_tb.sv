// Testbench mult_shift_add - program + clocking block (estilo tb_cb_reg_file_16x32)
// Señales actualizadas en el momento del reloj; sin carreras.
// Para simular: EDA Playground (ModelSim/VCS) o ModelSim/Questa/VCS local.
`timescale 1ns/1ps

program estimulos_mult_shift_add (
    input  bit                clk,
    output bit                 rst_n,
    output bit                 valid,
    output bit signed [7:0]    a,
    output bit signed [7:0]    b,
    input  logic signed [15:0] product,
    input  logic               done
);
    default clocking cb @(posedge clk);
        output rst_n, valid, a, b;
        input  product, done;
    endclocking

    int pass_count, fail_count;

    task run_mult(input logic signed [7:0] a_in, b_in);
        cb.a     <= a_in;
        cb.b     <= b_in;
        cb.valid <= 1;
        ##1 cb.valid <= 0;
        ##2;
        $display("[TB] t=%0t a=%0d b=%0d -> product=%0d", $time, a_in, b_in, cb.product);
    endtask

    task check_mult(input logic signed [15:0] expected);
        if (cb.product == expected) begin
            pass_count++;
            $display("  [PASS] expected %0d", expected);
        end else begin
            fail_count++;
            $display("  [FAIL] got %0d expected %0d", cb.product, expected);
        end
    endtask

    initial begin
        pass_count = 0; fail_count = 0;
        cb.rst_n  <= 0;
        cb.valid  <= 0;
        cb.a      <= 0;
        cb.b      <= 0;
        ##2 cb.rst_n <= 1;
        ##1;

        $display("=== Mult Shift-Add Testbench (clocking block) ===");

        run_mult(8'sd10, 8'sd20);
        check_mult(16'sd200);

        run_mult(8'sd127, 8'sd127);
        check_mult(16'sd16129);

        run_mult(-8'sd5, 8'sd4);
        check_mult(-16'sd20);

        $display("---");
        $display("TOTAL: %0d PASS, %0d FAIL", pass_count, fail_count);
        $display("RESULT: %s", fail_count ? "SOME FAIL" : "ALL PASS");
        $display("=== Mult Shift-Add Testbench done ===");
        ##2 $finish;
    end
endprogram

module mult_shift_add_tb;
    timeunit 1ns;
    timeprecision 1ps;

    bit                clk;
    bit                rst_n;
    bit                valid;
    bit signed [7:0]    a;
    bit signed [7:0]   b;
    logic signed [15:0] product;
    logic               done;

    mult_shift_add dut (
        .clk(clk), .rst_n(rst_n), .valid(valid),
        .a(a), .b(b), .product(product), .done(done)
    );

    estimulos_mult_shift_add put (
        .clk(clk), .rst_n(rst_n), .valid(valid),
        .a(a), .b(b), .product(product), .done(done)
    );

    initial forever #5 clk = ~clk;

    initial begin
        $dumpfile("waveform.vcd");
        $dumpvars(0, mult_shift_add_tb);
    end
endmodule
