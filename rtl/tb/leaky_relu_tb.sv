// Testbench LeakyReLU - program + clocking block (estilo tb_cb_reg_file_16x32)
// Señales actualizadas en el momento del reloj; sin carreras.
// Para simular: EDA Playground (ModelSim/VCS) o ModelSim/Questa/VCS local.
`timescale 1ns/1ps

program estimulos_leaky_relu (
    input  bit                clk,
    output bit                 rst_n,
    output bit                 valid,
    output bit signed [31:0]   x,
    input  logic signed [31:0] y,
    input  logic               done
);
    default clocking cb @(posedge clk);
        output rst_n, valid, x;
        input  y, done;
    endclocking

    int pass_count, fail_count;

    task run_relu(input logic signed [31:0] x_in);
        cb.x     <= x_in;
        cb.valid <= 1;
        ##1 cb.valid <= 0;
        ##2;
        $display("[TB] t=%0t x=%0d -> y=%0d", $time, x_in, cb.y);
    endtask

    task check_relu(input logic signed [31:0] expected);
        if (cb.y == expected) begin
            pass_count++;
            $display("  [PASS] expected %0d", expected);
        end else begin
            fail_count++;
            $display("  [FAIL] got %0d expected %0d", cb.y, expected);
        end
    endtask

    initial begin
        pass_count = 0; fail_count = 0;
        cb.rst_n  <= 0;
        cb.valid  <= 0;
        cb.x      <= 0;
        ##2 cb.rst_n <= 1;
        ##1;

        $display("=== LeakyReLU Testbench (clocking block) ===");

        run_relu(-32'sd80);
        check_relu(-32'sd10);

        run_relu(32'sd40);
        check_relu(32'sd40);

        run_relu(32'sd0);
        check_relu(32'sd0);

        $display("---");
        $display("TOTAL: %0d PASS, %0d FAIL", pass_count, fail_count);
        $display("RESULT: %s", fail_count ? "SOME FAIL" : "ALL PASS");
        $display("=== LeakyReLU Testbench done ===");
        ##2 $finish;
    end
endprogram

module leaky_relu_tb;
    timeunit 1ns;
    timeprecision 1ps;

    bit                clk;
    bit                rst_n;
    bit                valid;
    bit signed [31:0]   x;
    logic signed [31:0] y;
    logic               done;

    leaky_relu dut (
        .clk(clk), .rst_n(rst_n), .valid(valid),
        .x(x), .y(y), .done(done)
    );

    estimulos_leaky_relu put (
        .clk(clk), .rst_n(rst_n), .valid(valid),
        .x(x), .y(y), .done(done)
    );

    initial forever #5 clk = ~clk;

    initial begin
        $dumpfile("waveform.vcd");
        $dumpvars(0, leaky_relu_tb);
    end
endmodule
