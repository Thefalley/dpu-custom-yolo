// Testbench MAC INT8 - program + clocking block (estilo tb_cb_reg_file_16x32)
// Las señales se actualizan en el momento del reloj; sin carreras. Valores estables antes del ciclo.
// Para simular: EDA Playground (ModelSim/VCS) o ModelSim/Questa/VCS local. Icarus NO soporta clocking.
`timescale 1ns/1ps

program estimulos_mac_int8 (
    input  bit                clk,
    output bit                 rst_n,
    output bit                 valid,
    output bit signed [7:0]    weight,
    output bit signed [7:0]    activation,
    output bit signed [31:0]   acc_in,
    input  logic signed [31:0] acc_out,
    input  logic               done
);
    default clocking cb @(posedge clk);
        output rst_n, valid, weight, activation, acc_in;
        input  acc_out, done;
    endclocking

    int pass_count, fail_count;

    task run_mac(input logic signed [7:0] w, a, input logic signed [31:0] acc);
        cb.weight     <= w;
        cb.activation <= a;
        cb.acc_in     <= acc;
        cb.valid      <= 1;
        ##1 cb.valid   <= 0;
        ##1;
        $display("[TB] t=%0t W=%0d A=%0d acc_in=%0d -> acc_out=%0d", $time, w, a, acc, cb.acc_out);
    endtask

    task check_mac(input logic signed [31:0] expected);
        if (cb.acc_out == expected) begin
            pass_count++;
            $display("  [PASS] expected %0d", expected);
        end else begin
            fail_count++;
            $display("  [FAIL] got %0d expected %0d", cb.acc_out, expected);
        end
    endtask

    initial begin
        pass_count = 0; fail_count = 0;
        cb.rst_n  <= 0;
        cb.valid  <= 0;
        cb.weight <= 0;
        cb.activation <= 0;
        cb.acc_in <= 0;
        ##2 cb.rst_n <= 1;
        ##1;

        $display("=== MAC INT8 Testbench (clocking block) ===");

        run_mac(8'sd10, 8'sd20, 32'sd0);
        check_mac(32'sd200);

        run_mac(-8'sd10, 8'sd20, 32'sd0);
        check_mac(-32'sd200);

        run_mac(8'sd127, 8'sd127, 32'sd0);
        check_mac(32'sd16129);

        run_mac(8'sd50, 8'sd50, 32'sd1000);
        check_mac(32'sd3500);

        $display("---");
        $display("TOTAL: %0d PASS, %0d FAIL", pass_count, fail_count);
        $display("RESULT: %s", fail_count ? "SOME FAIL" : "ALL PASS");
        $display("=== MAC INT8 Testbench done ===");
        ##2 $finish;
    end
endprogram

module mac_int8_tb;
    timeunit 1ns;
    timeprecision 1ps;

    bit                clk;
    bit                rst_n;
    bit                valid;
    bit signed [7:0]    weight;
    bit signed [7:0]   activation;
    bit signed [31:0]   acc_in;
    logic signed [31:0] acc_out;
    logic               done;

    mac_int8 dut (
        .clk(clk), .rst_n(rst_n), .valid(valid),
        .weight(weight), .activation(activation), .acc_in(acc_in),
        .acc_out(acc_out), .done(done)
    );

    estimulos_mac_int8 put (
        .clk(clk), .rst_n(rst_n), .valid(valid),
        .weight(weight), .activation(activation), .acc_in(acc_in),
        .acc_out(acc_out), .done(done)
    );

    initial forever #5 clk = ~clk;

    initial begin
        $dumpfile("waveform.vcd");
        $dumpvars(0, mac_int8_tb);
    end
endmodule
