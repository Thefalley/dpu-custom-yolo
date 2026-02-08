// Structured TB 4x4 for dpu_layer0_top: command interface, quick validation.
// Same tasks as full TB; uses 4x4 region (243-byte padded patch, 512 output bytes).
// Compares RTL output with layer0_full4x4_expected.hex.
`timescale 1ns/1ps

module dpu_layer0_cmd_tb_4x4;
    timeunit 1ns;
    timeprecision 1ps;

    localparam H_OUT    = 4;
    localparam W_OUT    = 4;
    localparam NUM_CH   = 32;
    localparam PAD_H    = 9;
    localparam PAD_W    = 9;
    localparam PAD_SIZE = 3 * PAD_H * PAD_W;   // 243
    localparam W_SIZE   = NUM_CH * 27;
    localparam B_SIZE   = NUM_CH * 4;
    localparam OUT_SIZE = H_OUT * W_OUT * NUM_CH;  // 512
    localparam WEIGHT_BASE = PAD_SIZE;
    localparam BIAS_BASE   = PAD_SIZE + W_SIZE;
    localparam OUTPUT_BASE = PAD_SIZE + W_SIZE + B_SIZE;

    logic clk, rst_n;
    logic        cmd_valid, cmd_ready;
    logic [1:0]  cmd_type;
    logic [23:0] cmd_addr;
    logic [7:0]  cmd_data;
    logic        rsp_valid;
    logic [7:0]  rsp_data;
    logic        busy, done;

    reg signed [7:0]  tb_padded [0:PAD_SIZE-1];
    reg signed [7:0]  tb_weights [0:W_SIZE-1];
    reg signed [31:0] tb_bias [0:NUM_CH-1];
    reg signed [7:0]  tb_expected [0:OUT_SIZE-1];

    dpu_layer0_top #(
        .H_OUT(H_OUT), .W_OUT(W_OUT), .NUM_CH(NUM_CH),
        .PAD_H(PAD_H), .PAD_W(PAD_W)
    ) u_dut (
        .clk(clk), .rst_n(rst_n),
        .cmd_valid(cmd_valid), .cmd_ready(cmd_ready),
        .cmd_type(cmd_type), .cmd_addr(cmd_addr), .cmd_data(cmd_data),
        .rsp_valid(rsp_valid), .rsp_data(rsp_data),
        .busy(busy), .done(done)
    );

    initial begin clk = 0; forever #5 clk = ~clk; end
    initial begin #500000000; $display("TIMEOUT 4x4"); $finish; end

    task send_write_byte(input [23:0] addr, input [7:0] data);
        cmd_valid <= 1;
        cmd_type  <= 2'd0;
        cmd_addr  <= addr;
        cmd_data  <= data;
        @(posedge clk);
        while (!cmd_ready) @(posedge clk);
        cmd_valid <= 0;
        @(posedge clk);
    endtask

    task send_run();
        cmd_valid <= 1;
        cmd_type  <= 2'd1;
        cmd_addr  <= 24'd0;
        cmd_data  <= 8'd0;
        @(posedge clk);
        while (!cmd_ready) @(posedge clk);
        cmd_valid <= 0;
        @(posedge clk);
    endtask

    task read_byte(input [23:0] addr, output [7:0] data);
        cmd_valid <= 1;
        cmd_type  <= 2'd2;
        cmd_addr  <= addr;
        cmd_data  <= 8'd0;
        @(posedge clk);
        while (!cmd_ready) @(posedge clk);
        cmd_valid <= 0;
        @(posedge clk);
        data = rsp_data;
    endtask

    task LoadImage();
        integer i;
        $display("[TB 4x4] LoadImage: %0d bytes", PAD_SIZE);
        for (i = 0; i < PAD_SIZE; i++)
            send_write_byte(i[23:0], tb_padded[i][7:0]);
        $display("[TB 4x4] LoadImage done");
    endtask

    task LoadWeights();
        integer i;
        reg [23:0] addr;
        $display("[TB 4x4] LoadWeights: %0d bytes", W_SIZE);
        for (i = 0; i < W_SIZE; i++) begin
            addr = WEIGHT_BASE + i;
            send_write_byte(addr, tb_weights[i][7:0]);
        end
        $display("[TB 4x4] LoadWeights done");
    endtask

    task LoadBias();
        integer ch;
        reg [23:0] addr;
        reg [31:0] w;
        $display("[TB 4x4] LoadBias: %0d channels", NUM_CH);
        for (ch = 0; ch < NUM_CH; ch++) begin
            w = tb_bias[ch];
            addr = BIAS_BASE + ch * 4;
            send_write_byte(addr,   w[7:0]);
            send_write_byte(addr+1, w[15:8]);
            send_write_byte(addr+2, w[23:16]);
            send_write_byte(addr+3, w[31:24]);
        end
        $display("[TB 4x4] LoadBias done");
    endtask

    task RunLayer0();
        $display("[TB 4x4] RunLayer0 ...");
        send_run();
        while (!done) @(posedge clk);
        @(posedge clk);
        $display("[TB 4x4] RunLayer0 done");
    endtask

    task CompareOutput(output integer err_count);
        integer i;
        reg [7:0] got;
        reg [23:0] addr;
        err_count = 0;
        $display("[TB 4x4] CompareOutput: %0d bytes", OUT_SIZE);
        for (i = 0; i < OUT_SIZE; i++) begin
            addr = OUTPUT_BASE + i;
            read_byte(addr, got);
            if (got !== tb_expected[i]) begin
                if (err_count < 5)
                    $display("  [MISMATCH] idx=%0d expected=%0d got=%0d", i, tb_expected[i], got);
                err_count = err_count + 1;
            end
        end
        if (err_count == 0)
            $display("[TB 4x4] CompareOutput: PASS (%0d bytes)", OUT_SIZE);
        else
            $display("[TB 4x4] CompareOutput: FAIL (%0d mismatches)", err_count);
    endtask

    initial begin
        integer errs;
        cmd_valid <= 0;
        cmd_type  <= 0;
        cmd_addr  <= 0;
        cmd_data  <= 0;
        rst_n     <= 0;

        $readmemh("image_sim_out/layer0_full4x4_padded.hex", tb_padded);
        $readmemh("image_sim_out/layer0_full4x4_weights.hex", tb_weights);
        $readmemh("image_sim_out/layer0_full4x4_bias.hex", tb_bias);
        $readmemh("image_sim_out/layer0_full4x4_expected.hex", tb_expected);

        repeat(3) @(posedge clk);
        rst_n <= 1;
        repeat(2) @(posedge clk);

        $display("=== DPU Layer0 Command TB 4x4 (structured) ===");
        LoadImage();
        LoadWeights();
        LoadBias();
        RunLayer0();
        CompareOutput(errs);

        if (errs == 0)
            $display("RESULT: ALL PASS (4x4 command interface)");
        else
            $display("RESULT: FAIL (%0d mismatches)", errs);
        repeat(2) @(posedge clk);
        $finish;
    end
endmodule
