// Structured TB for dpu_layer0_top: command interface only.
// Tasks: LoadImage(), LoadWeights(), LoadBias(), RunLayer0(), ReadOutputToFile().
// Pass image by placing hex files in image_sim_out/ (from Python export).
`timescale 1ns/1ps

module dpu_layer0_cmd_tb;
    timeunit 1ns;
    timeprecision 1ps;

    localparam H_OUT    = 208;
    localparam W_OUT    = 208;
    localparam NUM_CH   = 32;
    localparam PAD_H    = 418;
    localparam PAD_W    = 418;
    localparam PAD_SIZE = 3 * PAD_H * PAD_W;
    localparam W_SIZE   = NUM_CH * 27;
    localparam B_SIZE   = NUM_CH * 4;
    localparam OUT_SIZE = H_OUT * W_OUT * NUM_CH;
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

    // TB local arrays (loaded from hex, then sent to DUT via commands)
    reg signed [7:0]  tb_padded [0:PAD_SIZE-1];
    reg signed [7:0]  tb_weights [0:W_SIZE-1];
    reg signed [31:0] tb_bias [0:NUM_CH-1];

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
    initial begin #50000000000; $display("TIMEOUT"); $finish; end

    // ---- Command interface tasks ----
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

    task wait_done();
        while (!done) @(posedge clk);
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
        data = rsp_data;
        @(posedge clk);
    endtask

    // ---- High-level tasks ----
    task LoadImage();
        integer i;
        $display("[TB] LoadImage: %0d bytes", PAD_SIZE);
        for (i = 0; i < PAD_SIZE; i++) begin
            send_write_byte(i[23:0], tb_padded[i][7:0]);
            if (i % 100000 == 0 && i > 0) $display("  loaded %0d / %0d", i, PAD_SIZE);
        end
        $display("[TB] LoadImage done");
    endtask

    task LoadWeights();
        integer i;
        $display("[TB] LoadWeights: %0d bytes", W_SIZE);
        for (i = 0; i < W_SIZE; i++)
            send_write_byte((WEIGHT_BASE + i)[23:0], tb_weights[i][7:0]);
        $display("[TB] LoadWeights done");
    endtask

    task LoadBias();
        integer ch;
        integer addr;
        reg [31:0] w;
        $display("[TB] LoadBias: %0d channels", NUM_CH);
        for (ch = 0; ch < NUM_CH; ch++) begin
            w = tb_bias[ch];
            addr = BIAS_BASE + ch * 4;
            send_write_byte(addr[23:0],     w[7:0]);
            send_write_byte((addr+1)[23:0], w[15:8]);
            send_write_byte((addr+2)[23:0], w[23:16]);
            send_write_byte((addr+3)[23:0], w[31:24]);
        end
        $display("[TB] LoadBias done");
    endtask

    task RunLayer0();
        $display("[TB] RunLayer0 ...");
        send_run();
        wait_done();
        $display("[TB] RunLayer0 done");
    endtask

    task ReadOutputToFile(input [256*8:1] filename);
        integer i;
        integer f;
        reg [7:0] b;
        $display("[TB] ReadOutputToFile: %0s (%0d bytes)", filename, OUT_SIZE);
        f = $fopen(filename, "w");
        if (f == 0) begin
            $display("ERROR: could not open %0s", filename);
            return;
        end
        for (i = 0; i < OUT_SIZE; i++) begin
            read_byte((OUTPUT_BASE + i)[23:0], b);
            $fwrite(f, "%02x\n", b);
            if (i % 100000 == 0 && i > 0) $display("  read %0d / %0d", i, OUT_SIZE);
        end
        $fclose(f);
        $display("[TB] ReadOutputToFile done");
    endtask

    // ---- Main ----
    initial begin
        integer out_file;
        cmd_valid <= 0;
        cmd_type  <= 0;
        cmd_addr  <= 0;
        cmd_data  <= 0;
        rst_n     <= 0;

        $readmemh("image_sim_out/layer0_full208_padded.hex", tb_padded);
        $readmemh("image_sim_out/layer0_full208_weights.hex", tb_weights);
        $readmemh("image_sim_out/layer0_full208_bias.hex", tb_bias);

        repeat(3) @(posedge clk);
        rst_n <= 1;
        repeat(2) @(posedge clk);

        $display("=== DPU Layer0 Command TB (structured) ===");
        LoadImage();
        LoadWeights();
        LoadBias();
        RunLayer0();
        ReadOutputToFile("image_sim_out/layer0_rtl_output.hex");

        $display("RESULT: DONE. Compare with: python tests/compare_layer0_rtl_ref.py");
        repeat(2) @(posedge clk);
        $finish;
    end
endmodule
