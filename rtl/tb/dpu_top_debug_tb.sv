// Targeted debug TB: tests layer 0 only, prints key intermediate values
`timescale 1ns/1ps

module dpu_top_debug_tb;
    timeunit 1ns;
    timeprecision 1ps;

    localparam int H0 = 16;
    localparam int W0 = 16;
    localparam int MAX_CH   = 256;
    localparam int MAX_FMAP = 65536;
    localparam int MAX_WBUF = 147456;
    localparam int FMAP_BASE = MAX_WBUF + MAX_CH*4;

    logic clk, rst_n;
    logic        cmd_valid, cmd_ready;
    logic [2:0]  cmd_type;
    logic [23:0] cmd_addr;
    logic [7:0]  cmd_data;
    logic        rsp_valid;
    logic [7:0]  rsp_data;
    logic        busy, dut_done;
    logic [4:0]  current_layer;

    dpu_top #(
        .H0(H0), .W0(W0),
        .MAX_CH(MAX_CH), .MAX_FMAP(MAX_FMAP), .MAX_WBUF(MAX_WBUF)
    ) u_dut (
        .clk(clk), .rst_n(rst_n),
        .cmd_valid(cmd_valid), .cmd_ready(cmd_ready),
        .cmd_type(cmd_type), .cmd_addr(cmd_addr), .cmd_data(cmd_data),
        .rsp_valid(rsp_valid), .rsp_data(rsp_data),
        .busy(busy), .done(dut_done), .current_layer(current_layer)
    );

    initial begin clk = 0; forever #5 clk = ~clk; end
    initial begin #200000000; $display("TIMEOUT"); $finish; end

    // Monitor engine output pulses
    always @(posedge clk) begin
        if (u_dut.u_engine.out_valid)
            $display("t=%0t [ENG_OUT] ch_base=%0d count=%0d data[0]=%h data[1]=%h data[2]=%h data[3]=%h",
                     $time,
                     u_dut.u_engine.out_ch_base,
                     u_dut.u_engine.out_count,
                     u_dut.u_engine.out_data[0],
                     u_dut.u_engine.out_data[1],
                     u_dut.u_engine.out_data[2],
                     u_dut.u_engine.out_data[3]);
        if (u_dut.u_engine.done)
            $display("t=%0t [ENG_DONE]", $time);
    end

    // Monitor DPU state changes (only CONV states, skip PIO)
    reg [4:0] prev_state;
    always @(posedge clk) begin
        if (u_dut.state != prev_state && u_dut.state >= 4) begin
            $display("t=%0t [DPU] state %0d -> %0d  oh=%0d ow=%0d",
                     $time, prev_state, u_dut.state, u_dut.oh, u_dut.ow);
        end
        prev_state <= u_dut.state;
    end

    // Monitor engine state changes
    reg [3:0] prev_eng_state;
    integer eng_state_count;
    initial eng_state_count = 0;
    always @(posedge clk) begin
        if (u_dut.u_engine.state != prev_eng_state) begin
            eng_state_count = eng_state_count + 1;
            if (eng_state_count <= 100)  // limit output
                $display("t=%0t [ENG] state %0d -> %0d  cout=%0d cin=%0d kpos=%0d ld_r=%0d ld_c=%0d",
                         $time, prev_eng_state, u_dut.u_engine.state,
                         u_dut.u_engine.cout_base, u_dut.u_engine.cin_base,
                         u_dut.u_engine.kpos,
                         u_dut.u_engine.ld_r, u_dut.u_engine.ld_c);
        end
        prev_eng_state <= u_dut.u_engine.state;
    end

    // Monitor MAC array
    always @(posedge clk) begin
        if (u_dut.u_engine.arr_valid)
            $display("t=%0t [MAC] valid! clear=%b act[0]=%h act[1]=%h act[2]=%h w[0]=%h w[1]=%h w[2]=%h",
                     $time, u_dut.u_engine.arr_clear,
                     u_dut.u_engine.act_reg[0],
                     u_dut.u_engine.act_reg[1],
                     u_dut.u_engine.act_reg[2],
                     u_dut.u_engine.w_reg[0],
                     u_dut.u_engine.w_reg[1],
                     u_dut.u_engine.w_reg[2]);
        if (u_dut.u_engine.arr_done)
            $display("t=%0t [MAC_DONE] acc[0]=%h acc[1]=%h",
                     $time,
                     u_dut.u_engine.u_mac_array.acc[0],
                     u_dut.u_engine.u_mac_array.acc[1]);
    end

    // Monitor post-process
    always @(posedge clk) begin
        if (u_dut.u_engine.pp_valid) begin
            $display("t=%0t [PP] valid! acc[0]=%h acc[1]=%h bias[0]=%h bias[1]=%h",
                     $time,
                     u_dut.u_engine.u_mac_array.acc[0],
                     u_dut.u_engine.u_mac_array.acc[1],
                     u_dut.u_engine.bias_reg[0],
                     u_dut.u_engine.bias_reg[1]);
            // Check what the post_process module actually sees at its ports
            $display("t=%0t [PP_PORTS] acc_in[0]=%h acc_in[1]=%h bias[0]=%h scale=%h",
                     $time,
                     u_dut.u_engine.u_post.acc_in[0],
                     u_dut.u_engine.u_post.acc_in[1],
                     u_dut.u_engine.u_post.bias[0],
                     u_dut.u_engine.u_post.scale);
        end
        if (u_dut.u_engine.pp_done) begin
            $display("t=%0t [PP_DONE] result[0]=%h result[1]=%h",
                     $time,
                     u_dut.u_engine.u_post.result[0],
                     u_dut.u_engine.u_post.result[1]);
            $display("t=%0t [PP_INT] biased_r[0]=%h relu_r[0]=%h prod_w[0]=%h rnd_w[0]=%h clamp_w[0]=%h",
                     $time,
                     u_dut.u_engine.u_post.biased_r[0],
                     u_dut.u_engine.u_post.relu_r[0],
                     u_dut.u_engine.u_post.prod_w[0],
                     u_dut.u_engine.u_post.rnd_w[0],
                     u_dut.u_engine.u_post.clamp_w[0]);
        end
    end

    // Monitor batch write
    always @(posedge clk) begin
        if (u_dut.state == 7) begin  // S_CONV_WRITE_BATCH = 7
            $display("t=%0t [WRITE] batch_idx=%0d out_addr=%0d data=%h ping=%b",
                     $time, u_dut.batch_idx, u_dut.out_addr,
                     u_dut.eng_out_data[u_dut.batch_idx], u_dut.ping_pong);
        end
    end

    // Monitor patch buffer loading (just first and last)
    always @(posedge clk) begin
        if (u_dut.state == 4 && (u_dut.load_idx == 0 || u_dut.load_idx == 26)) begin
            $display("t=%0t [PATCH] load_idx=%0d c=%0d ky=%0d kx=%0d",
                     $time, u_dut.load_idx, u_dut.load_c, u_dut.load_ky, u_dut.load_kx);
        end
    end

    task send_cmd(input [2:0] ctype, input [23:0] addr, input [7:0] data);
        @(posedge clk);
        cmd_valid <= 1;
        cmd_type  <= ctype;
        cmd_addr  <= addr;
        cmd_data  <= data;
        @(posedge clk);
        while (!cmd_ready) @(posedge clk);
        cmd_valid <= 0;
        @(posedge clk);
    endtask

    task write_byte(input [23:0] addr, input [7:0] data);
        send_cmd(3'd0, addr, data);
    endtask

    task set_layer(input [4:0] layer_idx);
        send_cmd(3'd3, 24'd0, {3'd0, layer_idx});
    endtask

    task run_layer_cmd();
        send_cmd(3'd1, 24'd0, 8'd0);
        @(posedge clk);
        while (!dut_done) @(posedge clk);
    endtask

    task write_scale(input [15:0] scale_val);
        send_cmd(3'd5, 24'd0, scale_val[7:0]);
        send_cmd(3'd5, 24'd1, scale_val[15:8]);
    endtask

    task read_byte(input [23:0] addr, output [7:0] data);
        @(posedge clk);
        cmd_valid <= 1;
        cmd_type  <= 3'd2;
        cmd_addr  <= addr;
        cmd_data  <= 8'd0;
        @(posedge clk);
        while (!cmd_ready) @(posedge clk);
        cmd_valid <= 0;
        @(posedge clk);
        while (!rsp_valid) @(posedge clk);
        data = rsp_data;
        @(posedge clk);
    endtask

    reg [7:0] hex_buf [0:MAX_WBUF-1];
    reg [7:0] exp_buf [0:MAX_FMAP-1];

    integer j, errs;
    reg [7:0] got;

    initial begin
        cmd_valid = 0; cmd_type = 0; cmd_addr = 0; cmd_data = 0;
        rst_n = 0;
        #100;
        rst_n = 1;
        #20;

        $display("=== Debug TB: Layer 0 only ===");

        // Load input image
        $readmemh("image_sim_out/dpu_top/input_image.hex", hex_buf);
        for (j = 0; j < 768; j = j + 1)
            write_byte(FMAP_BASE + j, hex_buf[j]);

        // Verify first few fmap_a values
        $display("[DBG] fmap_a[0]=%h fmap_a[1]=%h fmap_a[2]=%h",
                 u_dut.fmap_a[0], u_dut.fmap_a[1], u_dut.fmap_a[2]);

        // Set scale
        write_scale(16'h028F);
        $display("[DBG] scale_reg=%h", u_dut.scale_reg);

        // Load weights for layer 0
        $readmemh("image_sim_out/dpu_top/layer0_weights.hex", hex_buf);
        for (j = 0; j < 864; j = j + 1)
            write_byte(j, hex_buf[j]);

        // Verify first few weights
        $display("[DBG] weight_buf[0]=%h [1]=%h [2]=%h",
                 u_dut.weight_buf[0], u_dut.weight_buf[1], u_dut.weight_buf[2]);

        // Load biases for layer 0
        $readmemh("image_sim_out/dpu_top/layer0_bias.hex", hex_buf);
        for (j = 0; j < 32 * 4; j = j + 1)
            write_byte(MAX_WBUF + j, hex_buf[j]);

        // Verify first bias
        $display("[DBG] bias_buf[0]=%h bias_buf[1]=%h",
                 u_dut.bias_buf[0], u_dut.bias_buf[1]);

        $display("[RUN] Layer 0 (c_in=3, c_out=32, 3x3, stride=2)");
        $display("[DBG] ping_pong=%b", u_dut.ping_pong);
        set_layer(5'd0);
        run_layer_cmd();
        $display("[DONE] Layer 0 finished at t=%0t", $time);
        $display("[DBG] ping_pong=%b", u_dut.ping_pong);

        // Check first 8 bytes
        $readmemh("image_sim_out/dpu_top/layer0_expected.hex", exp_buf);
        errs = 0;
        for (j = 0; j < 8; j = j + 1) begin
            read_byte(j[23:0], got);
            if (got !== exp_buf[j]) begin
                $display("  MISMATCH idx %0d: got=%02x exp=%02x", j, got, exp_buf[j]);
                errs = errs + 1;
            end else begin
                $display("  MATCH idx %0d: got=%02x", j, got);
            end
        end

        // Also directly read fmap_b to verify
        $display("[DBG] Direct fmap_b[0]=%h fmap_b[64]=%h fmap_b[128]=%h",
                 u_dut.fmap_b[0], u_dut.fmap_b[64], u_dut.fmap_b[128]);
        $display("[DBG] Direct fmap_a[0]=%h fmap_a[64]=%h",
                 u_dut.fmap_a[0], u_dut.fmap_a[64]);

        if (errs == 0)
            $display("RESULT: FIRST 8 BYTES PASS");
        else
            $display("RESULT: %0d MISMATCHES in first 8 bytes", errs);

        $finish;
    end

endmodule
