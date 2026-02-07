// Layer0 full 208x208 TB: entire layer 0 output (208x208x32) to file.
// Reads layer0_full208_*.hex. Writes layer0_rtl_output.hex (one byte per line, 1382912 lines).
// Compare with Python: layer0_output_ref.npy
`timescale 1ns/1ps

module layer0_full_208_tb;
    timeunit 1ns;
    timeprecision 1ps;

    localparam H_OUT = 208;
    localparam W_OUT = 208;
    localparam NUM_CH = 32;
    localparam PAD_H = 418;
    localparam PAD_W = 418;
    localparam PAD_SIZE = 3 * PAD_H * PAD_W;  // 523254

    logic clk, rst_n, valid;
    logic signed [7:0]  weight, activation;
    logic signed [31:0] acc_in, acc_out;
    logic signed [31:0] leaky_x, leaky_y;
    logic signed [31:0] req_acc;
    logic [15:0] req_scale;
    logic signed [7:0] req_out;
    logic leaky_x_valid, req_valid;

    integer i, ch, oh, ow;
    integer a_index;
    logic signed [31:0] acc_feedback;
    logic signed [31:0] bias_val;
    integer out_file;
    integer pixels_done;

    reg signed [7:0] padded_mem [0:PAD_SIZE-1];
    reg signed [7:0] w_flat [0:NUM_CH*27-1];
    reg signed [31:0] bias_mem [0:NUM_CH-1];

    mac_int8 u_mac (
        .clk(clk), .rst_n(rst_n), .valid(valid),
        .weight(weight), .activation(activation), .acc_in(acc_in),
        .acc_out(acc_out), .done()
    );
    leaky_relu u_leaky (
        .clk(clk), .rst_n(rst_n), .valid(leaky_x_valid),
        .x(leaky_x), .y(leaky_y), .done()
    );
    requantize #(.SCALE_Q(16)) u_req (
        .clk(clk), .rst_n(rst_n), .valid(req_valid),
        .acc(req_acc), .scale(req_scale),
        .out_int8(req_out), .done()
    );

    initial begin clk = 0; forever #5 clk = ~clk; end
    initial begin #50000000000; $display("TIMEOUT full 208"); $finish; end

    initial begin
        rst_n = 0;
        valid = 0;
        leaky_x_valid = 0;
        req_valid = 0;
        req_scale = 16'd655;
        pixels_done = 0;

        $readmemh("image_sim_out/layer0_full208_padded.hex", padded_mem);
        $readmemh("image_sim_out/layer0_full208_weights.hex", w_flat);
        $readmemh("image_sim_out/layer0_full208_bias.hex", bias_mem);

        out_file = $fopen("image_sim_out/layer0_rtl_output.hex", "w");
        if (out_file == 0) begin
            $display("ERROR: could not open image_sim_out/layer0_rtl_output.hex for write");
            $finish;
        end

        repeat(3) @(posedge clk);
        rst_n <= 1;
        repeat(1) @(posedge clk);

        $display("=== Layer0 full 208x208 TB (writing to layer0_rtl_output.hex) ===");

        for (oh = 0; oh < H_OUT; oh++) begin
            for (ow = 0; ow < W_OUT; ow++) begin
                for (ch = 0; ch < NUM_CH; ch++) begin
                    bias_val = bias_mem[ch];
                    acc_feedback = 0;

                    for (i = 0; i < 27; i++) begin
                        a_index = (i/9)*PAD_H*PAD_W + (oh*2 + (i%9)/3)*PAD_W + (ow*2) + (i%3);
                        @(posedge clk);
                        weight <= w_flat[ch*27 + i];
                        activation <= padded_mem[a_index];
                        acc_in <= acc_feedback;
                        valid <= 1;
                        @(posedge clk);
                        valid <= 0;
                        @(posedge clk);
                        acc_feedback <= acc_out;
                    end
                    @(posedge clk);

                    leaky_x <= acc_feedback + bias_val;
                    leaky_x_valid <= 1;
                    @(posedge clk);
                    leaky_x_valid <= 0;
                    repeat(2) @(posedge clk);

                    req_acc <= leaky_y;
                    req_valid <= 1;
                    @(posedge clk);
                    req_valid <= 0;
                    repeat(2) @(posedge clk);

                    $fwrite(out_file, "%02x\n", req_out[7:0]);
                    pixels_done = pixels_done + 1;
                    if (pixels_done % 50000 == 0)
                        $display("  pixels written: %0d / 1382912", pixels_done);
                end
            end
        end

        $fclose(out_file);
        $display("TOTAL: %0d pixels written to image_sim_out/layer0_rtl_output.hex", pixels_done);
        $display("RESULT: DONE (compare with layer0_output_ref.npy in Python)");
        repeat(2) @(posedge clk);
        $finish;
    end
endmodule
