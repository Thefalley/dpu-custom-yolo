// Post-process array: 32 lanes parallel bias + LeakyReLU + Requantize.
// Matches existing primitives (leaky_relu.sv, requantize.sv) bit-exactly.
// Pipeline: 3 registered stages.
//   Stage 1: biased = acc_in + bias
//   Stage 2: relu   = biased >= 0 ? biased : (biased >>> 3)
//   Stage 3: quant  = clamp((relu * scale) >>> SCALE_Q, -128, 127)
//
// NOTE: result output uses a packed flat vector because Icarus Verilog
// does not correctly propagate unpacked array output ports between modules.
`default_nettype none

module post_process_array #(
    parameter int LANES   = 32,
    parameter int SCALE_Q = 16
) (
    input  logic        clk,
    input  logic        rst_n,
    input  logic        valid,
    input  logic signed [31:0] acc_in [0:LANES-1],
    input  logic signed [31:0] bias   [0:LANES-1],
    input  logic [15:0]        scale,
    output wire  [LANES*8-1:0] result_flat,    // packed: [i*8+:8] = result[i]
    output reg                 done
);

    // Pipeline valid shift register
    reg v1, v2, v3;

    // Stage 1 registered outputs
    reg signed [31:0] biased_r [0:LANES-1];

    // Stage 2 registered outputs
    reg signed [31:0] relu_r [0:LANES-1];

    // Stage 3 result registers
    reg signed [7:0] result_int [0:LANES-1];

    // Stage 3: combinational per-lane (using assign, like requantize.sv)
    wire signed [47:0] prod_w  [0:LANES-1];
    wire signed [31:0] rnd_w   [0:LANES-1];
    wire signed [7:0]  clamp_w [0:LANES-1];

    genvar gi;
    generate
        for (gi = 0; gi < LANES; gi = gi + 1) begin : gen_pp
            assign prod_w[gi]  = relu_r[gi] * $signed(scale);
            assign rnd_w[gi]   = prod_w[gi] >>> SCALE_Q;
            assign clamp_w[gi] = (rnd_w[gi] > 32'sd127)  ? 8'sd127  :
                                 (rnd_w[gi] < -32'sd128) ? -8'sd128 :
                                 rnd_w[gi][7:0];
            // Pack result as flat vector
            assign result_flat[gi*8 +: 8] = result_int[gi];
        end
    endgenerate

    integer i;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            v1 <= 1'b0;
            v2 <= 1'b0;
            v3 <= 1'b0;
            done <= 1'b0;
            for (i = 0; i < LANES; i = i + 1) begin
                biased_r[i]   <= 32'sd0;
                relu_r[i]     <= 32'sd0;
                result_int[i] <= 8'sd0;
            end
        end else begin
            // Pipeline valid propagation
            v1 <= valid;
            v2 <= v1;
            v3 <= v2;
            done <= v3;

            // Stage 1: Bias add -> register
            if (valid) begin
                for (i = 0; i < LANES; i = i + 1)
                    biased_r[i] <= acc_in[i] + bias[i];
            end

            // Stage 2: LeakyReLU -> register
            if (v1) begin
                for (i = 0; i < LANES; i = i + 1)
                    relu_r[i] <= (biased_r[i][31] == 1'b0) ? biased_r[i] : (biased_r[i] >>> 3);
            end

            // Stage 3: Requantize (combinational via assign, latch result)
            if (v2) begin
                for (i = 0; i < LANES; i = i + 1)
                    result_int[i] <= clamp_w[i];
            end
        end
    end

endmodule
`default_nettype wire
