// DPU Primitive: Requantize INT32 -> INT8
// Phase 6 - Maps to Python requantize: round(acc*scale) + zp, clamp to [-128,127]
// Simplified: scale is 16-bit fixed point (e.g. 8.8), zero_point = 0 for symmetric
`default_nettype none

module requantize #(
    parameter int SCALE_Q = 16   // scale is SCALE_Q-bit fixed point (fraction bits)
) (
    input  logic        clk,
    input  logic        rst_n,
    input  logic        valid,
    input  logic signed [31:0] acc,
    input  logic [15:0] scale,   // fixed-point scale (e.g. 8.8 format)
    output logic signed [7:0]  out_int8,
    output logic        done
);

    // acc (32b signed) * scale (16b unsigned) -> 48b product; shift right SCALE_Q
    // NOTE: scale is UNSIGNED. We prepend a 0 bit so $signed() sees it as positive.
    // Without this, scales >= 32768 (0x8000) would be misinterpreted as negative.
    logic signed [47:0] product48;
    logic signed [31:0] rounded;
    logic signed [7:0]  clamped;

    assign product48 = acc * $signed({1'b0, scale});
    assign rounded   = product48 >>> SCALE_Q;
    // Clamp to INT8 range
    assign clamped = (rounded > 32'sd127)  ? 8'sd127 :
                     (rounded < -32'sd128) ? -8'sd128 :
                     rounded[7:0];

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            out_int8 <= 8'sd0;
            done     <= 1'b0;
        end else begin
            done     <= valid;
            out_int8 <= valid ? clamped : out_int8;
        end
    end

endmodule
`default_nettype wire
