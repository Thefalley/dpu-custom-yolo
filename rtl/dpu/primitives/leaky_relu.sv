// DPU Primitive: LeakyReLU (hardware approximation alpha = 3/32 = 0.09375)
// Phase 6 - Maps to Python leaky_relu_hardware(x): x>0 ? x : (x>>3)-(x>>5)
// INT32 in -> INT32 out
`default_nettype none

module leaky_relu (
    input  logic        clk,
    input  logic        rst_n,
    input  logic        valid,
    input  logic signed [31:0] x,
    output logic signed [31:0] y,
    output logic        done
);

    logic signed [31:0] x_shifted;  // (x>>>3) - (x>>>5) = x * 3/32
    logic signed [31:0] y_comb;

    assign x_shifted = (x >>> 3) - (x >>> 5);
    assign y_comb    = (x[31] == 1'b0) ? x : x_shifted;  // x>=0 -> x, else x*3/32

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            y    <= 32'sd0;
            done <= 1'b0;
        end else begin
            done <= valid;
            y    <= valid ? y_comb : y;
        end
    end

endmodule
`default_nettype wire
