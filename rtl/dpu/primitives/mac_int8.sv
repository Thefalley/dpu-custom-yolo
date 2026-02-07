// DPU Primitive: MAC INT8 (Multiply-Accumulate)
// Phase 6 - Maps to Python mac(weight, activation, accumulator)
// INT8 x INT8 -> product extended to 32b, then ACC += product
`default_nettype none

module mac_int8 (
    input  logic        clk,
    input  logic        rst_n,
    input  logic        valid,
    input  logic signed [7:0]  weight,
    input  logic signed [7:0]  activation,
    input  logic signed [31:0] acc_in,
    output logic signed [31:0] acc_out,
    output logic        done
);

    // INT8 * INT8 = INT16; sign-extend to 32b for accumulation (force signed for iverilog)
    logic signed [15:0] product;
    logic signed [31:0] product_sext;
    logic signed [31:0] sum;

    assign product      = 16'($signed(weight) * $signed(activation));
    assign product_sext = 32'($signed(product));
    assign sum          = acc_in + product_sext;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            acc_out <= 32'sd0;
            done    <= 1'b0;
        end else begin
            done    <= valid;
            acc_out <= valid ? sum : acc_out;
        end
    end

endmodule
`default_nettype wire
