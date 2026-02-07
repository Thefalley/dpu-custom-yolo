// DPU Primitive: INT8 x INT8 multiplier using shift-and-add (Phase 4b)
// A x B = sum_i (A * B[i]) << i  for i=0..7
// No DSP; LUT-only for ASIC portability
`default_nettype none

module mult_shift_add (
    input  logic        clk,
    input  logic        rst_n,
    input  logic        valid,
    input  logic signed [7:0]  a,
    input  logic signed [7:0]  b,
    output logic signed [15:0] product,
    output logic        done
);

    // Partial products: a * b[i] = a if b[i] else 0; then shift left by i
    logic signed [15:0] pp0, pp1, pp2, pp3, pp4, pp5, pp6, pp7;
    logic signed [15:0] sum01, sum23, sum45, sum67;
    logic signed [15:0] sum0123, sum4567;
    logic signed [15:0] sum_all;

    assign pp0 = b[0] ? {{8{a[7]}}, a} << 0 : 16'sd0;
    assign pp1 = b[1] ? {{8{a[7]}}, a} << 1 : 16'sd0;
    assign pp2 = b[2] ? {{8{a[7]}}, a} << 2 : 16'sd0;
    assign pp3 = b[3] ? {{8{a[7]}}, a} << 3 : 16'sd0;
    assign pp4 = b[4] ? {{8{a[7]}}, a} << 4 : 16'sd0;
    assign pp5 = b[5] ? {{8{a[7]}}, a} << 5 : 16'sd0;
    assign pp6 = b[6] ? {{8{a[7]}}, a} << 6 : 16'sd0;
    assign pp7 = b[7] ? {{8{a[7]}}, a} << 7 : 16'sd0;

    // Adder tree (combinational)
    assign sum01   = pp0 + pp1;
    assign sum23   = pp2 + pp3;
    assign sum45   = pp4 + pp5;
    assign sum67   = pp6 + pp7;
    assign sum0123 = sum01 + sum23;
    assign sum4567 = sum45 + sum67;
    assign sum_all = sum0123 + sum4567;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            product <= 16'sd0;
            done    <= 1'b0;
        end else begin
            done    <= valid;
            product <= valid ? sum_all : product;
        end
    end

endmodule
`default_nettype wire
