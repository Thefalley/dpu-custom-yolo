// MaxPool 2x2 unit: 4 signed INT8 inputs -> max output.
// 2-level comparison tree, registered output.
`default_nettype none

module maxpool_unit (
    input  logic        clk,
    input  logic        rst_n,
    input  logic        valid,
    input  logic signed [7:0] a, b, c, d,
    output logic signed [7:0] max_out,
    output logic        done
);

    // Level 1: compare pairs
    logic signed [7:0] max_ab, max_cd;
    assign max_ab = (a > b) ? a : b;
    assign max_cd = (c > d) ? c : d;

    // Level 2: compare winners
    logic signed [7:0] max_final;
    assign max_final = (max_ab > max_cd) ? max_ab : max_cd;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            max_out <= 8'sd0;
            done    <= 1'b0;
        end else begin
            done    <= valid;
            max_out <= valid ? max_final : max_out;
        end
    end

endmodule
`default_nettype wire
