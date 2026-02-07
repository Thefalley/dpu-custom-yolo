// DPU Subsystem: Small MAC array 2x2 (for incremental build and test)
// Phase 6 - 2 output channels x 2 input channels; one cycle: 2*2 MACs
// Row 0: acc0_out = acc0_in + w00*a0 + w01*a1; Row 1: acc1_out = acc1_in + w10*a0 + w11*a1
`default_nettype none

module mac_array_2x2 (
    input  logic        clk,
    input  logic        rst_n,
    input  logic        valid,
    input  logic signed [7:0]  w00, w01, w10, w11,
    input  logic signed [7:0]  a0,  a1,
    input  logic signed [31:0] acc0_in, acc1_in,
    output logic signed [31:0] acc0_out, acc1_out,
    output logic        done
);

    logic signed [31:0] row0_part0, row0_part1, row1_part0, row1_part1;
    logic signed [31:0] acc0_in_r, acc1_in_r;

    mac_int8 mac_r0_c0 (
        .clk(clk), .rst_n(rst_n), .valid(valid),
        .weight(w00), .activation(a0), .acc_in(32'sd0),
        .acc_out(row0_part0), .done()
    );
    mac_int8 mac_r0_c1 (
        .clk(clk), .rst_n(rst_n), .valid(valid),
        .weight(w01), .activation(a1), .acc_in(32'sd0),
        .acc_out(row0_part1), .done()
    );
    mac_int8 mac_r1_c0 (
        .clk(clk), .rst_n(rst_n), .valid(valid),
        .weight(w10), .activation(a0), .acc_in(32'sd0),
        .acc_out(row1_part0), .done()
    );
    mac_int8 mac_r1_c1 (
        .clk(clk), .rst_n(rst_n), .valid(valid),
        .weight(w11), .activation(a1), .acc_in(32'sd0),
        .acc_out(row1_part1), .done()
    );

    // Latch acc_in when valid; MACs have 1-cycle latency so add next cycle
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            acc0_in_r <= 32'sd0;
            acc1_in_r <= 32'sd0;
            done      <= 1'b0;
        end else begin
            if (valid) begin
                acc0_in_r <= acc0_in;
                acc1_in_r <= acc1_in;
            end
            done <= valid;
        end
    end

    assign acc0_out = acc0_in_r + row0_part0 + row0_part1;
    assign acc1_out = acc1_in_r + row1_part0 + row1_part1;

endmodule
`default_nettype wire
