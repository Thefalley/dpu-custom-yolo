// MAC Array 32x32: Behavioral model of 1024 MACs.
// Output-stationary: 32 rows = 32 output channels, 32 cols = 32 input channels.
// Each cycle: acc[r] += sum(w[r][c] * act[c]) for c=0..31
// Arithmetic matches mac_int8: INT8*INT8->INT16, sign-extend to 32b, accumulate.
//
// NOTE: acc output uses a packed flat vector (1024 bits) because Icarus Verilog
// does not correctly propagate unpacked array output ports between modules.
`default_nettype none

module mac_array_32x32 (
    input  logic        clk,
    input  logic        rst_n,
    input  logic        valid,           // pulse: 1 MAC op per cycle
    input  logic        clear_acc,       // clear accumulators (start of Cout tile)
    input  logic signed [7:0]  act_in  [0:31],      // 32 activations (broadcast)
    input  logic signed [7:0]  w_in    [0:1023],     // weights [row*32+col], flattened
    output wire  [1023:0]      acc_out_flat,         // packed: [r*32+:32] = acc[r]
    output logic        done                         // 1 cycle after valid
);

    reg signed [31:0] acc [0:31];
    integer r, c;
    logic signed [31:0] partial;
    logic signed [15:0] prod16;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (r = 0; r < 32; r = r + 1)
                acc[r] <= 32'sd0;
            done <= 1'b0;
        end else begin
            done <= valid;
            if (valid) begin
                for (r = 0; r < 32; r = r + 1) begin
                    partial = 32'sd0;
                    for (c = 0; c < 32; c = c + 1) begin
                        prod16 = w_in[r * 32 + c] * act_in[c];
                        partial = partial + {{16{prod16[15]}}, prod16};
                    end
                    if (clear_acc)
                        acc[r] <= partial;
                    else
                        acc[r] <= acc[r] + partial;
                end
            end
        end
    end

    // Pack accumulators as flat vector (combinational, zero delay)
    genvar gi;
    generate
        for (gi = 0; gi < 32; gi = gi + 1) begin : gen_pack
            assign acc_out_flat[gi*32 +: 32] = acc[gi];
        end
    endgenerate

endmodule
`default_nettype wire
