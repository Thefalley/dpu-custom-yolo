// Conv engine with 32x32 MAC array: processes one pixel, ALL output channels.
// Tiling: iterates Cout tiles (32 at a time), within each:
//   iterates kernel_pos x Cin tiles, accumulating in the array.
// After all MACs for a Cout tile, post-processes 32 channels and outputs them.
//
// Weight layout (cin-contiguous per kernel position for wide reads):
//   weight_buf[co * macs_per_ch + kpos * c_in + c]
//   where macs_per_ch = c_in * K * K
//
// Patch layout (cin-contiguous per kernel position):
//   patch_buf[kpos * c_in + c]   for both 3x3 and 1x1
//
// WIDE MEMORY INTERFACE: reads 32 bytes per cycle from weight and patch buffers.
// Weight: loads one row (32 input channels) per cycle -> 32 cycles per 32x32 tile.
// Patch: loads all 32 activations in 1 cycle.
//
// NOTE: Module interfaces use packed flat vectors for array outputs because
// Icarus Verilog does not correctly propagate unpacked array output ports.
`default_nettype none

module conv_engine_array #(
    parameter int ROWS    = 32,
    parameter int COLS    = 32,
    parameter int SCALE_Q = 16
) (
    input  logic        clk,
    input  logic        rst_n,
    input  logic        start,
    // Layer config
    input  logic [10:0] c_in,
    input  logic [10:0] c_out,
    input  logic [3:0]  kernel_size,     // 1 or 3
    // Patch buffer read (activations) — 32-byte wide
    output logic [12:0] patch_rd_addr,
    input  logic [255:0] patch_rd_data_wide,  // 32 bytes packed
    // Weight buffer read — 32-byte wide
    output logic [21:0] weight_rd_addr,
    input  logic [255:0] weight_rd_data_wide, // 32 bytes packed
    // Bias buffer read
    output logic [8:0]  bias_rd_ch,
    input  logic signed [31:0] bias_rd_data,
    // Scale
    input  logic [15:0] scale,
    // Activation mode
    input  logic        skip_relu,     // 1 = LINEAR (bypass LeakyReLU)
    // Output: up to 32 results per batch (packed flat vector)
    output logic        out_valid,
    output logic [8:0]  out_ch_base,
    output logic [5:0]  out_count,       // valid channels (1..32)
    output wire  [255:0] out_data_flat,  // packed: [i*8+:8] = out_data[i]
    output logic        done
);

    // =========================================================================
    // Internal registers for array inputs
    // =========================================================================
    reg signed [7:0] act_reg  [0:31];
    reg signed [7:0] w_reg    [0:1023];   // flattened [row*32+col]
    reg signed [31:0] bias_reg [0:31];

    // =========================================================================
    // MAC array instance (packed output)
    // =========================================================================
    logic        arr_valid, arr_clear, arr_done;
    wire [1023:0] acc_flat;

    mac_array_32x32 u_mac_array (
        .clk(clk), .rst_n(rst_n),
        .valid(arr_valid), .clear_acc(arr_clear),
        .act_in(act_reg), .w_in(w_reg),
        .acc_out_flat(acc_flat), .done(arr_done)
    );

    // Unpack MAC accumulator into local regs for post-process
    // (Icarus needs reg arrays driven from local context for input ports)
    reg signed [31:0] arr_acc [0:31];

    // =========================================================================
    // Post-process instance (packed result output)
    // =========================================================================
    logic        pp_valid, pp_done;
    wire [255:0] pp_result_flat;

    post_process_array #(.LANES(32), .SCALE_Q(SCALE_Q)) u_post (
        .clk(clk), .rst_n(rst_n),
        .valid(pp_valid),
        .acc_in(arr_acc), .bias(bias_reg), .scale(scale),
        .skip_relu(skip_relu),
        .result_flat(pp_result_flat), .done(pp_done)
    );

    // Unpack post-process result into local regs
    reg signed [7:0] pp_result [0:31];

    // Output data registers (packed for dpu_top)
    reg signed [7:0] out_data_int [0:31];

    // Pack output data as flat vector
    genvar gk;
    generate
        for (gk = 0; gk < 32; gk = gk + 1) begin : gen_out_pack
            assign out_data_flat[gk*8 +: 8] = out_data_int[gk];
        end
    endgenerate

    // =========================================================================
    // Derived parameters (latched on INIT)
    // =========================================================================
    logic [10:0] k_sq;           // 1 or 9
    logic [20:0] macs_per_ch;    // c_in * k_sq (up to 256*9=2304)

    // =========================================================================
    // FSM
    // =========================================================================
    typedef enum logic [3:0] {
        S_IDLE,
        S_INIT,
        S_BIAS_REQ,
        S_BIAS_LATCH,
        S_WLOAD_ROW,
        S_WLOAD_LATCH,
        S_ALOAD,
        S_ALOAD_LATCH,
        S_MAC_FIRE,
        S_MAC_WAIT,
        S_POST_PROCESS,
        S_PP_WAIT,
        S_OUTPUT,
        S_DONE
    } state_t;
    state_t state;

    // Tiling counters
    integer cout_base;
    integer cin_base;
    integer kpos;
    logic   first_mac;           // first MAC in this Cout tile

    // Loading counters
    integer ld_r;               // weight load row index
    integer ld_b;               // bias load index

    // Active dimensions
    integer active_rows;        // min(32, c_out - cout_base)
    integer cin_actual;         // min(32, c_in - cin_base)

    integer ii, jj;             // loop vars for zero-init

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state       <= S_IDLE;
            arr_valid   <= 1'b0;
            arr_clear   <= 1'b0;
            pp_valid    <= 1'b0;
            out_valid   <= 1'b0;
            done        <= 1'b0;
            patch_rd_addr  <= 13'd0;
            weight_rd_addr <= 22'd0;
            bias_rd_ch     <= 9'd0;
            out_ch_base    <= 9'd0;
            out_count      <= 6'd0;
            cout_base <= 0; cin_base <= 0; kpos <= 0;
            first_mac <= 1'b0;
            ld_r <= 0; ld_b <= 0;
            for (ii = 0; ii < 32; ii = ii + 1) begin
                arr_acc[ii]      <= 32'sd0;
                pp_result[ii]    <= 8'sd0;
                out_data_int[ii] <= 8'sd0;
            end
        end else begin
            arr_valid <= 1'b0;
            pp_valid  <= 1'b0;
            out_valid <= 1'b0;
            done      <= 1'b0;

            case (state)

                S_IDLE: begin
                    if (start) state <= S_INIT;
                end

                // =============================================================
                S_INIT: begin
                    k_sq        <= (kernel_size == 4'd3) ? 11'd9 : 11'd1;
                    macs_per_ch <= (kernel_size == 4'd3) ? (c_in * 21'd9) : {10'd0, c_in};
                    cout_base   <= 0;
                    // Zero all internal regs
                    for (ii = 0; ii < 32; ii = ii + 1) begin
                        act_reg[ii]  <= 8'sd0;
                        bias_reg[ii] <= 32'sd0;
                        for (jj = 0; jj < 32; jj = jj + 1)
                            w_reg[ii*32+jj] <= 8'sd0;
                    end
                    ld_b  <= 0;
                    state <= S_BIAS_REQ;
                end

                // =============================================================
                // BIAS loading: 2-phase (request / latch) per channel
                // Bias is 32-bit per channel, small count, keep sequential.
                // =============================================================
                S_BIAS_REQ: begin
                    active_rows = (cout_base + 32 <= c_out) ? 32 : (c_out - cout_base);
                    if (ld_b < active_rows) begin
                        bias_rd_ch <= (cout_base + ld_b);
                        state      <= S_BIAS_LATCH;
                    end else begin
                        // Zero remaining
                        for (ii = active_rows; ii < 32; ii = ii + 1)
                            bias_reg[ii] <= 32'sd0;
                        // Start MAC tiling
                        kpos      <= 0;
                        cin_base  <= 0;
                        first_mac <= 1'b1;
                        ld_r      <= 0;
                        // Zero w/a regs
                        for (ii = 0; ii < 32; ii = ii + 1) begin
                            act_reg[ii] <= 8'sd0;
                            for (jj = 0; jj < 32; jj = jj + 1)
                                w_reg[ii*32+jj] <= 8'sd0;
                        end
                        state <= S_WLOAD_ROW;
                    end
                end

                S_BIAS_LATCH: begin
                    bias_reg[ld_b] <= bias_rd_data;
                    ld_b  <= ld_b + 1;
                    state <= S_BIAS_REQ;
                end

                // =============================================================
                // WEIGHT loading: 32 bytes per cycle (one row of the weight tile).
                // Issues address, waits 1 cycle for data, then latches 32 bytes.
                // =============================================================
                S_WLOAD_ROW: begin
                    active_rows = (cout_base + 32 <= c_out) ? 32 : (c_out - cout_base);
                    if (ld_r < active_rows) begin
                        // Wide read: 32 consecutive cin bytes for this row's (cout, kpos)
                        weight_rd_addr <= (cout_base + ld_r) * macs_per_ch +
                                         kpos * c_in + cin_base;
                        state <= S_WLOAD_LATCH;
                    end else begin
                        // All rows done, load activations
                        state <= S_ALOAD;
                    end
                end

                S_WLOAD_LATCH: begin
                    cin_actual = (cin_base + 32 <= c_in) ? 32 : (c_in - cin_base);
                    // Latch 32 bytes from wide read into w_reg[ld_r*32 + 0..31]
                    for (jj = 0; jj < 32; jj = jj + 1) begin
                        if (jj < cin_actual)
                            w_reg[ld_r*32+jj] <= $signed(weight_rd_data_wide[jj*8 +: 8]);
                        else
                            w_reg[ld_r*32+jj] <= 8'sd0;
                    end
                    ld_r  <= ld_r + 1;
                    state <= S_WLOAD_ROW;
                end

                // =============================================================
                // ACTIVATION loading: 32 bytes in 2 cycles (addr + latch)
                // =============================================================
                S_ALOAD: begin
                    // Wide read: 32 consecutive cin bytes for this kpos
                    patch_rd_addr <= kpos * c_in + cin_base;
                    state <= S_ALOAD_LATCH;
                end

                S_ALOAD_LATCH: begin
                    cin_actual = (cin_base + 32 <= c_in) ? 32 : (c_in - cin_base);
                    for (jj = 0; jj < 32; jj = jj + 1) begin
                        if (jj < cin_actual)
                            act_reg[jj] <= $signed(patch_rd_data_wide[jj*8 +: 8]);
                        else
                            act_reg[jj] <= 8'sd0;
                    end
                    state <= S_MAC_FIRE;
                end

                // =============================================================
                // MAC fire
                // =============================================================
                S_MAC_FIRE: begin
                    arr_valid <= 1'b1;
                    arr_clear <= first_mac;
                    first_mac <= 1'b0;
                    state     <= S_MAC_WAIT;
                end

                S_MAC_WAIT: begin
                    if (arr_done) begin
                        // Next cin tile?
                        if (cin_base + 32 < c_in) begin
                            cin_base <= cin_base + 32;
                            ld_r <= 0;
                            state <= S_WLOAD_ROW;
                        end
                        // Next kernel pos?
                        else if (kpos + 1 < k_sq) begin
                            kpos     <= kpos + 1;
                            cin_base <= 0;
                            ld_r <= 0;
                            state <= S_WLOAD_ROW;
                        end
                        // All MACs done for Cout tile -> capture acc and post-process
                        else begin
                            // Capture MAC accumulators from packed wire into local regs
                            for (ii = 0; ii < 32; ii = ii + 1)
                                arr_acc[ii] <= $signed(acc_flat[ii*32 +: 32]);
                            state <= S_POST_PROCESS;
                        end
                    end
                end

                // =============================================================
                // Post-process and output
                // =============================================================
                S_POST_PROCESS: begin
                    pp_valid <= 1'b1;
                    state    <= S_PP_WAIT;
                end

                S_PP_WAIT: begin
                    if (pp_done) begin
                        // Capture post-process results from packed wire
                        for (ii = 0; ii < 32; ii = ii + 1)
                            pp_result[ii] <= $signed(pp_result_flat[ii*8 +: 8]);
                        state <= S_OUTPUT;
                    end
                end

                S_OUTPUT: begin
                    active_rows = (cout_base + 32 <= c_out) ? 32 : (c_out - cout_base);
                    out_valid   <= 1'b1;
                    out_ch_base <= cout_base[8:0];
                    out_count   <= active_rows[5:0];
                    for (ii = 0; ii < 32; ii = ii + 1)
                        out_data_int[ii] <= pp_result[ii];

                    // Next Cout tile?
                    if (cout_base + 32 < c_out) begin
                        cout_base <= cout_base + 32;
                        ld_b      <= 0;
                        state     <= S_BIAS_REQ;
                    end else begin
                        state <= S_DONE;
                    end
                end

                // =============================================================
                S_DONE: begin
                    done  <= 1'b1;
                    state <= S_IDLE;
                end

                default: state <= S_IDLE;
            endcase
        end
    end

endmodule
`default_nettype wire
