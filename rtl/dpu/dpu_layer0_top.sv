// DPU Layer0 Top: instantiates layer0_engine, memories, and command interface.
// Command interface: write bytes (load image/weights/bias), start run, read output.
// Enables a structured testbench: LoadImage(), RunLayer0(), CompareOutput().
`default_nettype none

module dpu_layer0_top #(
    parameter int H_OUT   = 208,
    parameter int W_OUT   = 208,
    parameter int NUM_CH  = 32,
    parameter int PAD_H   = 418,
    parameter int PAD_W   = 418,
    parameter int ADDR_BITS = 24
) (
    input  logic        clk,
    input  logic        rst_n,

    // ---- Command interface ----
    input  logic        cmd_valid,
    output logic        cmd_ready,
    input  logic [1:0]  cmd_type,    // 0=write_byte, 1=run_layer0, 2=read_byte
    input  logic [ADDR_BITS-1:0] cmd_addr,
    input  logic [7:0]  cmd_data,

    output logic        rsp_valid,
    output logic [7:0]  rsp_data,
    output logic        busy,
    output logic        done
);

    // Derived sizes (calculated from parameters)
    localparam int PAD_SIZE = 3 * PAD_H * PAD_W;
    localparam int W_SIZE   = NUM_CH * 27;
    localparam int B_SIZE   = NUM_CH * 4;
    localparam int OUT_SIZE = H_OUT * W_OUT * NUM_CH;

    // Address map (byte)
    localparam int INPUT_BASE  = 0;
    localparam int WEIGHT_BASE = PAD_SIZE;
    localparam int BIAS_BASE   = PAD_SIZE + W_SIZE;
    localparam int OUTPUT_BASE = PAD_SIZE + W_SIZE + B_SIZE;

    // ---- Memories ----
    reg signed [7:0]  input_mem  [0:PAD_SIZE-1];
    reg signed [7:0]  weight_mem [0:W_SIZE-1];
    reg signed [31:0] bias_mem   [0:NUM_CH-1];
    reg signed [7:0]  output_mem [0:OUT_SIZE-1];

    // Initialize bias_mem to 0 (needed for read-modify-write)
    integer init_i;
    initial begin
        for (init_i = 0; init_i < NUM_CH; init_i = init_i + 1)
            bias_mem[init_i] = 32'sd0;
    end

    // ---- Engine ----
    logic        eng_start, eng_done;
    logic signed [7:0]  eng_act_in, eng_w_in;
    logic signed [31:0] eng_bias;
    logic [15:0] eng_scale;
    logic signed [7:0]  eng_result_int8;
    logic [4:0]  eng_mac_index;

    // Patch/weight buffers for current pixel (engine expects act_in/w_in per mac_index)
    reg signed [7:0] patch_buf [0:26];
    reg signed [7:0] weight_buf [0:26];

    layer0_engine #(.MACS(27), .SCALE_Q(16)) u_engine (
        .clk(clk), .rst_n(rst_n), .start(eng_start),
        .act_in(eng_act_in), .w_in(eng_w_in), .bias(eng_bias), .scale(eng_scale),
        .done(eng_done), .result_int8(eng_result_int8), .mac_index(eng_mac_index)
    );

    assign eng_act_in = patch_buf[eng_mac_index];
    assign eng_w_in  = weight_buf[eng_mac_index];
    assign eng_scale = 16'd655;

    // ---- Command decode & FSM ----
    logic [ADDR_BITS-1:0] cmd_addr_u;
    assign cmd_addr_u = cmd_addr;

    typedef enum logic [3:0] {
        S_IDLE,
        S_WRITE_ACK,
        S_READ_ACK,
        S_RUN_LOAD_PATCH,
        S_RUN_START_ENG,
        S_RUN_ENGINE,
        S_RUN_LATCH_RESULT,  // Wait one cycle for result to be valid
        S_RUN_WRITE_OUT,
        S_RUN_NEXT,
        S_DONE
    } state_t;
    state_t state;

    integer oh, ow, ch, k;
    integer a_addr;
    logic [31:0] out_index;
    logic signed [31:0] bias_val;
    int load_count;
    logic signed [7:0] result_captured;  // Capture result when eng_done fires

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state    <= S_IDLE;
            cmd_ready<= 1'b0;
            rsp_valid<= 1'b0;
            rsp_data <= 8'd0;
            busy     <= 1'b0;
            done     <= 1'b0;
            eng_start<= 1'b0;
            eng_bias <= 32'sd0;
            oh       <= 0;
            ow       <= 0;
            ch       <= 0;
            load_count <= 0;
        end else begin
            cmd_ready <= 1'b0;
            rsp_valid <= 1'b0;
            done      <= 1'b0;
            eng_start <= 1'b0;

            case (state)
                S_IDLE: begin
                    cmd_ready <= 1'b1;  // Ready in IDLE
                    if (cmd_valid) begin
                        case (cmd_type)
                            2'd0: begin // write_byte
                                if (cmd_addr_u < PAD_SIZE)
                                    input_mem[cmd_addr_u[19:0]] <= $signed(cmd_data);
                                else if (cmd_addr_u < PAD_SIZE + W_SIZE)
                                    weight_mem[cmd_addr_u - WEIGHT_BASE] <= $signed(cmd_data);
                                else if (cmd_addr_u < OUTPUT_BASE) begin
                                    // bias: 4 bytes per channel (little-endian), read-modify-write
                                    integer bch;
                                    integer bsel;
                                    bch  = (cmd_addr_u - BIAS_BASE) >> 2;
                                    bsel = (cmd_addr_u - BIAS_BASE) & 32'd3;
                                    if (bsel == 0)      bias_mem[bch] <= { bias_mem[bch][31:8], cmd_data };
                                    else if (bsel == 1) bias_mem[bch] <= { bias_mem[bch][31:16], cmd_data, bias_mem[bch][7:0] };
                                    else if (bsel == 2) bias_mem[bch] <= { bias_mem[bch][31:24], cmd_data, bias_mem[bch][15:0] };
                                    else                bias_mem[bch] <= { cmd_data, bias_mem[bch][23:0] };
                                end
                                state <= S_WRITE_ACK;
                            end
                            2'd1: begin // run_layer0
                                busy  <= 1'b1;
                                oh    <= 0;
                                ow    <= 0;
                                ch    <= 0;
                                load_count <= 0;
                                state <= S_RUN_LOAD_PATCH;
                            end
                            2'd2: begin // read_byte
                                if (cmd_addr_u >= OUTPUT_BASE && cmd_addr_u < OUTPUT_BASE + OUT_SIZE)
                                    rsp_data <= output_mem[cmd_addr_u - OUTPUT_BASE];
                                else
                                    rsp_data <= 8'd0;
                                rsp_valid <= 1'b1;
                                state <= S_READ_ACK;
                            end
                            default: cmd_ready <= 1'b1;
                        endcase
                    end
                end

                S_WRITE_ACK: begin
                    cmd_ready <= 1'b1;
                    state     <= S_IDLE;
                end

                S_READ_ACK: begin
                    cmd_ready <= 1'b1;
                    state     <= S_IDLE;
                end

                S_RUN_LOAD_PATCH: begin
                    // Load 27 activations and 27 weights for (oh,ow,ch)
                    k = load_count;
                    a_addr = (k/9)*PAD_H*PAD_W + (oh*2 + (k%9)/3)*PAD_W + (ow*2) + (k%3);
                    patch_buf[k]  <= input_mem[a_addr];
                    weight_buf[k] <= weight_mem[ch*27 + k];
                    if (load_count == 26) begin
                        bias_val <= bias_mem[ch];
                        eng_bias <= bias_mem[ch];
                        state    <= S_RUN_START_ENG;
                    end else
                        load_count <= load_count + 1;
                end

                S_RUN_START_ENG: begin
                    eng_start <= 1'b1;
                    state     <= S_RUN_ENGINE;
                end

                S_RUN_ENGINE: begin
                    if (eng_done) begin
                        out_index <= (oh * W_OUT + ow) * NUM_CH + ch;
                        state     <= S_RUN_LATCH_RESULT;
                    end
                end

                S_RUN_LATCH_RESULT: begin
                    // Capture eng_result_int8 now that it's valid
                    result_captured <= eng_result_int8;
                    state <= S_RUN_WRITE_OUT;
                end

                S_RUN_WRITE_OUT: begin
                    output_mem[out_index] <= result_captured;
                    state <= S_RUN_NEXT;
                end

                S_RUN_NEXT: begin
                    if (ch + 1 >= NUM_CH) begin
                        ch <= 0;
                        ow <= ow + 1;
                        if (ow + 1 >= W_OUT) begin
                            ow <= 0;
                            oh <= oh + 1;
                            if (oh + 1 >= H_OUT) begin
                                state <= S_DONE;
                                busy  <= 1'b0;
                                done  <= 1'b1;
                                cmd_ready <= 1'b1;
                            end else begin
                                load_count <= 0;
                                state <= S_RUN_LOAD_PATCH;
                            end
                        end else begin
                            load_count <= 0;
                            state <= S_RUN_LOAD_PATCH;
                        end
                    end else begin
                        ch <= ch + 1;
                        load_count <= 0;
                        state <= S_RUN_LOAD_PATCH;
                    end
                end

                S_DONE: begin
                    state <= S_IDLE;
                end

                default: state <= S_IDLE;
            endcase
        end
    end

endmodule
`default_nettype wire
