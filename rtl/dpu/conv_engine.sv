// Conv engine: one output pixel, one channel (up to 1152 MACs + bias + LeakyReLU + requantize).
// Generalized from layer0_engine for any conv layer (3x3 or 1x1).
// Parent must present act_in and w_in each cycle during MAC phase; engine outputs done + result_int8.
`default_nettype none

module conv_engine #(
    parameter int SCALE_Q = 16
) (
    input  logic        clk,
    input  logic        rst_n,
    input  logic        start,
    input  logic [10:0] macs_count,      // runtime: 27, 64, 128, 288, 576, 1152
    input  logic signed [7:0]  act_in,
    input  logic signed [7:0]  w_in,
    input  logic signed [31:0] bias,
    input  logic [15:0] scale,
    output logic        done,
    output logic signed [7:0]  result_int8,
    output logic [10:0] mac_index   // which (act,w) pair to present (valid during MAC phase)
);

    logic        mac_valid;
    logic signed [31:0] mac_acc_in, mac_acc_out;
    logic signed [31:0] acc_feedback;

    logic        leaky_valid;
    logic signed [31:0] leaky_x, leaky_y;

    logic        req_valid;
    logic signed [31:0] req_acc;
    logic signed [7:0]  req_out;

    mac_int8 u_mac (
        .clk(clk), .rst_n(rst_n), .valid(mac_valid),
        .weight(w_in), .activation(act_in), .acc_in(mac_acc_in),
        .acc_out(mac_acc_out), .done()
    );
    leaky_relu u_leaky (
        .clk(clk), .rst_n(rst_n), .valid(leaky_valid),
        .x(leaky_x), .y(leaky_y), .done()
    );
    requantize #(.SCALE_Q(SCALE_Q)) u_req (
        .clk(clk), .rst_n(rst_n), .valid(req_valid),
        .acc(req_acc), .scale(scale),
        .out_int8(req_out), .done()
    );

    typedef enum logic [3:0] {
        S_IDLE, S_MAC_DRIVE, S_MAC_WAIT, S_MAC_LATCH,
        S_LEAKY_FEED, S_LEAKY_WAIT, S_REQ_FEED, S_REQ_WAIT, S_DONE
    } state_t;
    state_t state;
    logic [10:0] mac_count;
    logic [1:0]  wait_count;

    assign mac_acc_in = acc_feedback;
    assign leaky_x   = acc_feedback + bias;
    assign req_acc    = leaky_y;
    assign mac_index  = mac_count;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state       <= S_IDLE;
            mac_valid   <= 1'b0;
            leaky_valid <= 1'b0;
            req_valid   <= 1'b0;
            acc_feedback<= 32'sd0;
            result_int8 <= 8'sd0;
            done        <= 1'b0;
            mac_count   <= 11'd0;
            wait_count  <= 2'd0;
        end else begin
            done <= 1'b0;
            case (state)
                S_IDLE: begin
                    acc_feedback <= 32'sd0;
                    if (start) begin
                        state     <= S_MAC_DRIVE;
                        mac_count <= 11'd0;
                        mac_valid <= 1'b1;
                    end
                end
                S_MAC_DRIVE: begin
                    mac_valid <= 1'b1;
                    state     <= S_MAC_WAIT;
                end
                S_MAC_WAIT: begin
                    mac_valid <= 1'b0;
                    state     <= S_MAC_LATCH;
                end
                S_MAC_LATCH: begin
                    acc_feedback <= mac_acc_out;
                    if (mac_count == (macs_count - 11'd1)) begin
                        state <= S_LEAKY_FEED;
                    end else begin
                        mac_count <= mac_count + 11'd1;
                        state     <= S_MAC_DRIVE;
                        mac_valid <= 1'b1;
                    end
                end
                S_LEAKY_FEED: begin
                    leaky_valid <= 1'b1;
                    wait_count  <= 2'd2;
                    state       <= S_LEAKY_WAIT;
                end
                S_LEAKY_WAIT: begin
                    leaky_valid <= 1'b0;
                    if (wait_count == 2'd0)
                        state <= S_REQ_FEED;
                    else
                        wait_count <= wait_count - 2'd1;
                end
                S_REQ_FEED: begin
                    req_valid  <= 1'b1;
                    state      <= S_REQ_WAIT;
                    wait_count <= 2'd2;
                end
                S_REQ_WAIT: begin
                    req_valid <= 1'b0;
                    if (wait_count == 2'd0) begin
                        state       <= S_DONE;
                        result_int8 <= req_out;
                        done        <= 1'b1;
                    end else
                        wait_count <= wait_count - 2'd1;
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
