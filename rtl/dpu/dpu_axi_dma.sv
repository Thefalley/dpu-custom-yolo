// =============================================================================
// DPU AXI-DMA Streaming Interface
// Provides AXI4-Stream slave (S_AXIS) for bulk weight/input loading
// and AXI4-Stream master (M_AXIS) for output readback.
//
// Control flow:
//   1. Software sets target (weight_buf / fmap / bias) and base address via CSR
//   2. Software starts DMA transfer via Xilinx AXI-DMA IP
//   3. This module receives data on S_AXIS, translates to PIO write_byte cmds
//   4. For readback: module issues PIO read_byte, streams results on M_AXIS
//
// CSR (directly wired, not AXI — controlled by dpu_system_top):
//   dma_target[2:0]   = 0: weight_buf, 1: fmap_input, 2: bias_buf, 3: scale, 4: layer_desc
//   dma_base_addr[23:0] = starting address
//   dma_length[23:0]   = number of bytes to transfer
//   dma_start          = pulse to begin
//   dma_done           = high when complete
// =============================================================================

module dpu_axi_dma #(
    parameter int ADDR_BITS   = 24,
    parameter int AXIS_DATA_W = 32,  // AXI-Stream data width (bytes packed)
    parameter int MAX_WBUF    = 147456,
    parameter int MAX_CH      = 256
) (
    input  logic                    clk,
    input  logic                    rst_n,

    // ---- AXI4-Stream Slave (data IN from DMA) ----
    input  logic [AXIS_DATA_W-1:0] s_axis_tdata,
    input  logic [AXIS_DATA_W/8-1:0] s_axis_tkeep,
    input  logic                    s_axis_tvalid,
    output logic                    s_axis_tready,
    input  logic                    s_axis_tlast,

    // ---- AXI4-Stream Master (data OUT to DMA) ----
    output logic [AXIS_DATA_W-1:0] m_axis_tdata,
    output logic [AXIS_DATA_W/8-1:0] m_axis_tkeep,
    output logic                    m_axis_tvalid,
    input  logic                    m_axis_tready,
    output logic                    m_axis_tlast,

    // ---- CSR from system controller ----
    input  logic [2:0]             dma_target,
    input  logic [ADDR_BITS-1:0]   dma_base_addr,
    input  logic [ADDR_BITS-1:0]   dma_length,
    input  logic                    dma_start,
    input  logic                    dma_dir,       // 0=write(S_AXIS->DPU), 1=read(DPU->M_AXIS)
    output logic                    dma_done,
    output logic                    dma_busy,

    // ---- DPU PIO interface (directly to dpu_top) ----
    output logic                    cmd_valid,
    input  logic                    cmd_ready,
    output logic [2:0]             cmd_type,
    output logic [ADDR_BITS-1:0]   cmd_addr,
    output logic [7:0]             cmd_data,
    input  logic                    rsp_valid,
    input  logic [7:0]             rsp_data
);

    // =========================================================================
    // Target to cmd_type mapping & address offsets
    // =========================================================================
    // target 0 (weight)     -> cmd_type 0 (write_byte), addr = base + offset
    // target 1 (fmap_input) -> cmd_type 0 (write_byte), addr = FMAP_OFFSET + base + offset
    // target 2 (bias)       -> cmd_type 0 (write_byte), addr = BIAS_OFFSET + base + offset
    // target 3 (scale)      -> cmd_type 5 (write_scale)
    // target 4 (layer_desc) -> cmd_type 6 (write_layer_desc)
    // readback              -> cmd_type 2 (read_byte)
    localparam [ADDR_BITS-1:0] BIAS_OFFSET = MAX_WBUF;
    localparam [ADDR_BITS-1:0] FMAP_OFFSET = MAX_WBUF + MAX_CH * 4;

    // =========================================================================
    // State machine
    // =========================================================================
    localparam S_IDLE      = 3'd0;
    localparam S_WR_FETCH  = 3'd1;  // get next byte from S_AXIS buffer
    localparam S_WR_CMD    = 3'd2;  // issue PIO write command
    localparam S_WR_WAIT   = 3'd3;  // wait for cmd_ready
    localparam S_RD_CMD    = 3'd4;  // issue PIO read command
    localparam S_RD_WAIT   = 3'd5;  // wait for rsp_valid
    localparam S_RD_PUSH   = 3'd6;  // push byte on M_AXIS
    localparam S_DONE      = 3'd7;

    logic [2:0]           state;
    logic [ADDR_BITS-1:0] byte_cnt;
    logic [ADDR_BITS-1:0] cur_addr;
    logic [1:0]           word_byte;    // byte offset within 32-bit word
    logic [AXIS_DATA_W-1:0] axis_buf;  // latched input word
    logic                 axis_buf_valid;

    // Byte extraction from 32-bit AXI word
    logic [7:0] cur_byte;
    assign cur_byte = axis_buf[word_byte*8 +: 8];

    // Output assignments
    assign dma_busy = (state != S_IDLE && state != S_DONE);
    assign dma_done = (state == S_DONE);

    assign m_axis_tkeep = {(AXIS_DATA_W/8){1'b1}};

    // =========================================================================
    // Main FSM
    // =========================================================================
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            state          <= S_IDLE;
            byte_cnt       <= '0;
            cur_addr       <= '0;
            word_byte      <= 2'd0;
            axis_buf       <= '0;
            axis_buf_valid <= 1'b0;
            cmd_valid      <= 1'b0;
            cmd_type       <= 3'd0;
            cmd_addr       <= '0;
            cmd_data       <= 8'd0;
            m_axis_tdata   <= '0;
            m_axis_tvalid  <= 1'b0;
            m_axis_tlast   <= 1'b0;
            s_axis_tready  <= 1'b0;
        end else begin
            // Default de-assert
            cmd_valid     <= 1'b0;
            s_axis_tready <= 1'b0;
            m_axis_tvalid <= 1'b0;
            m_axis_tlast  <= 1'b0;

            case (state)
                // ---------------------------------------------------------
                S_IDLE: begin
                    if (dma_start) begin
                        byte_cnt  <= '0;
                        cur_addr  <= dma_base_addr;
                        word_byte <= 2'd0;
                        axis_buf_valid <= 1'b0;
                        if (dma_dir == 1'b0)
                            state <= S_WR_FETCH;
                        else
                            state <= S_RD_CMD;
                    end
                end

                // ============ WRITE PATH (S_AXIS -> DPU) =================
                S_WR_FETCH: begin
                    if (byte_cnt >= dma_length) begin
                        state <= S_DONE;
                    end else if (!axis_buf_valid || word_byte == 2'd0) begin
                        // Need new word from S_AXIS
                        s_axis_tready <= 1'b1;
                        if (s_axis_tvalid && s_axis_tready) begin
                            s_axis_tready  <= 1'b0;  // deassert after accept
                            axis_buf       <= s_axis_tdata;
                            axis_buf_valid <= 1'b1;
                            word_byte      <= 2'd0;
                            state          <= S_WR_CMD;
                        end
                    end else begin
                        state <= S_WR_CMD;
                    end
                end

                S_WR_CMD: begin
                    // Select cmd_type based on target
                    case (dma_target)
                        3'd0, 3'd1, 3'd2: cmd_type <= 3'd0; // write_byte
                        3'd3:              cmd_type <= 3'd5; // write_scale
                        3'd4:              cmd_type <= 3'd6; // write_layer_desc
                        default:           cmd_type <= 3'd0;
                    endcase
                    // Apply target-specific address offset
                    case (dma_target)
                        3'd0:    cmd_addr <= cur_addr;                    // weight_buf
                        3'd1:    cmd_addr <= FMAP_OFFSET + cur_addr;     // fmap_input
                        3'd2:    cmd_addr <= BIAS_OFFSET + cur_addr;     // bias_buf
                        default: cmd_addr <= cur_addr;                    // scale/layer_desc
                    endcase
                    cmd_data  <= cur_byte;
                    cmd_valid <= 1'b1;
                    state     <= S_WR_WAIT;
                end

                S_WR_WAIT: begin
                    cmd_valid <= 1'b1;
                    if (cmd_ready) begin
                        cmd_valid <= 1'b0;
                        byte_cnt  <= byte_cnt + 1;
                        cur_addr  <= cur_addr + 1;
                        if (word_byte == 2'd3) begin
                            word_byte      <= 2'd0;
                            axis_buf_valid <= 1'b0;
                            state          <= S_WR_FETCH;
                        end else begin
                            word_byte <= word_byte + 1;
                            state     <= (byte_cnt + 1 >= dma_length) ? S_DONE : S_WR_CMD;
                        end
                    end
                end

                // ============ READ PATH (DPU -> M_AXIS) ==================
                S_RD_CMD: begin
                    if (byte_cnt >= dma_length) begin
                        state <= S_DONE;
                    end else begin
                        cmd_type  <= 3'd2;   // read_byte
                        cmd_addr  <= cur_addr;
                        cmd_data  <= 8'd0;
                        cmd_valid <= 1'b1;
                        state     <= S_RD_WAIT;
                    end
                end

                S_RD_WAIT: begin
                    cmd_valid <= 1'b1;
                    if (cmd_ready) begin
                        cmd_valid <= 1'b0;
                    end
                    if (rsp_valid) begin
                        // Pack bytes into 32-bit word
                        m_axis_tdata[word_byte*8 +: 8] <= rsp_data;
                        byte_cnt <= byte_cnt + 1;
                        cur_addr <= cur_addr + 1;
                        if (word_byte == 2'd3 || byte_cnt + 1 >= dma_length) begin
                            state <= S_RD_PUSH;
                        end else begin
                            word_byte <= word_byte + 1;
                            state     <= S_RD_CMD;
                        end
                    end
                end

                S_RD_PUSH: begin
                    m_axis_tvalid <= 1'b1;
                    m_axis_tlast  <= (byte_cnt >= dma_length);
                    if (m_axis_tvalid && m_axis_tready) begin
                        word_byte <= 2'd0;
                        m_axis_tdata <= '0;
                        m_axis_tvalid <= 1'b0;
                        m_axis_tlast  <= 1'b0;
                        if (byte_cnt >= dma_length)
                            state <= S_DONE;
                        else
                            state <= S_RD_CMD;
                    end
                end

                // ---------------------------------------------------------
                S_DONE: begin
                    // Stay until dma_start deasserted
                    if (!dma_start)
                        state <= S_IDLE;
                end

                default: state <= S_IDLE;
            endcase
        end
    end

endmodule
