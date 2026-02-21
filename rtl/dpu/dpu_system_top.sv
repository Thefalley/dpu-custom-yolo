// =============================================================================
// DPU System Top - Complete SoC-ready wrapper
// Integrates: AXI4-Lite control + AXI-Stream DMA + dpu_top core
//
// For Zynq PS-PL integration:
//   - s_axi_lite  -> PS GP0 master (control registers)
//   - s_axis      -> AXI DMA MM2S (host -> DPU data)
//   - m_axis      -> AXI DMA S2MM (DPU data -> host)
//   - irq         -> PS interrupt controller
//
// Extended register map (AXI4-Lite):
//   0x00  CMD        [W]   [2:0] cmd_type for PIO
//   0x04  ADDR       [W]   [23:0] PIO address
//   0x08  WDATA      [W]   [7:0] PIO data (write triggers transaction)
//   0x0C  RDATA      [R]   [7:0] PIO read result
//   0x10  STATUS     [R]   busy/done/layer/reload/cmd_ready
//   0x14  PERF       [R]   total cycles
//   0x18  IRQ_EN     [RW]  interrupt enables
//   0x1C  IRQ_STAT   [R/W1C] interrupt status
//   0x20  DMA_TARGET [W]   [2:0] DMA target buffer
//   0x24  DMA_BASE   [W]   [23:0] DMA base address
//   0x28  DMA_LENGTH [W]   [23:0] DMA byte count
//   0x2C  DMA_CTRL   [W]   [0]=start [1]=direction(0=wr,1=rd)
//   0x30  DMA_STATUS [R]   [0]=busy [1]=done
//   0x34  VERSION    [R]   0x0001_0000 (v1.0)
// =============================================================================

module dpu_system_top #(
    parameter int H0         = 32,
    parameter int W0         = 32,
    parameter int MAX_CH     = 512,
    parameter int MAX_FMAP   = 65536,
    parameter int MAX_WBUF   = 2400000,
    parameter int NUM_LAYERS = 36,
    parameter int ADDR_BITS  = 24
) (
    // ---- Clock & Reset ----
    input  logic        aclk,
    input  logic        aresetn,

    // ---- AXI4-Lite Slave (Control) ----
    input  logic [7:0]  s_axi_awaddr,
    input  logic [2:0]  s_axi_awprot,
    input  logic        s_axi_awvalid,
    output logic        s_axi_awready,
    input  logic [31:0] s_axi_wdata,
    input  logic [3:0]  s_axi_wstrb,
    input  logic        s_axi_wvalid,
    output logic        s_axi_wready,
    output logic [1:0]  s_axi_bresp,
    output logic        s_axi_bvalid,
    input  logic        s_axi_bready,
    input  logic [7:0]  s_axi_araddr,
    input  logic [2:0]  s_axi_arprot,
    input  logic        s_axi_arvalid,
    output logic        s_axi_arready,
    output logic [31:0] s_axi_rdata,
    output logic [1:0]  s_axi_rresp,
    output logic        s_axi_rvalid,
    input  logic        s_axi_rready,

    // ---- AXI4-Stream Slave (DMA Write: Host -> DPU) ----
    input  logic [31:0] s_axis_tdata,
    input  logic [3:0]  s_axis_tkeep,
    input  logic        s_axis_tvalid,
    output logic        s_axis_tready,
    input  logic        s_axis_tlast,

    // ---- AXI4-Stream Master (DMA Read: DPU -> Host) ----
    output logic [31:0] m_axis_tdata,
    output logic [3:0]  m_axis_tkeep,
    output logic        m_axis_tvalid,
    input  logic        m_axis_tready,
    output logic        m_axis_tlast,

    // ---- Interrupt ----
    output logic        irq
);

    // =========================================================================
    // Internal signals
    // =========================================================================
    // PIO from AXI-Lite wrapper
    logic        lite_cmd_valid, lite_cmd_ready;
    logic [2:0]  lite_cmd_type;
    logic [ADDR_BITS-1:0] lite_cmd_addr;
    logic [7:0]  lite_cmd_data;

    // PIO from DMA engine
    logic        dma_cmd_valid, dma_cmd_ready;
    logic [2:0]  dma_cmd_type;
    logic [ADDR_BITS-1:0] dma_cmd_addr;
    logic [7:0]  dma_cmd_data;

    // Muxed PIO to DPU core
    logic        dpu_cmd_valid, dpu_cmd_ready;
    logic [2:0]  dpu_cmd_type;
    logic [ADDR_BITS-1:0] dpu_cmd_addr;
    logic [7:0]  dpu_cmd_data;
    logic        dpu_rsp_valid;
    logic [7:0]  dpu_rsp_data;
    logic        dpu_busy, dpu_done;
    logic [5:0]  dpu_current_layer;
    logic        dpu_reload_req;
    logic [31:0] dpu_perf;

    // DMA CSR
    logic [2:0]            dma_target;
    logic [ADDR_BITS-1:0]  dma_base_addr;
    logic [ADDR_BITS-1:0]  dma_length;
    logic                   dma_start;
    logic                   dma_dir;
    logic                   dma_done_flag;
    logic                   dma_busy_flag;

    // =========================================================================
    // DPU Core
    // =========================================================================
    dpu_top #(
        .H0        (H0),
        .W0        (W0),
        .MAX_CH    (MAX_CH),
        .MAX_FMAP  (MAX_FMAP),
        .MAX_WBUF  (MAX_WBUF),
        .NUM_LAYERS(NUM_LAYERS),
        .ADDR_BITS (ADDR_BITS)
    ) u_dpu (
        .clk             (aclk),
        .rst_n           (aresetn),
        .cmd_valid       (dpu_cmd_valid),
        .cmd_ready       (dpu_cmd_ready),
        .cmd_type        (dpu_cmd_type),
        .cmd_addr        (dpu_cmd_addr),
        .cmd_data        (dpu_cmd_data),
        .rsp_valid       (dpu_rsp_valid),
        .rsp_data        (dpu_rsp_data),
        .busy            (dpu_busy),
        .done            (dpu_done),
        .current_layer   (dpu_current_layer),
        .reload_req      (dpu_reload_req),
        .perf_total_cycles(dpu_perf),
        .layer_done_pulse(),
        .done_layer_idx  ()
    );

    // =========================================================================
    // PIO Arbiter: DMA has priority when active, else AXI-Lite PIO
    // =========================================================================
    assign dpu_cmd_valid = dma_busy_flag ? dma_cmd_valid : lite_cmd_valid;
    assign dpu_cmd_type  = dma_busy_flag ? dma_cmd_type  : lite_cmd_type;
    assign dpu_cmd_addr  = dma_busy_flag ? dma_cmd_addr  : lite_cmd_addr;
    assign dpu_cmd_data  = dma_busy_flag ? dma_cmd_data  : lite_cmd_data;
    assign dma_cmd_ready  = dma_busy_flag ? dpu_cmd_ready : 1'b0;
    assign lite_cmd_ready = dma_busy_flag ? 1'b0 : dpu_cmd_ready;

    // =========================================================================
    // AXI4-Lite Register Bank (extended with DMA CSRs)
    // =========================================================================
    // Write address channel
    logic        aw_en;
    logic [7:0]  axi_awaddr_r;
    logic        axi_awready_r, axi_wready_r, axi_bvalid_r;
    logic        axi_arready_r, axi_rvalid_r;
    logic [31:0] axi_rdata_r;

    assign s_axi_awready = axi_awready_r;
    assign s_axi_wready  = axi_wready_r;
    assign s_axi_bresp   = 2'b00;
    assign s_axi_bvalid  = axi_bvalid_r;
    assign s_axi_arready = axi_arready_r;
    assign s_axi_rdata   = axi_rdata_r;
    assign s_axi_rresp   = 2'b00;
    assign s_axi_rvalid  = axi_rvalid_r;

    // Soft registers
    logic [2:0]           reg_cmd;
    logic [ADDR_BITS-1:0] reg_addr;
    logic [7:0]           reg_wdata;
    logic [1:0]           reg_irq_en;
    logic [1:0]           reg_irq_stat;
    logic                 pio_fire;
    logic [7:0]           rsp_latch;

    // PIO state machine for AXI-Lite path
    localparam S_PIO_IDLE = 0, S_PIO_REQ = 1, S_PIO_WAIT = 2;
    logic [1:0] pio_state;

    assign lite_cmd_type  = reg_cmd;
    assign lite_cmd_addr  = reg_addr;
    assign lite_cmd_data  = reg_wdata;
    assign lite_cmd_valid = (pio_state == S_PIO_REQ);

    always_ff @(posedge aclk) begin
        if (!aresetn) begin
            pio_state <= S_PIO_IDLE;
            rsp_latch <= 8'd0;
        end else begin
            case (pio_state)
                S_PIO_IDLE: if (pio_fire) pio_state <= S_PIO_REQ;
                S_PIO_REQ:  if (lite_cmd_ready) pio_state <= S_PIO_WAIT;
                S_PIO_WAIT: begin
                    if (dpu_rsp_valid) rsp_latch <= dpu_rsp_data;
                    if (!dpu_busy || dpu_done || dpu_reload_req ||
                        reg_cmd == 3'd0 || reg_cmd == 3'd3 ||
                        reg_cmd == 3'd5 || reg_cmd == 3'd6)
                        pio_state <= S_PIO_IDLE;
                end
                default: pio_state <= S_PIO_IDLE;
            endcase
        end
    end

    // AXI write handshake
    always_ff @(posedge aclk) begin
        if (!aresetn) begin
            axi_awready_r <= 1'b0;
            aw_en         <= 1'b1;
            axi_awaddr_r  <= '0;
        end else begin
            if (~axi_awready_r && s_axi_awvalid && s_axi_wvalid && aw_en) begin
                axi_awready_r <= 1'b1;
                aw_en         <= 1'b0;
                axi_awaddr_r  <= s_axi_awaddr;
            end else if (s_axi_bready && axi_bvalid_r) begin
                aw_en         <= 1'b1;
                axi_awready_r <= 1'b0;
            end else begin
                axi_awready_r <= 1'b0;
            end
        end
    end

    always_ff @(posedge aclk) begin
        if (!aresetn) axi_wready_r <= 1'b0;
        else if (~axi_wready_r && s_axi_wvalid && s_axi_awvalid && aw_en)
            axi_wready_r <= 1'b1;
        else axi_wready_r <= 1'b0;
    end

    logic wr_en;
    assign wr_en = axi_wready_r && s_axi_wvalid && axi_awready_r && s_axi_awvalid;

    // Register writes
    always_ff @(posedge aclk) begin
        pio_fire  <= 1'b0;
        dma_start <= 1'b0;
        if (!aresetn) begin
            reg_cmd      <= 3'd0;
            reg_addr     <= '0;
            reg_wdata    <= 8'd0;
            reg_irq_en   <= 2'b00;
            dma_target    <= 3'd0;
            dma_base_addr <= '0;
            dma_length    <= '0;
            dma_dir       <= 1'b0;
        end else if (wr_en) begin
            case (axi_awaddr_r[7:2])
                6'h00: reg_cmd      <= s_axi_wdata[2:0];
                6'h01: reg_addr     <= s_axi_wdata[ADDR_BITS-1:0];
                6'h02: begin
                    reg_wdata <= s_axi_wdata[7:0];
                    if (pio_state == S_PIO_IDLE) pio_fire <= 1'b1;
                end
                6'h06: reg_irq_en   <= s_axi_wdata[1:0];
                6'h07: ; // IRQ_STAT W1C handled below
                6'h08: dma_target    <= s_axi_wdata[2:0];
                6'h09: dma_base_addr <= s_axi_wdata[ADDR_BITS-1:0];
                6'h0A: dma_length    <= s_axi_wdata[ADDR_BITS-1:0];
                6'h0B: begin
                    dma_dir   <= s_axi_wdata[1];
                    dma_start <= s_axi_wdata[0];
                end
                default: ;
            endcase
        end
    end

    // Write response
    always_ff @(posedge aclk) begin
        if (!aresetn) axi_bvalid_r <= 1'b0;
        else if (wr_en && ~axi_bvalid_r) axi_bvalid_r <= 1'b1;
        else if (s_axi_bready && axi_bvalid_r) axi_bvalid_r <= 1'b0;
    end

    // AXI read handshake
    logic [7:0] axi_araddr_r;
    always_ff @(posedge aclk) begin
        if (!aresetn) begin
            axi_arready_r <= 1'b0;
            axi_araddr_r  <= '0;
        end else if (~axi_arready_r && s_axi_arvalid) begin
            axi_arready_r <= 1'b1;
            axi_araddr_r  <= s_axi_araddr;
        end else begin
            axi_arready_r <= 1'b0;
        end
    end

    always_ff @(posedge aclk) begin
        if (!aresetn) begin
            axi_rvalid_r <= 1'b0;
            axi_rdata_r  <= '0;
        end else if (axi_arready_r && s_axi_arvalid && ~axi_rvalid_r) begin
            axi_rvalid_r <= 1'b1;
            case (s_axi_araddr[7:2])
                6'h00: axi_rdata_r <= {29'd0, reg_cmd};
                6'h01: axi_rdata_r <= {{(32-ADDR_BITS){1'b0}}, reg_addr};
                6'h03: axi_rdata_r <= {24'd0, rsp_latch};
                6'h04: axi_rdata_r <= {14'd0, dpu_cmd_ready, dpu_reload_req,
                                        2'd0, dpu_current_layer, 6'd0, dpu_done, dpu_busy};
                6'h05: axi_rdata_r <= dpu_perf;
                6'h06: axi_rdata_r <= {30'd0, reg_irq_en};
                6'h07: axi_rdata_r <= {30'd0, reg_irq_stat};
                6'h08: axi_rdata_r <= {29'd0, dma_target};
                6'h09: axi_rdata_r <= {{(32-ADDR_BITS){1'b0}}, dma_base_addr};
                6'h0A: axi_rdata_r <= {{(32-ADDR_BITS){1'b0}}, dma_length};
                6'h0C: axi_rdata_r <= {30'd0, dma_done_flag, dma_busy_flag};
                6'h0D: axi_rdata_r <= 32'h0001_0000;  // VERSION v1.0
                default: axi_rdata_r <= 32'hDEAD_BEEF;
            endcase
        end else if (axi_rvalid_r && s_axi_rready) begin
            axi_rvalid_r <= 1'b0;
        end
    end

    // Interrupt logic
    logic done_prev, reload_prev;
    always_ff @(posedge aclk) begin
        if (!aresetn) begin
            reg_irq_stat <= 2'b00;
            done_prev    <= 1'b0;
            reload_prev  <= 1'b0;
        end else begin
            done_prev   <= dpu_done;
            reload_prev <= dpu_reload_req;
            if (dpu_done && !done_prev)        reg_irq_stat[0] <= 1'b1;
            if (dpu_reload_req && !reload_prev) reg_irq_stat[1] <= 1'b1;
            if (wr_en && axi_awaddr_r[7:2] == 6'h07)
                reg_irq_stat <= reg_irq_stat & ~s_axi_wdata[1:0];
        end
    end
    assign irq = |(reg_irq_en & reg_irq_stat);

    // =========================================================================
    // DMA Engine
    // =========================================================================
    dpu_axi_dma #(
        .ADDR_BITS  (ADDR_BITS),
        .AXIS_DATA_W(32),
        .MAX_WBUF   (MAX_WBUF),
        .MAX_CH     (MAX_CH)
    ) u_dma (
        .clk           (aclk),
        .rst_n         (aresetn),
        .s_axis_tdata  (s_axis_tdata),
        .s_axis_tkeep  (s_axis_tkeep),
        .s_axis_tvalid (s_axis_tvalid),
        .s_axis_tready (s_axis_tready),
        .s_axis_tlast  (s_axis_tlast),
        .m_axis_tdata  (m_axis_tdata),
        .m_axis_tkeep  (m_axis_tkeep),
        .m_axis_tvalid (m_axis_tvalid),
        .m_axis_tready (m_axis_tready),
        .m_axis_tlast  (m_axis_tlast),
        .dma_target    (dma_target),
        .dma_base_addr (dma_base_addr),
        .dma_length    (dma_length),
        .dma_start     (dma_start),
        .dma_dir       (dma_dir),
        .dma_done      (dma_done_flag),
        .dma_busy      (dma_busy_flag),
        .cmd_valid     (dma_cmd_valid),
        .cmd_ready     (dma_cmd_ready),
        .cmd_type      (dma_cmd_type),
        .cmd_addr      (dma_cmd_addr),
        .cmd_data      (dma_cmd_data),
        .rsp_valid     (dpu_rsp_valid),
        .rsp_data      (dpu_rsp_data)
    );

endmodule
