// =============================================================================
// DPU AXI4-Lite Slave Wrapper
// Maps standard AXI4-Lite register interface to dpu_top PIO commands.
//
// Register Map (byte addresses):
//   0x00  CMD      [W]   [2:0] = cmd_type
//   0x04  ADDR     [W]   [23:0] = cmd_addr
//   0x08  WDATA    [W]   [7:0] = cmd_data; write triggers PIO transaction
//   0x0C  RDATA    [R]   [7:0] = rsp_data (last read_byte result)
//   0x10  STATUS   [R]   [0]=busy [1]=done [12:8]=current_layer [16]=reload_req [17]=cmd_ready
//   0x14  PERF     [R]   [31:0] = perf_total_cycles
//   0x18  IRQ_EN   [RW]  [0]=done_irq_en [1]=reload_irq_en
//   0x1C  IRQ_STAT [R/W1C] [0]=done_irq [1]=reload_irq
// =============================================================================

module dpu_axi4_lite #(
    parameter int H0        = 16,
    parameter int W0        = 16,
    parameter int MAX_CH    = 256,
    parameter int MAX_FMAP  = 65536,
    parameter int MAX_WBUF  = 147456,
    parameter int ADDR_BITS = 24,
    parameter int AXI_ADDR_W = 6,   // 64-byte register space
    parameter int AXI_DATA_W = 32
) (
    // ---- AXI4-Lite Slave Interface ----
    input  logic                    s_axi_aclk,
    input  logic                    s_axi_aresetn,

    // Write address channel
    input  logic [AXI_ADDR_W-1:0]  s_axi_awaddr,
    input  logic [2:0]             s_axi_awprot,
    input  logic                    s_axi_awvalid,
    output logic                    s_axi_awready,

    // Write data channel
    input  logic [AXI_DATA_W-1:0]  s_axi_wdata,
    input  logic [AXI_DATA_W/8-1:0] s_axi_wstrb,
    input  logic                    s_axi_wvalid,
    output logic                    s_axi_wready,

    // Write response channel
    output logic [1:0]             s_axi_bresp,
    output logic                    s_axi_bvalid,
    input  logic                    s_axi_bready,

    // Read address channel
    input  logic [AXI_ADDR_W-1:0]  s_axi_araddr,
    input  logic [2:0]             s_axi_arprot,
    input  logic                    s_axi_arvalid,
    output logic                    s_axi_arready,

    // Read data channel
    output logic [AXI_DATA_W-1:0]  s_axi_rdata,
    output logic [1:0]             s_axi_rresp,
    output logic                    s_axi_rvalid,
    input  logic                    s_axi_rready,

    // ---- Interrupt ----
    output logic                    irq
);

    // =========================================================================
    // AXI write handshake
    // =========================================================================
    logic                   aw_en;
    logic [AXI_ADDR_W-1:0] axi_awaddr;
    logic                   axi_awready;
    logic                   axi_wready;
    logic                   axi_bvalid;
    logic [AXI_ADDR_W-1:0] axi_araddr;
    logic                   axi_arready;
    logic                   axi_rvalid;
    logic [AXI_DATA_W-1:0] axi_rdata;

    assign s_axi_awready = axi_awready;
    assign s_axi_wready  = axi_wready;
    assign s_axi_bresp   = 2'b00;           // OKAY
    assign s_axi_bvalid  = axi_bvalid;
    assign s_axi_arready = axi_arready;
    assign s_axi_rdata   = axi_rdata;
    assign s_axi_rresp   = 2'b00;           // OKAY
    assign s_axi_rvalid  = axi_rvalid;

    // =========================================================================
    // Soft registers
    // =========================================================================
    logic [2:0]            reg_cmd;
    logic [ADDR_BITS-1:0]  reg_addr;
    logic [7:0]            reg_wdata;
    logic [1:0]            reg_irq_en;
    logic [1:0]            reg_irq_stat;

    // PIO transaction trigger
    logic                  pio_fire;      // pulse when WDATA register written

    // =========================================================================
    // DPU core instance
    // =========================================================================
    logic        dpu_cmd_valid;
    logic        dpu_cmd_ready;
    logic [2:0]  dpu_cmd_type;
    logic [ADDR_BITS-1:0] dpu_cmd_addr;
    logic [7:0]  dpu_cmd_data;
    logic        dpu_rsp_valid;
    logic [7:0]  dpu_rsp_data;
    logic        dpu_busy;
    logic        dpu_done;
    logic [4:0]  dpu_current_layer;
    logic        dpu_reload_req;
    logic [31:0] dpu_perf_total_cycles;

    dpu_top #(
        .H0       (H0),
        .W0       (W0),
        .MAX_CH   (MAX_CH),
        .MAX_FMAP (MAX_FMAP),
        .MAX_WBUF (MAX_WBUF),
        .ADDR_BITS(ADDR_BITS)
    ) u_dpu (
        .clk             (s_axi_aclk),
        .rst_n           (s_axi_aresetn),
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
        .perf_total_cycles(dpu_perf_total_cycles)
    );

    // =========================================================================
    // PIO transaction state machine
    // =========================================================================
    localparam S_PIO_IDLE = 0, S_PIO_REQ = 1, S_PIO_WAIT = 2;
    logic [1:0] pio_state;
    logic [7:0] rsp_latch;

    assign dpu_cmd_type = reg_cmd;
    assign dpu_cmd_addr = reg_addr;
    assign dpu_cmd_data = reg_wdata;
    assign dpu_cmd_valid = (pio_state == S_PIO_REQ);

    always_ff @(posedge s_axi_aclk) begin
        if (!s_axi_aresetn) begin
            pio_state <= S_PIO_IDLE;
            rsp_latch <= 8'd0;
        end else begin
            case (pio_state)
                S_PIO_IDLE: begin
                    if (pio_fire)
                        pio_state <= S_PIO_REQ;
                end
                S_PIO_REQ: begin
                    if (dpu_cmd_ready)
                        pio_state <= S_PIO_WAIT;
                end
                S_PIO_WAIT: begin
                    if (dpu_rsp_valid)
                        rsp_latch <= dpu_rsp_data;
                    if (!dpu_busy || dpu_done || dpu_reload_req ||
                        (reg_cmd == 3'd0) || (reg_cmd == 3'd3) ||
                        (reg_cmd == 3'd5) || (reg_cmd == 3'd6))
                        pio_state <= S_PIO_IDLE;
                end
                default: pio_state <= S_PIO_IDLE;
            endcase
        end
    end

    // =========================================================================
    // AXI4-Lite Write Channel
    // =========================================================================
    always_ff @(posedge s_axi_aclk) begin
        if (!s_axi_aresetn) begin
            axi_awready <= 1'b0;
            aw_en       <= 1'b1;
            axi_awaddr  <= '0;
        end else begin
            if (~axi_awready && s_axi_awvalid && s_axi_wvalid && aw_en) begin
                axi_awready <= 1'b1;
                aw_en       <= 1'b0;
                axi_awaddr  <= s_axi_awaddr;
            end else if (s_axi_bready && axi_bvalid) begin
                aw_en       <= 1'b1;
                axi_awready <= 1'b0;
            end else begin
                axi_awready <= 1'b0;
            end
        end
    end

    always_ff @(posedge s_axi_aclk) begin
        if (!s_axi_aresetn)
            axi_wready <= 1'b0;
        else if (~axi_wready && s_axi_wvalid && s_axi_awvalid && aw_en)
            axi_wready <= 1'b1;
        else
            axi_wready <= 1'b0;
    end

    // Register writes
    logic wr_en;
    assign wr_en = axi_wready && s_axi_wvalid && axi_awready && s_axi_awvalid;

    always_ff @(posedge s_axi_aclk) begin
        pio_fire <= 1'b0;
        if (!s_axi_aresetn) begin
            reg_cmd   <= 3'd0;
            reg_addr  <= '0;
            reg_wdata <= 8'd0;
            reg_irq_en <= 2'b00;
        end else if (wr_en) begin
            case (axi_awaddr[5:2])
                4'h0: reg_cmd   <= s_axi_wdata[2:0];       // CMD
                4'h1: reg_addr  <= s_axi_wdata[ADDR_BITS-1:0]; // ADDR
                4'h2: begin                                     // WDATA (trigger)
                    reg_wdata <= s_axi_wdata[7:0];
                    if (pio_state == S_PIO_IDLE)
                        pio_fire <= 1'b1;
                end
                4'h6: reg_irq_en <= s_axi_wdata[1:0];      // IRQ_EN
                4'h7: reg_irq_stat <= reg_irq_stat & ~s_axi_wdata[1:0]; // IRQ_STAT W1C
                default: ;
            endcase
        end
    end

    // Write response
    always_ff @(posedge s_axi_aclk) begin
        if (!s_axi_aresetn)
            axi_bvalid <= 1'b0;
        else if (wr_en && ~axi_bvalid)
            axi_bvalid <= 1'b1;
        else if (s_axi_bready && axi_bvalid)
            axi_bvalid <= 1'b0;
    end

    // =========================================================================
    // AXI4-Lite Read Channel
    // =========================================================================
    always_ff @(posedge s_axi_aclk) begin
        if (!s_axi_aresetn) begin
            axi_arready <= 1'b0;
            axi_araddr  <= '0;
        end else if (~axi_arready && s_axi_arvalid) begin
            axi_arready <= 1'b1;
            axi_araddr  <= s_axi_araddr;
        end else begin
            axi_arready <= 1'b0;
        end
    end

    always_ff @(posedge s_axi_aclk) begin
        if (!s_axi_aresetn) begin
            axi_rvalid <= 1'b0;
            axi_rdata  <= '0;
        end else if (axi_arready && s_axi_arvalid && ~axi_rvalid) begin
            axi_rvalid <= 1'b1;
            case (s_axi_araddr[5:2])
                4'h0: axi_rdata <= {29'd0, reg_cmd};                   // CMD
                4'h1: axi_rdata <= {{(32-ADDR_BITS){1'b0}}, reg_addr}; // ADDR
                4'h3: axi_rdata <= {24'd0, rsp_latch};                 // RDATA
                4'h4: axi_rdata <= {14'd0,                             // STATUS
                                    dpu_cmd_ready,
                                    dpu_reload_req,
                                    3'd0,
                                    dpu_current_layer,
                                    6'd0,
                                    dpu_done,
                                    dpu_busy};
                4'h5: axi_rdata <= dpu_perf_total_cycles;              // PERF
                4'h6: axi_rdata <= {30'd0, reg_irq_en};               // IRQ_EN
                4'h7: axi_rdata <= {30'd0, reg_irq_stat};             // IRQ_STAT
                default: axi_rdata <= 32'hDEAD_BEEF;
            endcase
        end else if (axi_rvalid && s_axi_rready) begin
            axi_rvalid <= 1'b0;
        end
    end

    // =========================================================================
    // Interrupt logic
    // =========================================================================
    logic done_prev, reload_prev;

    always_ff @(posedge s_axi_aclk) begin
        if (!s_axi_aresetn) begin
            reg_irq_stat <= 2'b00;
            done_prev    <= 1'b0;
            reload_prev  <= 1'b0;
        end else begin
            done_prev   <= dpu_done;
            reload_prev <= dpu_reload_req;
            // Rising-edge detection
            if (dpu_done && !done_prev)
                reg_irq_stat[0] <= 1'b1;
            if (dpu_reload_req && !reload_prev)
                reg_irq_stat[1] <= 1'b1;
            // W1C handled in write path
            if (wr_en && axi_awaddr[5:2] == 4'h7)
                reg_irq_stat <= reg_irq_stat & ~s_axi_wdata[1:0];
        end
    end

    assign irq = |(reg_irq_en & reg_irq_stat);

endmodule
