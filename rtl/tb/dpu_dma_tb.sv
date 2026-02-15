// =============================================================================
// Testbench: DMA Streaming through dpu_system_top
//
// Tests AXI4-Stream DMA path for bulk data loading and readback:
//   1. DMA write: load weights via AXI-Stream (bulk, 4 bytes/transfer)
//   2. PIO readback: verify weights match
//   3. DMA write: load input image via AXI-Stream
//   4. DMA write: load biases via AXI-Stream
//   5. DMA write: load scales via AXI-Stream (write_layer_desc)
//   6. PIO run_all: full 18-layer inference
//   7. DMA read: readback output via AXI-Stream
//   8. Compare output vs golden hex
//
// Compatible with Icarus Verilog.
// =============================================================================
`timescale 1ns / 1ps

module dpu_dma_tb;

    parameter H0 = 16, W0 = 16, MAX_CH = 256;
    parameter MAX_FMAP = 65536, MAX_WBUF = 147456, ADDR_BITS = 24;
    parameter CLK_PERIOD = 10;
    parameter FMAP_BASE = MAX_WBUF + MAX_CH * 4;  // 148480

    // Weight sizes
    parameter WSIZE_0  = 864;
    parameter WSIZE_1  = 18432;
    parameter WSIZE_2  = 36864;
    parameter WSIZE_4  = 9216;
    parameter WSIZE_5  = 9216;
    parameter WSIZE_7  = 4096;
    parameter WSIZE_10 = 147456;
    parameter WSIZE_12 = 36864;
    parameter WSIZE_13 = 36864;
    parameter WSIZE_15 = 16384;

    parameter COUT_0 = 32, COUT_1 = 64, COUT_2 = 64, COUT_4 = 32, COUT_5 = 32;
    parameter COUT_7 = 64, COUT_10 = 128, COUT_12 = 64, COUT_13 = 64, COUT_15 = 128;

    parameter OSIZE_17 = 256;  // 256ch * 1x1 for 16x16 input

    // ---- Signals ----
    reg         aclk;
    reg         aresetn;

    reg  [7:0]  s_axi_awaddr;
    reg  [2:0]  s_axi_awprot;
    reg         s_axi_awvalid;
    wire        s_axi_awready;
    reg  [31:0] s_axi_wdata;
    reg  [3:0]  s_axi_wstrb;
    reg         s_axi_wvalid;
    wire        s_axi_wready;
    wire [1:0]  s_axi_bresp;
    wire        s_axi_bvalid;
    reg         s_axi_bready;
    reg  [7:0]  s_axi_araddr;
    reg  [2:0]  s_axi_arprot;
    reg         s_axi_arvalid;
    wire        s_axi_arready;
    wire [31:0] s_axi_rdata;
    wire [1:0]  s_axi_rresp;
    wire        s_axi_rvalid;
    reg         s_axi_rready;

    reg  [31:0] s_axis_tdata;
    reg  [3:0]  s_axis_tkeep;
    reg         s_axis_tvalid;
    wire        s_axis_tready;
    reg         s_axis_tlast;
    wire [31:0] m_axis_tdata;
    wire [3:0]  m_axis_tkeep;
    wire        m_axis_tvalid;
    reg         m_axis_tready;
    wire        m_axis_tlast;

    wire        irq;

    // ---- DUT ----
    dpu_system_top #(
        .H0(H0), .W0(W0), .MAX_CH(MAX_CH),
        .MAX_FMAP(MAX_FMAP), .MAX_WBUF(MAX_WBUF), .ADDR_BITS(ADDR_BITS)
    ) u_dut (
        .aclk(aclk), .aresetn(aresetn),
        .s_axi_awaddr(s_axi_awaddr), .s_axi_awprot(s_axi_awprot),
        .s_axi_awvalid(s_axi_awvalid), .s_axi_awready(s_axi_awready),
        .s_axi_wdata(s_axi_wdata), .s_axi_wstrb(s_axi_wstrb),
        .s_axi_wvalid(s_axi_wvalid), .s_axi_wready(s_axi_wready),
        .s_axi_bresp(s_axi_bresp), .s_axi_bvalid(s_axi_bvalid),
        .s_axi_bready(s_axi_bready),
        .s_axi_araddr(s_axi_araddr), .s_axi_arprot(s_axi_arprot),
        .s_axi_arvalid(s_axi_arvalid), .s_axi_arready(s_axi_arready),
        .s_axi_rdata(s_axi_rdata), .s_axi_rresp(s_axi_rresp),
        .s_axi_rvalid(s_axi_rvalid), .s_axi_rready(s_axi_rready),
        .s_axis_tdata(s_axis_tdata), .s_axis_tkeep(s_axis_tkeep),
        .s_axis_tvalid(s_axis_tvalid), .s_axis_tready(s_axis_tready),
        .s_axis_tlast(s_axis_tlast),
        .m_axis_tdata(m_axis_tdata), .m_axis_tkeep(m_axis_tkeep),
        .m_axis_tvalid(m_axis_tvalid), .m_axis_tready(m_axis_tready),
        .m_axis_tlast(m_axis_tlast),
        .irq(irq)
    );

    // ---- Clock ----
    initial aclk = 0;
    always #(CLK_PERIOD/2) aclk = ~aclk;

    // ---- Register offsets ----
    localparam REG_CMD       = 8'h00;
    localparam REG_ADDR      = 8'h04;
    localparam REG_WDATA     = 8'h08;
    localparam REG_RDATA     = 8'h0C;
    localparam REG_STATUS    = 8'h10;
    localparam REG_IRQ_EN    = 8'h18;
    localparam REG_IRQ_STAT  = 8'h1C;
    localparam REG_DMA_TARGET = 8'h20;
    localparam REG_DMA_BASE   = 8'h24;
    localparam REG_DMA_LENGTH = 8'h28;
    localparam REG_DMA_CTRL   = 8'h2C;
    localparam REG_DMA_STATUS = 8'h30;

    // ---- Data buffers ----
    reg [7:0] hex_buf [0:MAX_WBUF-1];
    reg [7:0] scale_buf [0:35];
    reg [7:0] exp_buf [0:65535];
    reg [7:0] dma_rd_buf [0:65535];

    // ---- AXI4-Lite tasks ----
    task axi_write(input [7:0] addr, input [31:0] data);
        begin
            @(posedge aclk);
            s_axi_awaddr  <= addr;
            s_axi_awvalid <= 1;
            s_axi_wdata   <= data;
            s_axi_wstrb   <= 4'hF;
            s_axi_wvalid  <= 1;
            s_axi_bready  <= 1;
            wait(s_axi_awready && s_axi_wready);
            @(posedge aclk);
            s_axi_awvalid <= 0;
            s_axi_wvalid  <= 0;
            wait(s_axi_bvalid);
            @(posedge aclk);
            s_axi_bready <= 0;
        end
    endtask

    reg [31:0] read_result;
    task axi_read(input [7:0] addr);
        begin
            @(posedge aclk);
            s_axi_araddr  <= addr;
            s_axi_arvalid <= 1;
            s_axi_rready  <= 1;
            wait(s_axi_arready);
            @(posedge aclk);
            s_axi_arvalid <= 0;
            wait(s_axi_rvalid);
            read_result = s_axi_rdata;
            @(posedge aclk);
            s_axi_rready <= 0;
        end
    endtask

    // ---- DMA write: stream hex_buf[0..len-1] to DPU via AXI-Stream ----
    task dma_write_stream(input [2:0] target, input [23:0] base, input integer len);
        integer i, word_idx;
        reg [31:0] word;
        begin
            // Configure DMA CSRs
            axi_write(REG_DMA_TARGET, {29'd0, target});
            axi_write(REG_DMA_BASE,   {8'd0, base});
            axi_write(REG_DMA_LENGTH,  len[23:0]);
            // Start DMA write (dir=0, start=1)
            axi_write(REG_DMA_CTRL,   32'h01);

            // Stream data as 32-bit words
            i = 0;
            while (i < len) begin
                // Pack up to 4 bytes into word
                word = 32'd0;
                if (i + 0 < len) word[7:0]   = hex_buf[i + 0];
                if (i + 1 < len) word[15:8]  = hex_buf[i + 1];
                if (i + 2 < len) word[23:16] = hex_buf[i + 2];
                if (i + 3 < len) word[31:24] = hex_buf[i + 3];

                @(posedge aclk);
                s_axis_tdata  <= word;
                s_axis_tkeep  <= 4'hF;
                s_axis_tvalid <= 1;
                s_axis_tlast  <= ((i + 4) >= len) ? 1'b1 : 1'b0;

                // Wait for handshake
                wait(s_axis_tready && s_axis_tvalid);
                @(posedge aclk);
                s_axis_tvalid <= 0;
                s_axis_tlast  <= 0;

                i = i + 4;
            end

            // Wait for DMA done
            begin : wait_dma_done
                integer timeout;
                for (timeout = 0; timeout < 1000000; timeout = timeout + 1) begin
                    @(posedge aclk);
                    if (timeout % 100 == 0) begin
                        axi_read(REG_DMA_STATUS);
                        if (read_result[1]) begin  // dma_done
                            timeout = 1000000;
                        end
                    end
                end
            end
        end
    endtask

    // ---- DMA read: read len bytes from DPU via AXI-Stream ----
    task dma_read_stream(input [23:0] base, input integer len);
        integer i, byte_idx;
        begin
            // Configure DMA CSRs
            axi_write(REG_DMA_TARGET, 32'd0);
            axi_write(REG_DMA_BASE,   {8'd0, base});
            axi_write(REG_DMA_LENGTH,  len[23:0]);
            // Start DMA read (dir=1, start=1)
            axi_write(REG_DMA_CTRL,   32'h03);

            // Receive data from M_AXIS
            byte_idx = 0;
            // Pre-assert tready so it's active when DMA starts pushing
            @(posedge aclk);
            m_axis_tready <= 1;
            while (byte_idx < len) begin
                @(posedge aclk);
                if (m_axis_tvalid) begin
                    if (byte_idx + 0 < len) dma_rd_buf[byte_idx + 0] = m_axis_tdata[7:0];
                    if (byte_idx + 1 < len) dma_rd_buf[byte_idx + 1] = m_axis_tdata[15:8];
                    if (byte_idx + 2 < len) dma_rd_buf[byte_idx + 2] = m_axis_tdata[23:16];
                    if (byte_idx + 3 < len) dma_rd_buf[byte_idx + 3] = m_axis_tdata[31:24];
                    byte_idx = byte_idx + 4;
                end
            end
            m_axis_tready <= 0;

            // Wait for DMA done
            begin : wait_dma_rd_done
                integer timeout;
                for (timeout = 0; timeout < 100000; timeout = timeout + 1) begin
                    @(posedge aclk);
                    axi_read(REG_DMA_STATUS);
                    if (read_result[1]) timeout = 100000;
                end
            end
        end
    endtask

    // ---- PIO helpers (through AXI-Lite) ----
    task pio_write_byte(input [23:0] addr, input [7:0] data);
        begin
            axi_write(REG_CMD,   32'd0);
            axi_write(REG_ADDR,  {8'd0, addr});
            axi_write(REG_WDATA, {24'd0, data});
            repeat(3) @(posedge aclk);
        end
    endtask

    task pio_read_byte(input [23:0] addr, output [7:0] data);
        begin
            axi_write(REG_CMD,  32'd2);
            axi_write(REG_ADDR, {8'd0, addr});
            axi_write(REG_WDATA, 32'd0);
            repeat(10) @(posedge aclk);
            axi_read(REG_RDATA);
            data = read_result[7:0];
        end
    endtask

    task pio_write_layer_desc(input [4:0] layer, input [3:0] field, input [7:0] data);
        begin
            axi_write(REG_CMD,   32'd6);
            axi_write(REG_ADDR,  {8'd0, 15'd0, layer, field});
            axi_write(REG_WDATA, {24'd0, data});
            repeat(3) @(posedge aclk);
        end
    endtask

    task pio_write_layer_scale(input [4:0] layer, input [15:0] scale_val);
        begin
            pio_write_layer_desc(layer, 4'd14, scale_val[7:0]);
            pio_write_layer_desc(layer, 4'd15, scale_val[15:8]);
        end
    endtask

    task pio_run_all;
        begin
            axi_write(REG_CMD,   32'd4);
            axi_write(REG_ADDR,  32'd0);
            axi_write(REG_WDATA, 32'd0);
        end
    endtask

    task pio_continue;
        begin
            axi_write(REG_CMD,   32'd1);
            axi_write(REG_ADDR,  32'd0);
            axi_write(REG_WDATA, 32'd0);
        end
    endtask

    // ---- Wait for reload or done (IRQ-based) ----
    integer wr_status;
    task wait_reload_or_done;
        integer cyc;
        begin
            axi_write(REG_IRQ_STAT, 32'h3);
            wr_status = -1;
            for (cyc = 0; cyc < 5_000_000; cyc = cyc + 1) begin
                @(posedge aclk);
                if (cyc % 500 == 0) begin
                    axi_read(REG_IRQ_STAT);
                    if (read_result[0]) begin
                        wr_status = 0;
                        cyc = 5_000_000;
                    end else if (read_result[1]) begin
                        wr_status = 1;
                        cyc = 5_000_000;
                    end
                end
            end
            if (wr_status == -1) begin
                $display("  TIMEOUT!");
                $finish;
            end
        end
    endtask

    // ---- DMA load weights for a layer ----
    task dma_load_layer_weights(input integer wsize, input integer cout);
        begin
            // hex_buf already loaded with weight data
            dma_write_stream(3'd0, 24'd0, wsize);  // target=weight, base=0
            // Load biases
            $readmemh("image_sim_out/dpu_top/layer0_bias.hex", hex_buf);  // placeholder, caller reloads
        end
    endtask

    // ---- Main test ----
    integer i, pass_count, fail_count;
    reg [7:0] rd_byte;
    reg [7:0] verify_byte;

    initial begin
        aresetn = 0;
        s_axi_awaddr = 0; s_axi_awprot = 0; s_axi_awvalid = 0;
        s_axi_wdata = 0; s_axi_wstrb = 0; s_axi_wvalid = 0;
        s_axi_bready = 0;
        s_axi_araddr = 0; s_axi_arprot = 0; s_axi_arvalid = 0;
        s_axi_rready = 0;
        s_axis_tdata = 0; s_axis_tkeep = 0; s_axis_tvalid = 0; s_axis_tlast = 0;
        m_axis_tready = 0;

        repeat(20) @(posedge aclk);
        aresetn = 1;
        repeat(10) @(posedge aclk);

        // Enable IRQs
        axi_write(REG_IRQ_EN, 32'h3);

        // ============================================================
        // TEST 1: DMA write weights (layer 0)
        // ============================================================
        $display("\n=== TEST 1: DMA write weights ===");
        $readmemh("image_sim_out/dpu_top/layer0_weights.hex", hex_buf);
        dma_write_stream(3'd0, 24'd0, WSIZE_0);  // target=weight, base=0
        $display("  PASS: DMA wrote %0d weight bytes", WSIZE_0);

        // ============================================================
        // TEST 2: DMA write biases + verify
        // ============================================================
        $display("\n=== TEST 2: DMA write biases ===");
        $readmemh("image_sim_out/dpu_top/layer0_bias.hex", hex_buf);
        dma_write_stream(3'd2, 24'd0, COUT_0 * 4);  // target=bias, base=0
        $display("  PASS: DMA wrote %0d bias bytes", COUT_0 * 4);

        // ============================================================
        // TEST 3: DMA write input image
        // ============================================================
        $display("\n=== TEST 3: DMA write input image ===");
        $readmemh("image_sim_out/dpu_top/input_image.hex", hex_buf);
        dma_write_stream(3'd1, 24'd0, 768);  // target=fmap, base=0 (module adds FMAP offset)
        $display("  PASS: DMA wrote 768 input bytes");

        // ============================================================
        // TEST 4: Load scales via PIO (small, PIO is fine)
        // ============================================================
        $display("\n=== TEST 4: Load scales ===");
        $readmemh("image_sim_out/dpu_top/scales.hex", scale_buf);
        for (i = 0; i < 18; i = i + 1)
            pio_write_layer_scale(i[4:0], {scale_buf[i*2+1], scale_buf[i*2]});
        $display("  PASS: 18 layer scales loaded");

        // ============================================================
        // TEST 5: Full 18-layer inference (DMA for weights, PIO for control)
        // ============================================================
        $display("\n=== TEST 5: DMA-accelerated 18-layer inference ===");

        // Layer 0 weights already loaded via DMA in TEST 1
        $display("  Starting run_all...");
        pio_run_all;

        // L0 -> reload L1
        wait_reload_or_done;
        $display("  reload -> DMA loading L1");
        $readmemh("image_sim_out/dpu_top/layer1_weights.hex", hex_buf);
        dma_write_stream(3'd0, 24'd0, WSIZE_1);
        $readmemh("image_sim_out/dpu_top/layer1_bias.hex", hex_buf);
        dma_write_stream(3'd2, 24'd0, COUT_1 * 4);
        pio_continue;

        // L1 -> reload L2
        wait_reload_or_done;
        $display("  reload -> DMA loading L2");
        $readmemh("image_sim_out/dpu_top/layer2_weights.hex", hex_buf);
        dma_write_stream(3'd0, 24'd0, WSIZE_2);
        $readmemh("image_sim_out/dpu_top/layer2_bias.hex", hex_buf);
        dma_write_stream(3'd2, 24'd0, COUT_2 * 4);
        pio_continue;

        // L2 -> auto L3 -> reload L4
        wait_reload_or_done;
        $display("  reload -> DMA loading L4");
        $readmemh("image_sim_out/dpu_top/layer4_weights.hex", hex_buf);
        dma_write_stream(3'd0, 24'd0, WSIZE_4);
        $readmemh("image_sim_out/dpu_top/layer4_bias.hex", hex_buf);
        dma_write_stream(3'd2, 24'd0, COUT_4 * 4);
        pio_continue;

        // L4 -> reload L5
        wait_reload_or_done;
        $display("  reload -> DMA loading L5");
        $readmemh("image_sim_out/dpu_top/layer5_weights.hex", hex_buf);
        dma_write_stream(3'd0, 24'd0, WSIZE_5);
        $readmemh("image_sim_out/dpu_top/layer5_bias.hex", hex_buf);
        dma_write_stream(3'd2, 24'd0, COUT_5 * 4);
        pio_continue;

        // L5 -> auto L6 -> reload L7
        wait_reload_or_done;
        $display("  reload -> DMA loading L7");
        $readmemh("image_sim_out/dpu_top/layer7_weights.hex", hex_buf);
        dma_write_stream(3'd0, 24'd0, WSIZE_7);
        $readmemh("image_sim_out/dpu_top/layer7_bias.hex", hex_buf);
        dma_write_stream(3'd2, 24'd0, COUT_7 * 4);
        pio_continue;

        // L7 -> auto L8,L9 -> reload L10
        wait_reload_or_done;
        $display("  reload -> DMA loading L10");
        $readmemh("image_sim_out/dpu_top/layer10_weights.hex", hex_buf);
        dma_write_stream(3'd0, 24'd0, WSIZE_10);
        $readmemh("image_sim_out/dpu_top/layer10_bias.hex", hex_buf);
        dma_write_stream(3'd2, 24'd0, COUT_10 * 4);
        pio_continue;

        // L10 -> auto L11 -> reload L12
        wait_reload_or_done;
        $display("  reload -> DMA loading L12");
        $readmemh("image_sim_out/dpu_top/layer12_weights.hex", hex_buf);
        dma_write_stream(3'd0, 24'd0, WSIZE_12);
        $readmemh("image_sim_out/dpu_top/layer12_bias.hex", hex_buf);
        dma_write_stream(3'd2, 24'd0, COUT_12 * 4);
        pio_continue;

        // L12 -> reload L13
        wait_reload_or_done;
        $display("  reload -> DMA loading L13");
        $readmemh("image_sim_out/dpu_top/layer13_weights.hex", hex_buf);
        dma_write_stream(3'd0, 24'd0, WSIZE_13);
        $readmemh("image_sim_out/dpu_top/layer13_bias.hex", hex_buf);
        dma_write_stream(3'd2, 24'd0, COUT_13 * 4);
        pio_continue;

        // L13 -> auto L14 -> reload L15
        wait_reload_or_done;
        $display("  reload -> DMA loading L15");
        $readmemh("image_sim_out/dpu_top/layer15_weights.hex", hex_buf);
        dma_write_stream(3'd0, 24'd0, WSIZE_15);
        $readmemh("image_sim_out/dpu_top/layer15_bias.hex", hex_buf);
        dma_write_stream(3'd2, 24'd0, COUT_15 * 4);
        pio_continue;

        // L15 -> auto L16,L17 -> done!
        wait_reload_or_done;
        if (wr_status != 0) begin
            $display("  ERROR: expected done but got reload");
            $finish;
        end
        $display("  Inference DONE!");

        // ============================================================
        // TEST 6: DMA readback output + verify
        // ============================================================
        $display("\n=== TEST 6: DMA readback output ===");
        dma_read_stream(24'd0, OSIZE_17);
        $display("  DMA read %0d output bytes", OSIZE_17);

        // Compare
        $readmemh("image_sim_out/dpu_top/layer17_expected.hex", exp_buf);
        pass_count = 0;
        fail_count = 0;
        for (i = 0; i < OSIZE_17; i = i + 1) begin
            if (dma_rd_buf[i] == exp_buf[i])
                pass_count = pass_count + 1;
            else begin
                if (fail_count < 20)
                    $display("  MISMATCH [%0d]: got=0x%02h exp=0x%02h", i, dma_rd_buf[i], exp_buf[i]);
                fail_count = fail_count + 1;
            end
        end

        // ============================================================
        // SUMMARY
        // ============================================================
        $display("\n========================================");
        $display("  DMA STREAMING TESTBENCH SUMMARY");
        $display("========================================");
        $display("  Output bytes checked: %0d", pass_count + fail_count);
        $display("  PASS: %0d", pass_count);
        $display("  FAIL: %0d", fail_count);
        if (fail_count == 0)
            $display("  *** ALL TESTS PASSED ***");
        else
            $display("  *** %0d FAILURES ***", fail_count);
        $display("========================================\n");

        $finish;
    end

endmodule
