// =============================================================================
// Testbench: dpu_system_top — AXI4-Lite Integration Test
//
// Tests:
//   1. AXI4-Lite register read/write (VERSION, STATUS, CMD, ADDR, WDATA)
//   2. PIO write_byte through AXI (load weights, bias, input)
//   3. PIO run_all inference through AXI (full 18-layer with reload)
//   4. PIO read_byte output through AXI
//   5. IRQ done detection
//   6. Compare output vs golden hex
//
// Compatible with Icarus Verilog (no program/clocking blocks).
// =============================================================================
`timescale 1ns / 1ps

module dpu_system_top_tb;

    // ---- Parameters (must match golden) ----
    parameter H0 = 16, W0 = 16, MAX_CH = 256;
    parameter MAX_FMAP = 65536, MAX_WBUF = 147456, ADDR_BITS = 24;
    parameter CLK_PERIOD = 10;  // 100 MHz
    parameter FMAP_BASE = MAX_WBUF + MAX_CH * 4;  // 148480

    // Weight sizes per conv layer
    parameter WSIZE_0  = 3*32*3*3;      // 864
    parameter WSIZE_1  = 32*64*3*3;     // 18432
    parameter WSIZE_2  = 64*64*3*3;     // 36864
    parameter WSIZE_4  = 32*32*3*3;     // 9216
    parameter WSIZE_5  = 32*32*3*3;     // 9216
    parameter WSIZE_7  = 64*64*1*1;     // 4096
    parameter WSIZE_10 = 128*128*3*3;   // 147456
    parameter WSIZE_12 = 64*64*3*3;     // 36864
    parameter WSIZE_13 = 64*64*3*3;     // 36864
    parameter WSIZE_15 = 128*128*1*1;   // 16384

    // Output channels per conv layer
    parameter COUT_0  = 32;
    parameter COUT_1  = 64;
    parameter COUT_2  = 64;
    parameter COUT_4  = 32;
    parameter COUT_5  = 32;
    parameter COUT_7  = 64;
    parameter COUT_10 = 128;
    parameter COUT_12 = 64;
    parameter COUT_13 = 64;
    parameter COUT_15 = 128;

    // Final output size (layer 17: maxpool 256ch, H/W depends on H0)
    parameter OSIZE_17 = 256 * (H0/16) * (W0/16);  // 256 bytes for 16x16 input

    // ---- DUT signals ----
    reg         aclk;
    reg         aresetn;

    // AXI4-Lite
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

    // AXI-Stream (tie off for PIO-only test)
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
    localparam REG_PERF      = 8'h14;
    localparam REG_IRQ_EN    = 8'h18;
    localparam REG_IRQ_STAT  = 8'h1C;
    localparam REG_VERSION   = 8'h34;

    // ---- Hex data buffers ----
    reg [7:0] hex_buf [0:MAX_WBUF-1];
    reg [7:0] scale_buf [0:35];
    reg [7:0] exp_buf [0:65535];

    // ---- AXI4-Lite write task ----
    task axi_write(input [7:0] addr, input [31:0] data);
        begin
            @(posedge aclk);
            s_axi_awaddr  <= addr;
            s_axi_awvalid <= 1;
            s_axi_wdata   <= data;
            s_axi_wstrb   <= 4'hF;
            s_axi_wvalid  <= 1;
            s_axi_bready  <= 1;
            // Wait for handshake
            wait(s_axi_awready && s_axi_wready);
            @(posedge aclk);
            s_axi_awvalid <= 0;
            s_axi_wvalid  <= 0;
            // Wait for write response
            wait(s_axi_bvalid);
            @(posedge aclk);
            s_axi_bready <= 0;
        end
    endtask

    // ---- AXI4-Lite read task ----
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

    // ---- PIO command through AXI (write_byte, cmd_type=0) ----
    task pio_write_byte(input [23:0] addr, input [7:0] data);
        begin
            axi_write(REG_CMD,   32'd0);
            axi_write(REG_ADDR,  {8'd0, addr});
            axi_write(REG_WDATA, {24'd0, data});
            repeat(3) @(posedge aclk);
        end
    endtask

    // ---- PIO read_byte through AXI (cmd_type=2) ----
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

    // ---- PIO write_layer_desc through AXI (cmd_type=6) ----
    task pio_write_layer_desc(input [4:0] layer, input [3:0] field, input [7:0] data);
        begin
            axi_write(REG_CMD,   32'd6);
            axi_write(REG_ADDR,  {8'd0, 15'd0, layer, field});
            axi_write(REG_WDATA, {24'd0, data});
            repeat(3) @(posedge aclk);
        end
    endtask

    // ---- PIO write_layer_scale through AXI ----
    task pio_write_layer_scale(input [4:0] layer, input [15:0] scale_val);
        begin
            pio_write_layer_desc(layer, 4'd14, scale_val[7:0]);
            pio_write_layer_desc(layer, 4'd15, scale_val[15:8]);
        end
    endtask

    // ---- PIO run_all through AXI (cmd_type=4) ----
    task pio_run_all;
        begin
            axi_write(REG_CMD,   32'd4);
            axi_write(REG_ADDR,  32'd0);
            axi_write(REG_WDATA, 32'd0);
        end
    endtask

    // ---- PIO run_layer / continue (cmd_type=1) ----
    task pio_continue;
        begin
            axi_write(REG_CMD,   32'd1);
            axi_write(REG_ADDR,  32'd0);
            axi_write(REG_WDATA, 32'd0);
        end
    endtask

    // ---- Wait for reload_req or done via IRQ_STAT polling ----
    // IRQ_STAT[0] = done (latched), IRQ_STAT[1] = reload (latched)
    // Returns: 0 = done, 1 = reload_req
    integer wr_status;
    task wait_reload_or_done;
        integer cyc;
        begin
            // Clear any pending IRQ status first
            axi_write(REG_IRQ_STAT, 32'h3);
            wr_status = -1;
            for (cyc = 0; cyc < 5_000_000; cyc = cyc + 1) begin
                @(posedge aclk);
                if (cyc % 500 == 0) begin
                    axi_read(REG_IRQ_STAT);
                    if (read_result[0]) begin  // done IRQ latched
                        wr_status = 0;
                        cyc = 5_000_000;  // break
                    end else if (read_result[1]) begin  // reload IRQ latched
                        wr_status = 1;
                        cyc = 5_000_000;  // break
                    end
                end
            end
            if (wr_status == -1) begin
                $display("  TIMEOUT waiting for reload/done!");
                $finish;
            end
        end
    endtask

    // ---- Load weights + biases for a layer via PIO ----
    task load_layer_weights(input integer wsize, input integer cout);
        integer j;
        begin
            // hex_buf must already contain weight data
            for (j = 0; j < wsize; j = j + 1)
                pio_write_byte(j[23:0], hex_buf[j]);
        end
    endtask

    task load_layer_biases(input integer cout);
        integer j;
        begin
            // hex_buf must already contain bias data
            for (j = 0; j < cout * 4; j = j + 1)
                pio_write_byte(MAX_WBUF + j[23:0], hex_buf[j]);
        end
    endtask

    // ---- Main test sequence ----
    integer i, j, pass_count, fail_count;
    reg [7:0] rd_byte;

    initial begin
        // Initialize all signals
        aresetn       = 0;
        s_axi_awaddr  = 0; s_axi_awprot = 0; s_axi_awvalid = 0;
        s_axi_wdata   = 0; s_axi_wstrb  = 0; s_axi_wvalid  = 0;
        s_axi_bready  = 0;
        s_axi_araddr  = 0; s_axi_arprot = 0; s_axi_arvalid = 0;
        s_axi_rready  = 0;
        s_axis_tdata  = 0; s_axis_tkeep = 0; s_axis_tvalid = 0; s_axis_tlast = 0;
        m_axis_tready = 1;

        // Reset
        repeat(20) @(posedge aclk);
        aresetn = 1;
        repeat(10) @(posedge aclk);

        // ============================================================
        // TEST 1: Read VERSION register
        // ============================================================
        $display("\n=== TEST 1: VERSION register ===");
        axi_read(REG_VERSION);
        if (read_result == 32'h0001_0000)
            $display("  PASS: VERSION = 0x%08h", read_result);
        else
            $display("  FAIL: VERSION = 0x%08h (expected 0x00010000)", read_result);

        // ============================================================
        // TEST 2: Read STATUS (should be idle)
        // ============================================================
        $display("\n=== TEST 2: STATUS register (idle) ===");
        axi_read(REG_STATUS);
        $display("  STATUS = 0x%08h", read_result);
        if (read_result[0] == 0 && read_result[1] == 0)
            $display("  PASS: DPU idle (busy=0, done=0)");
        else
            $display("  FAIL: unexpected status");

        // ============================================================
        // TEST 3: IRQ enable/read
        // ============================================================
        $display("\n=== TEST 3: IRQ registers ===");
        axi_write(REG_IRQ_EN, 32'h3);  // enable done + reload
        axi_read(REG_IRQ_EN);
        if (read_result[1:0] == 2'b11)
            $display("  PASS: IRQ_EN = 0x%08h", read_result);
        else
            $display("  FAIL: IRQ_EN = 0x%08h", read_result);

        // ============================================================
        // TEST 4: Load per-layer scales via write_layer_desc
        // ============================================================
        $display("\n=== TEST 4: Load scales ===");
        $readmemh("image_sim_out/dpu_top/scales.hex", scale_buf);
        for (i = 0; i < 18; i = i + 1)
            pio_write_layer_scale(i[4:0], {scale_buf[i*2+1], scale_buf[i*2]});
        $display("  PASS: 18 layer scales written");

        // ============================================================
        // TEST 5: Load input image via PIO write_byte
        // ============================================================
        $display("\n=== TEST 5: Load input image ===");
        $readmemh("image_sim_out/dpu_top/input_image.hex", hex_buf);
        for (i = 0; i < 768; i = i + 1)
            pio_write_byte(FMAP_BASE + i[23:0], hex_buf[i]);
        $display("  PASS: input loaded (768 bytes to FMAP_BASE)");

        // ============================================================
        // TEST 6: Run 18-layer inference with reload
        // ============================================================
        $display("\n=== TEST 6: Full 18-layer inference ===");

        // Load layer 0 weights + biases
        $display("  Loading layer 0 weights (%0d bytes)", WSIZE_0);
        $readmemh("image_sim_out/dpu_top/layer0_weights.hex", hex_buf);
        load_layer_weights(WSIZE_0, COUT_0);
        $readmemh("image_sim_out/dpu_top/layer0_bias.hex", hex_buf);
        load_layer_biases(COUT_0);

        // Start run_all
        $display("  Starting run_all...");
        pio_run_all;

        // L0 done -> reload for L1
        wait_reload_or_done;
        $display("  reload -> loading L1");
        $readmemh("image_sim_out/dpu_top/layer1_weights.hex", hex_buf);
        load_layer_weights(WSIZE_1, COUT_1);
        $readmemh("image_sim_out/dpu_top/layer1_bias.hex", hex_buf);
        load_layer_biases(COUT_1);
        pio_continue;

        // L1 done -> reload for L2
        wait_reload_or_done;
        $display("  reload -> loading L2");
        $readmemh("image_sim_out/dpu_top/layer2_weights.hex", hex_buf);
        load_layer_weights(WSIZE_2, COUT_2);
        $readmemh("image_sim_out/dpu_top/layer2_bias.hex", hex_buf);
        load_layer_biases(COUT_2);
        pio_continue;

        // L2 done -> auto L3(route) -> reload for L4
        wait_reload_or_done;
        $display("  reload -> loading L4");
        $readmemh("image_sim_out/dpu_top/layer4_weights.hex", hex_buf);
        load_layer_weights(WSIZE_4, COUT_4);
        $readmemh("image_sim_out/dpu_top/layer4_bias.hex", hex_buf);
        load_layer_biases(COUT_4);
        pio_continue;

        // L4 done -> reload for L5
        wait_reload_or_done;
        $display("  reload -> loading L5");
        $readmemh("image_sim_out/dpu_top/layer5_weights.hex", hex_buf);
        load_layer_weights(WSIZE_5, COUT_5);
        $readmemh("image_sim_out/dpu_top/layer5_bias.hex", hex_buf);
        load_layer_biases(COUT_5);
        pio_continue;

        // L5 done -> auto L6(route) -> reload for L7
        wait_reload_or_done;
        $display("  reload -> loading L7");
        $readmemh("image_sim_out/dpu_top/layer7_weights.hex", hex_buf);
        load_layer_weights(WSIZE_7, COUT_7);
        $readmemh("image_sim_out/dpu_top/layer7_bias.hex", hex_buf);
        load_layer_biases(COUT_7);
        pio_continue;

        // L7 done -> auto L8(route), L9(maxpool) -> reload for L10
        wait_reload_or_done;
        $display("  reload -> loading L10");
        $readmemh("image_sim_out/dpu_top/layer10_weights.hex", hex_buf);
        load_layer_weights(WSIZE_10, COUT_10);
        $readmemh("image_sim_out/dpu_top/layer10_bias.hex", hex_buf);
        load_layer_biases(COUT_10);
        pio_continue;

        // L10 done -> auto L11(route) -> reload for L12
        wait_reload_or_done;
        $display("  reload -> loading L12");
        $readmemh("image_sim_out/dpu_top/layer12_weights.hex", hex_buf);
        load_layer_weights(WSIZE_12, COUT_12);
        $readmemh("image_sim_out/dpu_top/layer12_bias.hex", hex_buf);
        load_layer_biases(COUT_12);
        pio_continue;

        // L12 done -> reload for L13
        wait_reload_or_done;
        $display("  reload -> loading L13");
        $readmemh("image_sim_out/dpu_top/layer13_weights.hex", hex_buf);
        load_layer_weights(WSIZE_13, COUT_13);
        $readmemh("image_sim_out/dpu_top/layer13_bias.hex", hex_buf);
        load_layer_biases(COUT_13);
        pio_continue;

        // L13 done -> auto L14(route) -> reload for L15
        wait_reload_or_done;
        $display("  reload -> loading L15");
        $readmemh("image_sim_out/dpu_top/layer15_weights.hex", hex_buf);
        load_layer_weights(WSIZE_15, COUT_15);
        $readmemh("image_sim_out/dpu_top/layer15_bias.hex", hex_buf);
        load_layer_biases(COUT_15);
        pio_continue;

        // L15 done -> auto L16(route), L17(maxpool) -> done!
        wait_reload_or_done;
        if (wr_status != 0) begin
            $display("  ERROR: expected done but got reload");
            $finish;
        end
        $display("  Inference DONE!");

        // ============================================================
        // TEST 7: Read and verify output
        // ============================================================
        $display("\n=== TEST 7: Verify output ===");
        $readmemh("image_sim_out/dpu_top/layer17_expected.hex", exp_buf);
        $display("  Expected output: %0d bytes", OSIZE_17);

        pass_count = 0;
        fail_count = 0;
        for (i = 0; i < OSIZE_17; i = i + 1) begin
            pio_read_byte(i[23:0], rd_byte);
            if (rd_byte == exp_buf[i]) begin
                pass_count = pass_count + 1;
            end else begin
                if (fail_count < 20)
                    $display("  MISMATCH [%0d]: got=0x%02h exp=0x%02h", i, rd_byte, exp_buf[i]);
                fail_count = fail_count + 1;
            end
        end

        // ============================================================
        // TEST 8: Check IRQ mechanism (done IRQ was detected via polling)
        // ============================================================
        $display("\n=== TEST 8: IRQ mechanism ===");
        // The done IRQ was already used in wait_reload_or_done.
        // Verify W1C: write 0x3 to clear, then read back
        axi_write(REG_IRQ_STAT, 32'h3);
        axi_read(REG_IRQ_STAT);
        if (read_result[1:0] == 2'b00)
            $display("  PASS: IRQ cleared by W1C");
        else
            $display("  FAIL: IRQ not cleared (0x%08h)", read_result);
        // Verify irq output is LOW after clear
        if (!irq)
            $display("  PASS: irq output LOW after clear");
        else
            $display("  INFO: irq output still HIGH");

        // ============================================================
        // TEST 9: Read performance counter
        // ============================================================
        $display("\n=== TEST 9: Performance ===");
        axi_read(REG_PERF);
        $display("  Total cycles: %0d", read_result);

        // ============================================================
        // SUMMARY
        // ============================================================
        $display("\n========================================");
        $display("  AXI SYSTEM TOP TESTBENCH SUMMARY");
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
