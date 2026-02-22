// =============================================================================
// Testbench: dpu_system_top — 36-Layer AXI4-Lite + AXI-Stream System Test
//
// Exercises the COMPLETE DPU through AXI interfaces only, exactly as the
// CPU would control it on a real FPGA (Zynq PS→PL):
//
//   1. AXI4-Lite: register R/W, PIO commands (run_all, continue, set_layer,
//                 write_scale, write_layer_desc)
//   2. AXI-Stream DMA: bulk weight/bias/input loading (S_AXIS → DPU)
//   3. IRQ polling: wait for reload_req / done via IRQ_STAT register
//   4. PIO read_byte: output readback and verification
//
// NO hierarchical DUT references are used for data loading — everything
// goes through the AXI bus, matching the real deployment flow.
//
// Compatible with Icarus Verilog (no program/clocking blocks).
// =============================================================================
`timescale 1ns / 1ps

module dpu_system_top_36layer_tb;

    // ---- Parameters (must match golden) ----
    parameter H0 = 32, W0 = 32, MAX_CH = 512;
    parameter MAX_FMAP = 65536, MAX_WBUF = 2400000, ADDR_BITS = 24;
    parameter NUM_LAYERS = 36;
    parameter CLK_PERIOD = 10;  // 100 MHz

    // Final output: layer 35 = 255ch × (H0/16) × (W0/16) = 255*2*2 = 1020
    parameter OSIZE_35 = 255 * (H0/16) * (W0/16);
    parameter INPUT_SIZE = 3 * H0 * W0;  // 3072

    // Number of conv layers
    parameter NUM_CONV = 21;

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

    // AXI-Stream
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
        .MAX_FMAP(MAX_FMAP), .MAX_WBUF(MAX_WBUF), .ADDR_BITS(ADDR_BITS),
        .NUM_LAYERS(NUM_LAYERS)
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
    localparam REG_DMA_TARGET= 8'h20;
    localparam REG_DMA_BASE  = 8'h24;
    localparam REG_DMA_LEN   = 8'h28;
    localparam REG_DMA_CTRL  = 8'h2C;
    localparam REG_DMA_STAT  = 8'h30;
    localparam REG_VERSION   = 8'h34;

    // ---- Hex data buffers (TB-local) ----
    reg [7:0] hex_buf [0:2399999];   // large enough for biggest weight file
    reg [7:0] scale_buf [0:71];      // 36 layers × 2 bytes
    reg [7:0] exp_buf [0:65535];     // expected output

    // ---- Conv layer lookup table ----
    // Conv layer indices (21 conv layers in order)
    integer conv_idx [0:20];
    integer conv_wsize [0:20];
    integer conv_cout [0:20];

    initial begin
        //                 layer  wsize            cout
        conv_idx[0]  = 0;  conv_wsize[0]  = 3*32*3*3;        conv_cout[0]  = 32;
        conv_idx[1]  = 1;  conv_wsize[1]  = 32*64*3*3;       conv_cout[1]  = 64;
        conv_idx[2]  = 2;  conv_wsize[2]  = 64*64*3*3;       conv_cout[2]  = 64;
        conv_idx[3]  = 4;  conv_wsize[3]  = 32*32*3*3;       conv_cout[3]  = 32;
        conv_idx[4]  = 5;  conv_wsize[4]  = 32*32*3*3;       conv_cout[4]  = 32;
        conv_idx[5]  = 7;  conv_wsize[5]  = 64*64*1*1;       conv_cout[5]  = 64;
        conv_idx[6]  = 10; conv_wsize[6]  = 128*128*3*3;     conv_cout[6]  = 128;
        conv_idx[7]  = 12; conv_wsize[7]  = 64*64*3*3;       conv_cout[7]  = 64;
        conv_idx[8]  = 13; conv_wsize[8]  = 64*64*3*3;       conv_cout[8]  = 64;
        conv_idx[9]  = 15; conv_wsize[9]  = 128*128*1*1;     conv_cout[9]  = 128;
        conv_idx[10] = 18; conv_wsize[10] = 256*256*3*3;     conv_cout[10] = 256;
        conv_idx[11] = 20; conv_wsize[11] = 128*128*3*3;     conv_cout[11] = 128;
        conv_idx[12] = 21; conv_wsize[12] = 128*128*3*3;     conv_cout[12] = 128;
        conv_idx[13] = 23; conv_wsize[13] = 256*256*1*1;     conv_cout[13] = 256;
        conv_idx[14] = 26; conv_wsize[14] = 512*512*3*3;     conv_cout[14] = 512;
        conv_idx[15] = 27; conv_wsize[15] = 512*256*1*1;     conv_cout[15] = 256;
        conv_idx[16] = 28; conv_wsize[16] = 256*512*3*3;     conv_cout[16] = 512;
        conv_idx[17] = 29; conv_wsize[17] = 512*255*1*1;     conv_cout[17] = 255;
        conv_idx[18] = 31; conv_wsize[18] = 256*128*1*1;     conv_cout[18] = 128;
        conv_idx[19] = 34; conv_wsize[19] = 384*256*3*3;     conv_cout[19] = 256;
        conv_idx[20] = 35; conv_wsize[20] = 256*255*1*1;     conv_cout[20] = 255;
    end

    // =====================================================================
    // AXI4-Lite write task
    // =====================================================================
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

    // =====================================================================
    // AXI4-Lite read task
    // =====================================================================
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

    // =====================================================================
    // PIO write_layer_desc through AXI (cmd_type=6)
    // =====================================================================
    task pio_write_layer_desc(input [5:0] layer, input [3:0] field, input [7:0] data);
        begin
            axi_write(REG_CMD,   32'd6);
            axi_write(REG_ADDR,  {8'd0, 14'd0, layer, field});
            axi_write(REG_WDATA, {24'd0, data});
            repeat(3) @(posedge aclk);
        end
    endtask

    // =====================================================================
    // PIO write per-layer scale (fields 14 and 15 of write_layer_desc)
    // =====================================================================
    task pio_write_layer_scale(input [5:0] layer, input [15:0] scale_val);
        begin
            pio_write_layer_desc(layer, 4'd14, scale_val[7:0]);
            pio_write_layer_desc(layer, 4'd15, scale_val[15:8]);
        end
    endtask

    // =====================================================================
    // PIO run_all through AXI (cmd_type=4)
    // =====================================================================
    task pio_run_all;
        begin
            axi_write(REG_CMD,   32'd4);
            axi_write(REG_ADDR,  32'd0);
            axi_write(REG_WDATA, 32'd0);
        end
    endtask

    // =====================================================================
    // PIO continue through AXI (cmd_type=1)
    // =====================================================================
    task pio_continue;
        begin
            axi_write(REG_CMD,   32'd1);
            axi_write(REG_ADDR,  32'd0);
            axi_write(REG_WDATA, 32'd0);
        end
    endtask

    // =====================================================================
    // PIO read_byte through AXI (cmd_type=2)
    // =====================================================================
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

    // =====================================================================
    // Wait for reload_req or done via STATUS register polling
    // Returns: 0 = done, 1 = reload_req
    //
    // STATUS layout: [0]=busy [1]=done(pulse) [13:8]=layer [16]=reload_req
    //
    // Uses LEVEL signals: reload_req stays HIGH while DPU is in
    // S_LAYER_RELOAD; busy goes LOW after all layers complete.
    // This avoids edge-detection issues with IRQ_STAT.
    // =====================================================================
    integer wr_status;
    task wait_reload_or_done;
        integer cyc;
        begin
            wr_status = -1;
            for (cyc = 0; cyc < 50_000_000; cyc = cyc + 1) begin
                @(posedge aclk);
                if (cyc % 200 == 0) begin
                    axi_read(REG_STATUS);
                    if (read_result[16]) begin  // reload_req LEVEL
                        wr_status = 1;
                        cyc = 50_000_000;  // break
                    end else if (!read_result[0]) begin  // !busy = done
                        wr_status = 0;
                        cyc = 50_000_000;  // break
                    end
                end
            end
            if (wr_status == -1) begin
                $display("  TIMEOUT waiting for reload/done!");
                axi_read(REG_STATUS);
                $display("  STATUS=0x%08h (busy=%b done=%b layer=%0d reload=%b)",
                         read_result, read_result[0], read_result[1],
                         (read_result >> 8) & 6'h3F, read_result[16]);
                $finish;
            end
        end
    endtask

    // =====================================================================
    // DMA write: stream data from hex_buf to DPU via AXI-Stream
    //   target: 0=weight, 1=fmap_input, 2=bias
    //   base:   starting address within the target buffer
    //   length: number of bytes to transfer
    // =====================================================================
    task dma_stream_write(input [2:0] target, input [23:0] base, input integer length);
        integer j, word_count, remaining;
        integer dma_wait_ok;
        begin
            // Program DMA CSRs
            axi_write(REG_DMA_TARGET, {29'd0, target});
            axi_write(REG_DMA_BASE,   {8'd0, base});
            axi_write(REG_DMA_LEN,    length[23:0]);
            axi_write(REG_DMA_CTRL,   32'h01);  // start, direction=write(0)

            // Wait a few cycles for DMA FSM to start
            repeat(5) @(posedge aclk);

            // Check DMA started
            axi_read(REG_DMA_STAT);
            $display("      DMA start: target=%0d base=%0d len=%0d STAT=0x%08h (t=%0t)",
                     target, base, length, read_result, $time);

            // Stream data via S_AXIS
            word_count = (length + 3) / 4;
            for (j = 0; j < word_count; j = j + 1) begin
                // Pack 4 bytes (LE) into 32-bit word
                s_axis_tdata <= {hex_buf[j*4+3], hex_buf[j*4+2],
                                 hex_buf[j*4+1], hex_buf[j*4]};
                s_axis_tkeep <= 4'hF;
                s_axis_tvalid <= 1'b1;
                s_axis_tlast <= (j == word_count - 1) ? 1'b1 : 1'b0;
                @(posedge aclk);
                while (!s_axis_tready) @(posedge aclk);
            end
            s_axis_tvalid <= 1'b0;
            s_axis_tlast <= 1'b0;

            $display("      DMA stream done: %0d words sent (t=%0t)", word_count, $time);

            // Wait for DMA not-busy (dma_done is a 1-cycle pulse, easily missed;
            // instead check !busy which is a stable level after completion)
            dma_wait_ok = 0;
            begin : wait_dma_block
                integer timeout;
                for (timeout = 0; timeout < length * 20 + 1000; timeout = timeout + 1) begin
                    @(posedge aclk);
                    if (timeout % 10 == 0) begin
                        axi_read(REG_DMA_STAT);
                        if (!read_result[0]) begin  // !dma_busy
                            dma_wait_ok = 1;
                            timeout = length * 20 + 1000;  // break
                        end
                    end
                end
            end
            axi_read(REG_DMA_STAT);
            if (dma_wait_ok)
                $display("      DMA complete: STAT=0x%08h (t=%0t)", read_result, $time);
            else begin
                $display("      DMA TIMEOUT! STAT=0x%08h (t=%0t)", read_result, $time);
                // Debug: probe DMA and DPU internals via hierarchy
                $display("      DEBUG DMA: state=%0d byte_cnt=%0d/%0d word_byte=%0d axis_buf_valid=%b",
                         u_dut.u_dma.state, u_dut.u_dma.byte_cnt,
                         u_dut.u_dma.dma_length, u_dut.u_dma.word_byte,
                         u_dut.u_dma.axis_buf_valid);
                $display("      DEBUG DMA: cmd_valid=%b cmd_ready=%b cmd_type=%0d cmd_addr=%0d",
                         u_dut.u_dma.cmd_valid, u_dut.u_dma.cmd_ready,
                         u_dut.u_dma.cmd_type, u_dut.u_dma.cmd_addr);
                $display("      DEBUG DPU: state=%0d cmd_ready=%b busy=%b",
                         u_dut.u_dpu.state, u_dut.u_dpu.cmd_ready, u_dut.u_dpu.busy);
                $display("      DEBUG ARB: dma_busy_flag=%b lite_cmd_ready=%b dma_cmd_ready=%b",
                         u_dut.dma_busy_flag, u_dut.lite_cmd_ready, u_dut.dma_cmd_ready);
            end
        end
    endtask

    // =====================================================================
    // Load weights+biases for a conv layer via DMA
    // hex_buf is loaded from file, then streamed to DPU
    // =====================================================================
    task load_conv_via_dma(input integer conv_table_idx);
        integer li, ws, co;
        reg [255:0] wfile;
        reg [255:0] bfile;
        begin
            li = conv_idx[conv_table_idx];
            ws = conv_wsize[conv_table_idx];
            co = conv_cout[conv_table_idx];

            // Load weight hex file into hex_buf
            case (li)
                0:  $readmemh("image_sim_out/dpu_top_37/layer0_weights.hex", hex_buf);
                1:  $readmemh("image_sim_out/dpu_top_37/layer1_weights.hex", hex_buf);
                2:  $readmemh("image_sim_out/dpu_top_37/layer2_weights.hex", hex_buf);
                4:  $readmemh("image_sim_out/dpu_top_37/layer4_weights.hex", hex_buf);
                5:  $readmemh("image_sim_out/dpu_top_37/layer5_weights.hex", hex_buf);
                7:  $readmemh("image_sim_out/dpu_top_37/layer7_weights.hex", hex_buf);
                10: $readmemh("image_sim_out/dpu_top_37/layer10_weights.hex", hex_buf);
                12: $readmemh("image_sim_out/dpu_top_37/layer12_weights.hex", hex_buf);
                13: $readmemh("image_sim_out/dpu_top_37/layer13_weights.hex", hex_buf);
                15: $readmemh("image_sim_out/dpu_top_37/layer15_weights.hex", hex_buf);
                18: $readmemh("image_sim_out/dpu_top_37/layer18_weights.hex", hex_buf);
                20: $readmemh("image_sim_out/dpu_top_37/layer20_weights.hex", hex_buf);
                21: $readmemh("image_sim_out/dpu_top_37/layer21_weights.hex", hex_buf);
                23: $readmemh("image_sim_out/dpu_top_37/layer23_weights.hex", hex_buf);
                26: $readmemh("image_sim_out/dpu_top_37/layer26_weights.hex", hex_buf);
                27: $readmemh("image_sim_out/dpu_top_37/layer27_weights.hex", hex_buf);
                28: $readmemh("image_sim_out/dpu_top_37/layer28_weights.hex", hex_buf);
                29: $readmemh("image_sim_out/dpu_top_37/layer29_weights.hex", hex_buf);
                31: $readmemh("image_sim_out/dpu_top_37/layer31_weights.hex", hex_buf);
                34: $readmemh("image_sim_out/dpu_top_37/layer34_weights.hex", hex_buf);
                35: $readmemh("image_sim_out/dpu_top_37/layer35_weights.hex", hex_buf);
            endcase

            // Stream weights to DPU via DMA (target=0: weight_buf)
            dma_stream_write(3'd0, 24'd0, ws);

            // Load bias hex file into hex_buf
            case (li)
                0:  $readmemh("image_sim_out/dpu_top_37/layer0_bias.hex", hex_buf);
                1:  $readmemh("image_sim_out/dpu_top_37/layer1_bias.hex", hex_buf);
                2:  $readmemh("image_sim_out/dpu_top_37/layer2_bias.hex", hex_buf);
                4:  $readmemh("image_sim_out/dpu_top_37/layer4_bias.hex", hex_buf);
                5:  $readmemh("image_sim_out/dpu_top_37/layer5_bias.hex", hex_buf);
                7:  $readmemh("image_sim_out/dpu_top_37/layer7_bias.hex", hex_buf);
                10: $readmemh("image_sim_out/dpu_top_37/layer10_bias.hex", hex_buf);
                12: $readmemh("image_sim_out/dpu_top_37/layer12_bias.hex", hex_buf);
                13: $readmemh("image_sim_out/dpu_top_37/layer13_bias.hex", hex_buf);
                15: $readmemh("image_sim_out/dpu_top_37/layer15_bias.hex", hex_buf);
                18: $readmemh("image_sim_out/dpu_top_37/layer18_bias.hex", hex_buf);
                20: $readmemh("image_sim_out/dpu_top_37/layer20_bias.hex", hex_buf);
                21: $readmemh("image_sim_out/dpu_top_37/layer21_bias.hex", hex_buf);
                23: $readmemh("image_sim_out/dpu_top_37/layer23_bias.hex", hex_buf);
                26: $readmemh("image_sim_out/dpu_top_37/layer26_bias.hex", hex_buf);
                27: $readmemh("image_sim_out/dpu_top_37/layer27_bias.hex", hex_buf);
                28: $readmemh("image_sim_out/dpu_top_37/layer28_bias.hex", hex_buf);
                29: $readmemh("image_sim_out/dpu_top_37/layer29_bias.hex", hex_buf);
                31: $readmemh("image_sim_out/dpu_top_37/layer31_bias.hex", hex_buf);
                34: $readmemh("image_sim_out/dpu_top_37/layer34_bias.hex", hex_buf);
                35: $readmemh("image_sim_out/dpu_top_37/layer35_bias.hex", hex_buf);
            endcase

            // Stream biases to DPU via DMA (target=2: bias_buf)
            dma_stream_write(3'd2, 24'd0, co * 4);
        end
    endtask

    // =====================================================================
    // Find conv table index from DPU layer index
    // =====================================================================
    function integer find_conv_table_idx(input integer dpu_layer);
        integer k;
        begin
            find_conv_table_idx = -1;
            for (k = 0; k < NUM_CONV; k = k + 1)
                if (conv_idx[k] == dpu_layer)
                    find_conv_table_idx = k;
        end
    endfunction

    // =====================================================================
    // Main test sequence
    // =====================================================================
    integer i, j, pass_count, fail_count;
    integer current_layer, conv_ti;
    reg [7:0] rd_byte;
    integer all_done_flag;

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

        $display("");
        $display("=============================================================");
        $display(" DPU System Top — 36-Layer AXI System Test");
        $display(" H0=%0d  W0=%0d  NUM_LAYERS=%0d", H0, W0, NUM_LAYERS);
        $display(" ALL data through AXI interfaces (no hierarchy refs)");
        $display("=============================================================");

        // ---- Reset ----
        repeat(20) @(posedge aclk);
        aresetn = 1;
        repeat(10) @(posedge aclk);

        // ==============================================================
        // TEST 1: Read VERSION register
        // ==============================================================
        $display("\n[1] VERSION register");
        axi_read(REG_VERSION);
        if (read_result == 32'h0001_0000)
            $display("    PASS: VERSION = 0x%08h", read_result);
        else
            $display("    FAIL: VERSION = 0x%08h (expected 0x00010000)", read_result);

        // ==============================================================
        // TEST 2: Read STATUS (should be idle)
        // ==============================================================
        $display("\n[2] STATUS register");
        axi_read(REG_STATUS);
        $display("    STATUS = 0x%08h (busy=%b done=%b)", read_result,
                 read_result[0], read_result[1]);

        // ==============================================================
        // TEST 3: Enable IRQs (done + reload)
        // ==============================================================
        $display("\n[3] IRQ enable");
        axi_write(REG_IRQ_EN, 32'h3);
        axi_read(REG_IRQ_EN);
        $display("    IRQ_EN = 0x%08h", read_result);

        // ==============================================================
        // TEST 4: Load per-layer scales via PIO write_layer_desc
        // ==============================================================
        $display("\n[4] Loading per-layer scales via AXI-Lite PIO...");
        $readmemh("image_sim_out/dpu_top_37/scales.hex", scale_buf);
        for (i = 0; i < NUM_LAYERS; i = i + 1)
            pio_write_layer_scale(i[5:0], {scale_buf[i*2+1], scale_buf[i*2]});
        $display("    %0d layer scales written", NUM_LAYERS);

        // ==============================================================
        // TEST 5: Load input image via AXI-Stream DMA
        // ==============================================================
        $display("\n[5] Loading input image via DMA (%0d bytes)...", INPUT_SIZE);
        $readmemh("image_sim_out/dpu_top_37/input_image.hex", hex_buf);
        dma_stream_write(3'd1, 24'd0, INPUT_SIZE);
        axi_read(REG_DMA_STAT);
        $display("    Input image loaded via DMA (DMA_STAT=0x%08h)", read_result);

        // ==============================================================
        // TEST 6: Load layer 0 weights+biases via DMA
        // ==============================================================
        $display("\n[6] Loading L0 weights+biases via DMA...");
        load_conv_via_dma(0);
        axi_read(REG_DMA_STAT);
        $display("    L0 loaded: %0d weight bytes + %0d bias bytes (DMA_STAT=0x%08h)",
                 conv_wsize[0], conv_cout[0]*4, read_result);

        // ==============================================================
        // TEST 7: Run 36-layer inference with reload
        // ==============================================================
        axi_read(REG_STATUS);
        $display("\n[7] Starting run_all (36 layers)... STATUS=0x%08h (t=%0t)",
                 read_result, $time);
        axi_read(REG_DMA_STAT);
        $display("    DMA_STAT before run_all: 0x%08h", read_result);
        pio_run_all;
        // Wait for PIO command to propagate through PIO SM to DPU
        repeat(20) @(posedge aclk);
        axi_read(REG_STATUS);
        $display("    STATUS after run_all: 0x%08h (busy=%b layer=%0d reload=%b) (t=%0t)",
                 read_result, read_result[0], (read_result >> 8) & 6'h3F,
                 read_result[16], $time);

        // Reload loop: DPU pauses before each conv layer for weight reload
        all_done_flag = 0;
        while (!all_done_flag) begin
            wait_reload_or_done;
            if (wr_status == 0) begin
                // Done!
                all_done_flag = 1;
            end else begin
                // Reload: read STATUS to get current_layer
                axi_read(REG_STATUS);
                current_layer = (read_result >> 8) & 6'h3F;
                $display("    reload_req -> L%0d (t=%0t)", current_layer, $time);

                // Find conv table index for this layer
                conv_ti = find_conv_table_idx(current_layer);
                if (conv_ti < 0) begin
                    $display("    ERROR: L%0d not found in conv table!", current_layer);
                    $finish;
                end

                // Load weights+biases via DMA
                load_conv_via_dma(conv_ti);

                // Continue
                pio_continue;
            end
        end

        $display("    Inference DONE (t=%0t)", $time);

        // ==============================================================
        // TEST 8: Verify output (layer 35) via PIO read_byte
        // ==============================================================
        $display("\n[8] Verifying output (layer 35: %0d bytes)...", OSIZE_35);
        $readmemh("image_sim_out/dpu_top_37/layer35_expected.hex", exp_buf);

        pass_count = 0;
        fail_count = 0;
        for (i = 0; i < OSIZE_35; i = i + 1) begin
            pio_read_byte(i[23:0], rd_byte);
            if (rd_byte == exp_buf[i]) begin
                pass_count = pass_count + 1;
            end else begin
                if (fail_count < 30)
                    $display("    MISMATCH [%0d]: got=0x%02h exp=0x%02h",
                             i, rd_byte, exp_buf[i]);
                fail_count = fail_count + 1;
            end
        end

        // ==============================================================
        // TEST 9: IRQ mechanism — verify W1C clear
        // ==============================================================
        $display("\n[9] IRQ mechanism");
        axi_write(REG_IRQ_STAT, 32'h3);
        axi_read(REG_IRQ_STAT);
        if (read_result[1:0] == 2'b00)
            $display("    PASS: IRQ cleared by W1C");
        else
            $display("    FAIL: IRQ not cleared (0x%08h)", read_result);

        // ==============================================================
        // TEST 10: Performance counter
        // ==============================================================
        $display("\n[10] Performance");
        axi_read(REG_PERF);
        $display("    Total compute cycles: %0d", read_result);

        // ==============================================================
        // SUMMARY
        // ==============================================================
        $display("");
        $display("=============================================================");
        $display(" AXI SYSTEM TOP — 36-LAYER TEST SUMMARY");
        $display("=============================================================");
        $display("  Output bytes checked: %0d", pass_count + fail_count);
        $display("  PASS: %0d", pass_count);
        $display("  FAIL: %0d", fail_count);
        if (fail_count == 0)
            $display("  *** ALL TESTS PASSED — AXI SYSTEM VERIFIED ***");
        else
            $display("  *** %0d FAILURES ***", fail_count);
        $display("=============================================================");
        $display("");

        $finish;
    end

endmodule
