// DPU Top: YOLOv4-tiny sequencer with PIO interface.
// Supports up to 36 layers (configurable via NUM_LAYERS parameter).
// Uses conv_engine_array (32x32 MAC array) for parallel convolution.
// Instantiates conv_engine_array + maxpool_unit, ping-pong feature maps,
// save buffers for skip connections, and a layer descriptor ROM.
//
// Layer types: Conv3x3, Conv1x1, Conv1x1_Linear, MaxPool, Route_Split,
//              Route_Concat, Upsample_2x, Route_Save
//
// CMD interface (byte-at-a-time):
//   cmd_type 0 = write_byte (addr, data)
//   cmd_type 1 = run_layer  (runs current_layer_reg; also "continue" in run_all)
//   cmd_type 2 = read_byte  (addr -> rsp_data)
//   cmd_type 3 = set_layer  (cmd_data[5:0] -> current_layer_reg)
//   cmd_type 4 = run_all    (run layers 0..NUM_LAYERS-1; pauses before conv
//                             layers for weight reload, auto-advances others)
//   cmd_type 5 = write_scale(cmd_data -> scale_reg bytes, addr selects byte)
//   cmd_type 6 = write_layer_desc(addr[9:4]=layer, addr[3:0]=field, data)
//                field: 0=type, 1=c_in_lo, 2=c_in_hi, 3=c_out_lo, 4=c_out_hi,
//                       5=h_in_lo, 6=h_in_hi, 7=w_in_lo, 8=w_in_hi,
//                       9=h_out_lo, 10=h_out_hi, 11=w_out_lo, 12=w_out_hi,
//                       13=stride, 14=scale_lo, 15=scale_hi
//
// run_all flow: Host sends cmd_type 4. DPU runs layer 0. If next layer is
// conv, DPU asserts reload_req and accepts write_byte(0)/write_scale(5)/
// write_layer_desc(6). Host loads weights, then sends run_layer(1) to continue.
// Route/maxpool layers auto-advance without host intervention.
`default_nettype none

module dpu_top #(
    parameter int H0        = 32,
    parameter int W0        = 32,
    parameter int MAX_CH    = 512,
    parameter int MAX_FMAP  = 65536,
    parameter int MAX_WBUF  = 2400000,
    parameter int NUM_LAYERS= 18,       // how many layers run_all executes (18 or 36)
    parameter int ADDR_BITS = 24
) (
    input  logic        clk,
    input  logic        rst_n,

    input  logic        cmd_valid,
    output logic        cmd_ready,
    input  logic [2:0]  cmd_type,
    input  logic [ADDR_BITS-1:0] cmd_addr,
    input  logic [7:0]  cmd_data,

    output logic        rsp_valid,
    output logic [7:0]  rsp_data,
    output logic        busy,
    output logic        done,
    output logic [5:0]  current_layer,
    output logic        reload_req,   // asserted when waiting for weight load in run_all
    output logic [31:0] perf_total_cycles  // total compute cycles (busy high)
);

    // =========================================================================
    // Layer descriptor types
    // =========================================================================
    localparam int LT_CONV3X3      = 0;
    localparam int LT_CONV1X1      = 1;
    localparam int LT_ROUTE_SPLIT  = 2;
    localparam int LT_ROUTE_CONCAT = 3;
    localparam int LT_MAXPOOL      = 4;
    localparam int LT_CONV1X1_LIN  = 5;  // Conv1x1 LINEAR (no LeakyReLU)
    localparam int LT_UPSAMPLE     = 6;  // Nearest neighbor 2x
    localparam int LT_ROUTE_SAVE   = 7;  // Route from save buffer (copy)

    // MAX_LAYERS: fixed array size for descriptors (always 36 entries)
    localparam int MAX_LAYERS = 36;

    // =========================================================================
    // Layer descriptor ROM (18 entries)
    // =========================================================================
    localparam int H_L0  = H0/2;
    localparam int W_L0  = W0/2;
    localparam int H_L1  = H_L0/2;
    localparam int W_L1  = W_L0/2;
    localparam int H_L2  = H_L1;
    localparam int W_L2  = W_L1;
    localparam int H_L3  = H_L2;
    localparam int W_L3  = W_L2;
    localparam int H_L4  = H_L3;
    localparam int W_L4  = W_L3;
    localparam int H_L5  = H_L4;
    localparam int W_L5  = W_L4;
    localparam int H_L6  = H_L5;
    localparam int W_L6  = W_L5;
    localparam int H_L7  = H_L6;
    localparam int W_L7  = W_L6;
    localparam int H_L8  = H_L7;
    localparam int W_L8  = W_L7;
    localparam int H_L9  = H_L8/2;
    localparam int W_L9  = W_L8/2;
    localparam int H_L10 = H_L9;
    localparam int W_L10 = W_L9;
    localparam int H_L11 = H_L10;
    localparam int W_L11 = W_L10;
    localparam int H_L12 = H_L11;
    localparam int W_L12 = W_L11;
    localparam int H_L13 = H_L12;
    localparam int W_L13 = W_L12;
    localparam int H_L14 = H_L13;
    localparam int W_L14 = W_L13;
    localparam int H_L15 = H_L14;
    localparam int W_L15 = W_L14;
    localparam int H_L16 = H_L15;
    localparam int W_L16 = W_L15;
    localparam int H_L17 = H_L16/2;
    localparam int W_L17 = W_L16/2;
    // 3rd CSP block (layers 18-25)
    localparam int H_L18 = H_L17;     localparam int W_L18 = W_L17;
    localparam int H_L19 = H_L18;     localparam int W_L19 = W_L18;
    localparam int H_L20 = H_L19;     localparam int W_L20 = W_L19;
    localparam int H_L21 = H_L20;     localparam int W_L21 = W_L20;
    localparam int H_L22 = H_L21;     localparam int W_L22 = W_L21;
    localparam int H_L23 = H_L22;     localparam int W_L23 = W_L22;
    localparam int H_L24 = H_L23;     localparam int W_L24 = W_L23;
    localparam int H_L25 = (H_L24 > 1) ? H_L24/2 : 1;
    localparam int W_L25 = (W_L24 > 1) ? W_L24/2 : 1;
    // Detection head 1 (layers 26-29)
    localparam int H_L26 = H_L25;     localparam int W_L26 = W_L25;
    localparam int H_L27 = H_L26;     localparam int W_L27 = W_L26;
    localparam int H_L28 = H_L27;     localparam int W_L28 = W_L27;
    localparam int H_L29 = H_L28;     localparam int W_L29 = W_L28;
    // Bridge + Detection head 2 (layers 30-35)
    localparam int H_L30 = H_L27;     localparam int W_L30 = W_L27;  // route from L27
    localparam int H_L31 = H_L30;     localparam int W_L31 = W_L30;
    localparam int H_L32 = H_L31*2;   localparam int W_L32 = W_L31*2; // upsample 2x
    localparam int H_L33 = H_L32;     localparam int W_L33 = W_L32;  // concat
    localparam int H_L34 = H_L33;     localparam int W_L34 = W_L33;
    localparam int H_L35 = H_L34;     localparam int W_L35 = W_L34;

    // Descriptor arrays (writable via cmd_type 6)
    logic [2:0]  ld_type  [0:MAX_LAYERS-1];
    logic [10:0] ld_c_in  [0:MAX_LAYERS-1];
    logic [10:0] ld_c_out [0:MAX_LAYERS-1];
    logic [10:0] ld_h_in  [0:MAX_LAYERS-1];
    logic [10:0] ld_w_in  [0:MAX_LAYERS-1];
    logic [10:0] ld_h_out [0:MAX_LAYERS-1];
    logic [10:0] ld_w_out [0:MAX_LAYERS-1];
    logic [1:0]  ld_stride[0:MAX_LAYERS-1];
    logic [12:0] ld_macs  [0:MAX_LAYERS-1];
    logic [2:0]  ld_src_a [0:MAX_LAYERS-1];
    logic [2:0]  ld_src_b [0:MAX_LAYERS-1];
    logic [15:0] ld_scale [0:MAX_LAYERS-1];

    initial begin
        ld_type[0]=LT_CONV3X3; ld_c_in[0]=3;   ld_c_out[0]=32;  ld_h_in[0]=H0;   ld_w_in[0]=W0;   ld_h_out[0]=H_L0;  ld_w_out[0]=W_L0;  ld_stride[0]=2; ld_macs[0]=27;   ld_src_a[0]=0; ld_src_b[0]=0; ld_scale[0]=16'd655;
        ld_type[1]=LT_CONV3X3; ld_c_in[1]=32;  ld_c_out[1]=64;  ld_h_in[1]=H_L0; ld_w_in[1]=W_L0; ld_h_out[1]=H_L1;  ld_w_out[1]=W_L1;  ld_stride[1]=2; ld_macs[1]=288;  ld_src_a[1]=0; ld_src_b[1]=0; ld_scale[1]=16'd655;
        ld_type[2]=LT_CONV3X3; ld_c_in[2]=64;  ld_c_out[2]=64;  ld_h_in[2]=H_L1; ld_w_in[2]=W_L1; ld_h_out[2]=H_L2;  ld_w_out[2]=W_L2;  ld_stride[2]=1; ld_macs[2]=576;  ld_src_a[2]=0; ld_src_b[2]=0; ld_scale[2]=16'd655;
        ld_type[3]=LT_ROUTE_SPLIT; ld_c_in[3]=64; ld_c_out[3]=32; ld_h_in[3]=H_L2; ld_w_in[3]=W_L2; ld_h_out[3]=H_L3; ld_w_out[3]=W_L3; ld_stride[3]=0; ld_macs[3]=0; ld_src_a[3]=0; ld_src_b[3]=0; ld_scale[3]=16'd655;
        ld_type[4]=LT_CONV3X3; ld_c_in[4]=32;  ld_c_out[4]=32;  ld_h_in[4]=H_L3; ld_w_in[4]=W_L3; ld_h_out[4]=H_L4;  ld_w_out[4]=W_L4;  ld_stride[4]=1; ld_macs[4]=288;  ld_src_a[4]=0; ld_src_b[4]=0; ld_scale[4]=16'd655;
        ld_type[5]=LT_CONV3X3; ld_c_in[5]=32;  ld_c_out[5]=32;  ld_h_in[5]=H_L3; ld_w_in[5]=W_L3; ld_h_out[5]=H_L5;  ld_w_out[5]=W_L5;  ld_stride[5]=1; ld_macs[5]=288;  ld_src_a[5]=0; ld_src_b[5]=0; ld_scale[5]=16'd655;
        ld_type[6]=LT_ROUTE_CONCAT; ld_c_in[6]=32; ld_c_out[6]=64; ld_h_in[6]=H_L5; ld_w_in[6]=W_L5; ld_h_out[6]=H_L6; ld_w_out[6]=W_L6; ld_stride[6]=0; ld_macs[6]=0; ld_src_a[6]=0; ld_src_b[6]=2; ld_scale[6]=16'd655;
        ld_type[7]=LT_CONV1X1; ld_c_in[7]=64;  ld_c_out[7]=64;  ld_h_in[7]=H_L6; ld_w_in[7]=W_L6; ld_h_out[7]=H_L7;  ld_w_out[7]=W_L7;  ld_stride[7]=1; ld_macs[7]=64;   ld_src_a[7]=0; ld_src_b[7]=0; ld_scale[7]=16'd655;
        ld_type[8]=LT_ROUTE_CONCAT; ld_c_in[8]=64; ld_c_out[8]=128; ld_h_in[8]=H_L7; ld_w_in[8]=W_L7; ld_h_out[8]=H_L8; ld_w_out[8]=W_L8; ld_stride[8]=0; ld_macs[8]=0; ld_src_a[8]=1; ld_src_b[8]=0; ld_scale[8]=16'd655;
        ld_type[9]=LT_MAXPOOL; ld_c_in[9]=128; ld_c_out[9]=128; ld_h_in[9]=H_L8; ld_w_in[9]=W_L8; ld_h_out[9]=H_L9; ld_w_out[9]=W_L9; ld_stride[9]=2; ld_macs[9]=0; ld_src_a[9]=0; ld_src_b[9]=0; ld_scale[9]=16'd655;
        ld_type[10]=LT_CONV3X3; ld_c_in[10]=128; ld_c_out[10]=128; ld_h_in[10]=H_L9; ld_w_in[10]=W_L9; ld_h_out[10]=H_L10; ld_w_out[10]=W_L10; ld_stride[10]=1; ld_macs[10]=1152; ld_src_a[10]=0; ld_src_b[10]=0; ld_scale[10]=16'd655;
        ld_type[11]=LT_ROUTE_SPLIT; ld_c_in[11]=128; ld_c_out[11]=64; ld_h_in[11]=H_L10; ld_w_in[11]=W_L10; ld_h_out[11]=H_L11; ld_w_out[11]=W_L11; ld_stride[11]=0; ld_macs[11]=0; ld_src_a[11]=0; ld_src_b[11]=0; ld_scale[11]=16'd655;
        ld_type[12]=LT_CONV3X3; ld_c_in[12]=64; ld_c_out[12]=64; ld_h_in[12]=H_L11; ld_w_in[12]=W_L11; ld_h_out[12]=H_L12; ld_w_out[12]=W_L12; ld_stride[12]=1; ld_macs[12]=576; ld_src_a[12]=0; ld_src_b[12]=0; ld_scale[12]=16'd655;
        ld_type[13]=LT_CONV3X3; ld_c_in[13]=64; ld_c_out[13]=64; ld_h_in[13]=H_L11; ld_w_in[13]=W_L11; ld_h_out[13]=H_L13; ld_w_out[13]=W_L13; ld_stride[13]=1; ld_macs[13]=576; ld_src_a[13]=0; ld_src_b[13]=0; ld_scale[13]=16'd655;
        ld_type[14]=LT_ROUTE_CONCAT; ld_c_in[14]=64; ld_c_out[14]=128; ld_h_in[14]=H_L13; ld_w_in[14]=W_L13; ld_h_out[14]=H_L14; ld_w_out[14]=W_L14; ld_stride[14]=0; ld_macs[14]=0; ld_src_a[14]=0; ld_src_b[14]=4; ld_scale[14]=16'd655;
        ld_type[15]=LT_CONV1X1; ld_c_in[15]=128; ld_c_out[15]=128; ld_h_in[15]=H_L14; ld_w_in[15]=W_L14; ld_h_out[15]=H_L15; ld_w_out[15]=W_L15; ld_stride[15]=1; ld_macs[15]=128; ld_src_a[15]=0; ld_src_b[15]=0; ld_scale[15]=16'd655;
        ld_type[16]=LT_ROUTE_CONCAT; ld_c_in[16]=128; ld_c_out[16]=256; ld_h_in[16]=H_L15; ld_w_in[16]=W_L15; ld_h_out[16]=H_L16; ld_w_out[16]=W_L16; ld_stride[16]=0; ld_macs[16]=0; ld_src_a[16]=3; ld_src_b[16]=0; ld_scale[16]=16'd655;
        ld_type[17]=LT_MAXPOOL; ld_c_in[17]=256; ld_c_out[17]=256; ld_h_in[17]=H_L16; ld_w_in[17]=W_L16; ld_h_out[17]=H_L17; ld_w_out[17]=W_L17; ld_stride[17]=2; ld_macs[17]=0; ld_src_a[17]=0; ld_src_b[17]=0; ld_scale[17]=16'd655;
        // 3rd CSP block (layers 18-25)
        ld_type[18]=LT_CONV3X3; ld_c_in[18]=256; ld_c_out[18]=256; ld_h_in[18]=H_L17; ld_w_in[18]=W_L17; ld_h_out[18]=H_L18; ld_w_out[18]=W_L18; ld_stride[18]=1; ld_macs[18]=2304; ld_src_a[18]=0; ld_src_b[18]=0; ld_scale[18]=16'd655;
        ld_type[19]=LT_ROUTE_SPLIT; ld_c_in[19]=256; ld_c_out[19]=128; ld_h_in[19]=H_L18; ld_w_in[19]=W_L18; ld_h_out[19]=H_L19; ld_w_out[19]=W_L19; ld_stride[19]=0; ld_macs[19]=0; ld_src_a[19]=0; ld_src_b[19]=0; ld_scale[19]=16'd655;
        ld_type[20]=LT_CONV3X3; ld_c_in[20]=128; ld_c_out[20]=128; ld_h_in[20]=H_L19; ld_w_in[20]=W_L19; ld_h_out[20]=H_L20; ld_w_out[20]=W_L20; ld_stride[20]=1; ld_macs[20]=1152; ld_src_a[20]=0; ld_src_b[20]=0; ld_scale[20]=16'd655;
        ld_type[21]=LT_CONV3X3; ld_c_in[21]=128; ld_c_out[21]=128; ld_h_in[21]=H_L19; ld_w_in[21]=W_L19; ld_h_out[21]=H_L21; ld_w_out[21]=W_L21; ld_stride[21]=1; ld_macs[21]=1152; ld_src_a[21]=0; ld_src_b[21]=0; ld_scale[21]=16'd655;
        ld_type[22]=LT_ROUTE_CONCAT; ld_c_in[22]=128; ld_c_out[22]=256; ld_h_in[22]=H_L21; ld_w_in[22]=W_L21; ld_h_out[22]=H_L22; ld_w_out[22]=W_L22; ld_stride[22]=0; ld_macs[22]=0; ld_src_a[22]=0; ld_src_b[22]=5; ld_scale[22]=16'd655;
        ld_type[23]=LT_CONV1X1; ld_c_in[23]=256; ld_c_out[23]=256; ld_h_in[23]=H_L22; ld_w_in[23]=W_L22; ld_h_out[23]=H_L23; ld_w_out[23]=W_L23; ld_stride[23]=1; ld_macs[23]=256; ld_src_a[23]=0; ld_src_b[23]=0; ld_scale[23]=16'd655;
        ld_type[24]=LT_ROUTE_CONCAT; ld_c_in[24]=256; ld_c_out[24]=512; ld_h_in[24]=H_L23; ld_w_in[24]=W_L23; ld_h_out[24]=H_L24; ld_w_out[24]=W_L24; ld_stride[24]=0; ld_macs[24]=0; ld_src_a[24]=5; ld_src_b[24]=0; ld_scale[24]=16'd655;
        ld_type[25]=LT_MAXPOOL; ld_c_in[25]=512; ld_c_out[25]=512; ld_h_in[25]=H_L24; ld_w_in[25]=W_L24; ld_h_out[25]=H_L25; ld_w_out[25]=W_L25; ld_stride[25]=2; ld_macs[25]=0; ld_src_a[25]=0; ld_src_b[25]=0; ld_scale[25]=16'd655;
        // Detection head 1 (layers 26-29)
        ld_type[26]=LT_CONV3X3; ld_c_in[26]=512; ld_c_out[26]=512; ld_h_in[26]=H_L25; ld_w_in[26]=W_L25; ld_h_out[26]=H_L26; ld_w_out[26]=W_L26; ld_stride[26]=1; ld_macs[26]=4608; ld_src_a[26]=0; ld_src_b[26]=0; ld_scale[26]=16'd655;
        ld_type[27]=LT_CONV1X1; ld_c_in[27]=512; ld_c_out[27]=256; ld_h_in[27]=H_L26; ld_w_in[27]=W_L26; ld_h_out[27]=H_L27; ld_w_out[27]=W_L27; ld_stride[27]=1; ld_macs[27]=512; ld_src_a[27]=0; ld_src_b[27]=0; ld_scale[27]=16'd655;
        ld_type[28]=LT_CONV3X3; ld_c_in[28]=256; ld_c_out[28]=512; ld_h_in[28]=H_L27; ld_w_in[28]=W_L27; ld_h_out[28]=H_L28; ld_w_out[28]=W_L28; ld_stride[28]=1; ld_macs[28]=2304; ld_src_a[28]=0; ld_src_b[28]=0; ld_scale[28]=16'd655;
        ld_type[29]=LT_CONV1X1_LIN; ld_c_in[29]=512; ld_c_out[29]=255; ld_h_in[29]=H_L28; ld_w_in[29]=W_L28; ld_h_out[29]=H_L29; ld_w_out[29]=W_L29; ld_stride[29]=1; ld_macs[29]=512; ld_src_a[29]=0; ld_src_b[29]=0; ld_scale[29]=16'd655;
        // Bridge + Detection head 2 (layers 30-35)
        ld_type[30]=LT_ROUTE_SAVE; ld_c_in[30]=256; ld_c_out[30]=256; ld_h_in[30]=H_L27; ld_w_in[30]=W_L27; ld_h_out[30]=H_L30; ld_w_out[30]=W_L30; ld_stride[30]=0; ld_macs[30]=0; ld_src_a[30]=7; ld_src_b[30]=0; ld_scale[30]=16'd655;
        ld_type[31]=LT_CONV1X1; ld_c_in[31]=256; ld_c_out[31]=128; ld_h_in[31]=H_L30; ld_w_in[31]=W_L30; ld_h_out[31]=H_L31; ld_w_out[31]=W_L31; ld_stride[31]=1; ld_macs[31]=256; ld_src_a[31]=0; ld_src_b[31]=0; ld_scale[31]=16'd655;
        ld_type[32]=LT_UPSAMPLE; ld_c_in[32]=128; ld_c_out[32]=128; ld_h_in[32]=H_L31; ld_w_in[32]=W_L31; ld_h_out[32]=H_L32; ld_w_out[32]=W_L32; ld_stride[32]=0; ld_macs[32]=0; ld_src_a[32]=0; ld_src_b[32]=0; ld_scale[32]=16'd655;
        ld_type[33]=LT_ROUTE_CONCAT; ld_c_in[33]=128; ld_c_out[33]=384; ld_h_in[33]=H_L32; ld_w_in[33]=W_L32; ld_h_out[33]=H_L33; ld_w_out[33]=W_L33; ld_stride[33]=0; ld_macs[33]=0; ld_src_a[33]=0; ld_src_b[33]=6; ld_scale[33]=16'd655;
        ld_type[34]=LT_CONV3X3; ld_c_in[34]=384; ld_c_out[34]=256; ld_h_in[34]=H_L33; ld_w_in[34]=W_L33; ld_h_out[34]=H_L34; ld_w_out[34]=W_L34; ld_stride[34]=1; ld_macs[34]=3456; ld_src_a[34]=0; ld_src_b[34]=0; ld_scale[34]=16'd655;
        ld_type[35]=LT_CONV1X1_LIN; ld_c_in[35]=256; ld_c_out[35]=255; ld_h_in[35]=H_L34; ld_w_in[35]=W_L34; ld_h_out[35]=H_L35; ld_w_out[35]=W_L35; ld_stride[35]=1; ld_macs[35]=256; ld_src_a[35]=0; ld_src_b[35]=0; ld_scale[35]=16'd655;
    end

    // =========================================================================
    // Memories
    // =========================================================================
    reg signed [7:0] fmap_a [0:MAX_FMAP-1];
    reg signed [7:0] fmap_b [0:MAX_FMAP-1];

    localparam int SAVE_L2_SIZE  = 64 * H_L2 * W_L2;
    localparam int SAVE_L4_SIZE  = 32 * H_L4 * W_L4;
    localparam int SAVE_L10_SIZE = 128 * H_L10 * W_L10;
    localparam int SAVE_L12_SIZE = 64 * H_L12 * W_L12;
    // Guard with max(1,...) so H0=16 doesn't produce 0-size arrays
    localparam int SAVE_L18_SIZE = (256 * H_L18 * W_L18 > 0) ? 256 * H_L18 * W_L18 : 1;
    localparam int SAVE_L20_SIZE = (128 * H_L20 * W_L20 > 0) ? 128 * H_L20 * W_L20 : 1;
    localparam int SAVE_L23_SIZE = (256 * H_L23 * W_L23 > 0) ? 256 * H_L23 * W_L23 : 1;
    localparam int SAVE_L27_SIZE = (256 * H_L27 * W_L27 > 0) ? 256 * H_L27 * W_L27 : 1;

    reg signed [7:0] fmap_save_l2  [0:SAVE_L2_SIZE-1];
    reg signed [7:0] fmap_save_l4  [0:SAVE_L4_SIZE-1];
    reg signed [7:0] fmap_save_l10 [0:SAVE_L10_SIZE-1];
    reg signed [7:0] fmap_save_l12 [0:SAVE_L12_SIZE-1];
    reg signed [7:0] fmap_save_l18 [0:SAVE_L18_SIZE-1];
    reg signed [7:0] fmap_save_l20 [0:SAVE_L20_SIZE-1];
    reg signed [7:0] fmap_save_l23 [0:SAVE_L23_SIZE-1];
    reg signed [7:0] fmap_save_l27 [0:SAVE_L27_SIZE-1];

    reg signed [7:0]  weight_buf [0:MAX_WBUF-1];
    reg signed [31:0] bias_buf   [0:MAX_CH-1];
    logic [15:0]      scale_reg;

    reg signed [7:0] patch_buf [0:4607];  // max c_in*K*K = 512*9 = 4608

    // Performance counters
    reg [31:0] cycle_counter;
    reg [31:0] layer_start_cycle;
    reg [31:0] layer_cycles [0:MAX_LAYERS-1];
    assign perf_total_cycles = cycle_counter;

    integer init_i;
    initial begin
        for (init_i = 0; init_i < MAX_CH; init_i = init_i + 1)
            bias_buf[init_i] = 32'sd0;
        scale_reg = 16'd655;
        cycle_counter = 32'd0;
        layer_start_cycle = 32'd0;
        for (init_i = 0; init_i < MAX_LAYERS; init_i = init_i + 1)
            layer_cycles[init_i] = 32'd0;
    end

    // =========================================================================
    // Ping-pong control
    // =========================================================================
    logic ping_pong;

    // =========================================================================
    // Conv engine array instance
    // =========================================================================
    logic        eng_start, eng_done;
    logic        eng_out_valid;
    logic [8:0]  eng_out_ch_base;
    logic [5:0]  eng_out_count;
    wire [255:0] eng_out_data_flat;

    // Unpack engine output from flat vector into local reg array
    reg signed [7:0] eng_out_data [0:31];
    integer unpack_i;
    always @(eng_out_data_flat) begin
        for (unpack_i = 0; unpack_i < 32; unpack_i = unpack_i + 1)
            eng_out_data[unpack_i] = $signed(eng_out_data_flat[unpack_i*8 +: 8]);
    end

    // Latched pulse signals (1-cycle pulses that may arrive while busy)
    logic        eng_done_latched;
    logic [12:0] eng_patch_rd_addr;
    wire [255:0] eng_patch_rd_data_wide;
    logic [21:0] eng_weight_rd_addr;
    wire [255:0] eng_weight_rd_data_wide;
    logic [8:0]  eng_bias_rd_ch;
    logic signed [31:0] eng_bias_rd_data;

    // skip_relu: 1 for LINEAR layers (LT_CONV1X1_LIN)
    logic eng_skip_relu;

    conv_engine_array #(.SCALE_Q(16)) u_engine (
        .clk(clk), .rst_n(rst_n), .start(eng_start),
        .c_in(cur_c_in), .c_out(cur_c_out),
        .kernel_size(cur_kernel_size),
        .patch_rd_addr(eng_patch_rd_addr),
        .patch_rd_data_wide(eng_patch_rd_data_wide),
        .weight_rd_addr(eng_weight_rd_addr),
        .weight_rd_data_wide(eng_weight_rd_data_wide),
        .bias_rd_ch(eng_bias_rd_ch),
        .bias_rd_data(eng_bias_rd_data),
        .scale(scale_reg),
        .skip_relu(eng_skip_relu),
        .out_valid(eng_out_valid),
        .out_ch_base(eng_out_ch_base),
        .out_count(eng_out_count),
        .out_data_flat(eng_out_data_flat),
        .done(eng_done)
    );

    // Wide (32-byte) read ports: provide 32 consecutive bytes from base address
    genvar gi;
    generate
        for (gi = 0; gi < 32; gi = gi + 1) begin : gen_wide_rd
            assign eng_weight_rd_data_wide[gi*8 +: 8] = weight_buf[eng_weight_rd_addr + gi];
            assign eng_patch_rd_data_wide[gi*8 +: 8]  = patch_buf[eng_patch_rd_addr + gi];
        end
    endgenerate
    assign eng_bias_rd_data = bias_buf[eng_bias_rd_ch];

    // =========================================================================
    // MaxPool unit instance
    // =========================================================================
    logic        pool_valid, pool_done;
    logic signed [7:0] pool_a, pool_b, pool_c, pool_d, pool_max;

    maxpool_unit u_pool (
        .clk(clk), .rst_n(rst_n), .valid(pool_valid),
        .a(pool_a), .b(pool_b), .c(pool_c), .d(pool_d),
        .max_out(pool_max), .done(pool_done)
    );

    // =========================================================================
    // FSM
    // =========================================================================
    typedef enum logic [4:0] {
        S_IDLE,
        S_WRITE_ACK,
        S_READ_ACK,
        S_LAYER_INIT,
        S_CONV_LOAD_PATCH,
        S_CONV_START_ENG,
        S_CONV_ENGINE,
        S_CONV_WRITE_BATCH,
        S_CONV_NEXT_PIXEL,
        S_POOL_LOAD,
        S_POOL_COMPUTE,
        S_POOL_WAIT,
        S_POOL_WRITE,
        S_POOL_NEXT,
        S_ROUTE_COPY_A,
        S_ROUTE_COPY_B,
        S_ROUTE_DONE,
        S_UPSAMPLE_COPY,
        S_ROUTE_SAVE_COPY,
        S_SAVE_FMAP,
        S_LAYER_DONE,
        S_LAYER_RELOAD,
        S_ALL_DONE
    } state_t;
    state_t state;

    logic [5:0] current_layer_reg;
    assign current_layer = current_layer_reg;

    logic run_all_mode;

    // Current layer descriptor (latched)
    logic [2:0]  cur_type;
    logic [10:0] cur_c_in, cur_c_out, cur_h_in, cur_w_in, cur_h_out, cur_w_out;
    logic [1:0]  cur_stride;
    logic [12:0] cur_macs;
    logic [2:0]  cur_src_a, cur_src_b;
    logic [3:0]  cur_kernel_size;

    // Loop counters
    integer oh, ow;
    integer load_c, load_ky, load_kx, load_idx;
    integer copy_idx, copy_total;
    integer pool_ch, pool_oh, pool_ow;
    integer save_idx;

    // Batch write counter
    integer batch_idx;
    integer batch_ch_base;
    integer batch_count;

    // Temporaries
    integer tmp_ih, tmp_iw, tmp_src_addr, tmp_dst_addr;
    integer tmp_total_a, tmp_total_b, tmp_ch_half;
    integer tmp_boff, tmp_bch, tmp_bsel, tmp_foff;
    integer tmp_base_h, tmp_base_w;
    integer tmp_addr00, tmp_addr01, tmp_addr10, tmp_addr11;
    integer out_addr;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state           <= S_IDLE;
            cmd_ready       <= 1'b0;
            rsp_valid       <= 1'b0;
            rsp_data        <= 8'd0;
            busy            <= 1'b0;
            done            <= 1'b0;
            eng_start       <= 1'b0;
            pool_valid      <= 1'b0;
            current_layer_reg <= 6'd0;
            ping_pong       <= 1'b0;
            run_all_mode    <= 1'b0;
            reload_req      <= 1'b0;
            cycle_counter   <= 32'd0;
            layer_start_cycle <= 32'd0;
            oh <= 0; ow <= 0;
            load_c <= 0; load_ky <= 0; load_kx <= 0; load_idx <= 0;
            copy_idx <= 0; save_idx <= 0;
            copy_total <= 0;
            pool_ch <= 0; pool_oh <= 0; pool_ow <= 0;
            batch_idx <= 0; batch_ch_base <= 0; batch_count <= 0;
            cur_kernel_size <= 4'd3;
            eng_done_latched <= 1'b0;
            eng_skip_relu <= 1'b0;
        end else begin
            cmd_ready  <= 1'b0;
            rsp_valid  <= 1'b0;
            done       <= 1'b0;
            eng_start  <= 1'b0;
            pool_valid <= 1'b0;
            reload_req <= 1'b0;

            // Performance counter: increment when busy
            if (busy)
                cycle_counter <= cycle_counter + 32'd1;

            // Latch engine done pulse (cleared when consumed)
            if (eng_done)
                eng_done_latched <= 1'b1;

            case (state)
                // =============================================================
                S_IDLE: begin
                    cmd_ready <= 1'b1;
                    if (cmd_valid) begin
                        case (cmd_type)
                            3'd0: begin // write_byte
                                if (cmd_addr < MAX_WBUF) begin
                                    weight_buf[cmd_addr] <= $signed(cmd_data);
                                end else if (cmd_addr < MAX_WBUF + MAX_CH*4) begin
                                    tmp_boff = cmd_addr - MAX_WBUF;
                                    tmp_bch  = tmp_boff >> 2;
                                    tmp_bsel = tmp_boff & 3;
                                    case (tmp_bsel)
                                        0: bias_buf[tmp_bch][7:0]   <= cmd_data;
                                        1: bias_buf[tmp_bch][15:8]  <= cmd_data;
                                        2: bias_buf[tmp_bch][23:16] <= cmd_data;
                                        3: bias_buf[tmp_bch][31:24] <= cmd_data;
                                    endcase
                                end else begin
                                    tmp_foff = cmd_addr - MAX_WBUF - MAX_CH*4;
                                    fmap_a[tmp_foff] <= $signed(cmd_data);
                                end
                                state <= S_WRITE_ACK;
                            end
                            3'd1: begin // run_layer
                                busy <= 1'b1;
                                run_all_mode <= 1'b0;
                                state <= S_LAYER_INIT;
                            end
                            3'd2: begin // read_byte
                                if (ping_pong == 1'b0)
                                    rsp_data <= fmap_a[cmd_addr];
                                else
                                    rsp_data <= fmap_b[cmd_addr];
                                rsp_valid <= 1'b1;
                                state <= S_READ_ACK;
                            end
                            3'd3: begin // set_layer
                                current_layer_reg <= cmd_data[5:0];
                                state <= S_WRITE_ACK;
                            end
                            3'd4: begin // run_all
                                busy <= 1'b1;
                                run_all_mode <= 1'b1;
                                current_layer_reg <= 6'd0;
                                ping_pong <= 1'b0;
                                cycle_counter <= 32'd0;
                                state <= S_LAYER_INIT;
                            end
                            3'd5: begin // write_scale (also updates current layer's ld_scale)
                                if (cmd_addr[0] == 1'b0) begin
                                    scale_reg[7:0]  <= cmd_data;
                                    ld_scale[current_layer_reg][7:0] <= cmd_data;
                                end else begin
                                    scale_reg[15:8] <= cmd_data;
                                    ld_scale[current_layer_reg][15:8] <= cmd_data;
                                end
                                state <= S_WRITE_ACK;
                            end
                            3'd6: begin // write_layer_desc
                                // addr[8:4] = layer index, addr[3:0] = field
                                case (cmd_addr[3:0])
                                    4'd0:  ld_type  [cmd_addr[9:4]] <= cmd_data[2:0];
                                    4'd1:  ld_c_in  [cmd_addr[9:4]][7:0]  <= cmd_data;
                                    4'd2:  ld_c_in  [cmd_addr[9:4]][10:8] <= cmd_data[2:0];
                                    4'd3:  ld_c_out [cmd_addr[9:4]][7:0]  <= cmd_data;
                                    4'd4:  ld_c_out [cmd_addr[9:4]][10:8] <= cmd_data[2:0];
                                    4'd5:  ld_h_in  [cmd_addr[9:4]][7:0]  <= cmd_data;
                                    4'd6:  ld_h_in  [cmd_addr[9:4]][10:8] <= cmd_data[2:0];
                                    4'd7:  ld_w_in  [cmd_addr[9:4]][7:0]  <= cmd_data;
                                    4'd8:  ld_w_in  [cmd_addr[9:4]][10:8] <= cmd_data[2:0];
                                    4'd9:  ld_h_out [cmd_addr[9:4]][7:0]  <= cmd_data;
                                    4'd10: ld_h_out [cmd_addr[9:4]][10:8] <= cmd_data[2:0];
                                    4'd11: ld_w_out [cmd_addr[9:4]][7:0]  <= cmd_data;
                                    4'd12: ld_w_out [cmd_addr[9:4]][10:8] <= cmd_data[2:0];
                                    4'd13: ld_stride[cmd_addr[9:4]] <= cmd_data[1:0];
                                    4'd14: ld_scale [cmd_addr[9:4]][7:0]  <= cmd_data;
                                    4'd15: ld_scale [cmd_addr[9:4]][15:8] <= cmd_data;
                                endcase
                                state <= S_WRITE_ACK;
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

                // =============================================================
                S_LAYER_INIT: begin
                    layer_start_cycle <= cycle_counter;
                    cur_type   <= ld_type  [current_layer_reg];
                    cur_c_in   <= ld_c_in  [current_layer_reg];
                    cur_c_out  <= ld_c_out [current_layer_reg];
                    cur_h_in   <= ld_h_in  [current_layer_reg];
                    cur_w_in   <= ld_w_in  [current_layer_reg];
                    cur_h_out  <= ld_h_out [current_layer_reg];
                    cur_w_out  <= ld_w_out [current_layer_reg];
                    cur_stride <= ld_stride[current_layer_reg];
                    cur_macs   <= ld_macs  [current_layer_reg];
                    cur_src_a  <= ld_src_a [current_layer_reg];
                    cur_src_b  <= ld_src_b [current_layer_reg];
                    scale_reg  <= ld_scale [current_layer_reg];

                    // Determine kernel size and activation mode
                    if (ld_type[current_layer_reg] == LT_CONV3X3)
                        cur_kernel_size <= 4'd3;
                    else
                        cur_kernel_size <= 4'd1;
                    eng_skip_relu <= (ld_type[current_layer_reg] == LT_CONV1X1_LIN);

                    oh <= 0; ow <= 0;
                    load_c <= 0; load_ky <= 0; load_kx <= 0; load_idx <= 0;
                    copy_idx <= 0;
                    pool_ch <= 0; pool_oh <= 0; pool_ow <= 0;
                    save_idx <= 0;

                    case (ld_type[current_layer_reg])
                        LT_CONV3X3, LT_CONV1X1,
                        LT_CONV1X1_LIN:         state <= S_CONV_LOAD_PATCH;
                        LT_MAXPOOL:              state <= S_POOL_LOAD;
                        LT_ROUTE_SPLIT:          state <= S_ROUTE_COPY_A;
                        LT_ROUTE_CONCAT:         state <= S_ROUTE_COPY_A;
                        LT_UPSAMPLE:             state <= S_UPSAMPLE_COPY;
                        LT_ROUTE_SAVE:           state <= S_ROUTE_SAVE_COPY;
                        default:                 state <= S_LAYER_DONE;
                    endcase
                end

                // =============================================================
                // CONV: Load patch buffer with activations for pixel (oh, ow)
                // Layout: patch_buf[kpos * c_in + c] (cin-contiguous per kpos)
                // Iterates: for each kpos (0..K*K-1), for each c (0..c_in-1)
                // load_idx goes 0..macs-1 where macs = c_in * K * K
                // =============================================================
                S_CONV_LOAD_PATCH: begin
                    if (cur_type == LT_CONV3X3) begin  // 3x3 conv
                        tmp_ih = oh * cur_stride + load_ky - 1;  // pad=1
                        tmp_iw = ow * cur_stride + load_kx - 1;
                        if (tmp_ih < 0 || tmp_ih >= cur_h_in || tmp_iw < 0 || tmp_iw >= cur_w_in) begin
                            patch_buf[load_idx] <= 8'sd0;
                        end else begin
                            tmp_src_addr = load_c * cur_h_in * cur_w_in + tmp_ih * cur_w_in + tmp_iw;
                            if (ping_pong == 1'b0)
                                patch_buf[load_idx] <= fmap_a[tmp_src_addr];
                            else
                                patch_buf[load_idx] <= fmap_b[tmp_src_addr];
                        end

                        // Iterate: inner loop = c (0..c_in-1), outer loop = kpos
                        if (load_c + 1 < cur_c_in) begin
                            load_c <= load_c + 1;
                        end else begin
                            load_c <= 0;
                            if (load_kx == 2) begin
                                load_kx <= 0;
                                if (load_ky == 2) begin
                                    load_ky <= 0;
                                end else begin
                                    load_ky <= load_ky + 1;
                                end
                            end else begin
                                load_kx <= load_kx + 1;
                            end
                        end
                    end else begin
                        // 1x1 conv: kpos=0 only, just iterate c
                        tmp_src_addr = load_c * cur_h_in * cur_w_in + oh * cur_w_in + ow;
                        if (ping_pong == 1'b0)
                            patch_buf[load_idx] <= fmap_a[tmp_src_addr];
                        else
                            patch_buf[load_idx] <= fmap_b[tmp_src_addr];
                        load_c <= load_c + 1;
                    end

                    if (load_idx == cur_macs - 1) begin
                        state <= S_CONV_START_ENG;
                    end else begin
                        load_idx <= load_idx + 1;
                    end
                end

                // =============================================================
                // CONV: Start engine array (processes ALL output channels)
                // =============================================================
                S_CONV_START_ENG: begin
                    eng_start <= 1'b1;
                    eng_done_latched <= 1'b0;
                    state     <= S_CONV_ENGINE;
                end

                // =============================================================
                // CONV: Wait for engine output batch
                // eng_out_valid is a 1-cycle pulse; eng_out_data/ch_base/count
                // hold their values until the NEXT out_valid, so we can read
                // them over multiple cycles in S_CONV_WRITE_BATCH.
                // =============================================================
                S_CONV_ENGINE: begin
                    if (eng_out_valid) begin
                        batch_ch_base <= eng_out_ch_base;
                        batch_count   <= eng_out_count;
                        batch_idx     <= 0;
                        state         <= S_CONV_WRITE_BATCH;
                    end
                end

                // =============================================================
                // CONV: Write batch of results to output fmap
                // eng_out_data[] holds values until next S_OUTPUT in engine
                // =============================================================
                S_CONV_WRITE_BATCH: begin
                    out_addr = (batch_ch_base + batch_idx) * cur_h_out * cur_w_out
                             + oh * cur_w_out + ow;
                    if (ping_pong == 1'b0)
                        fmap_b[out_addr] <= eng_out_data[batch_idx];
                    else
                        fmap_a[out_addr] <= eng_out_data[batch_idx];

                    if (batch_idx + 1 >= batch_count) begin
                        state <= S_CONV_NEXT_PIXEL;
                    end else begin
                        batch_idx <= batch_idx + 1;
                    end
                end

                // =============================================================
                // CONV: After batch write, wait for more batches or done
                // =============================================================
                S_CONV_NEXT_PIXEL: begin
                    if (eng_done || eng_done_latched) begin
                        // All Cout channels done for this pixel. Next pixel.
                        eng_done_latched <= 1'b0;
                        if (ow + 1 < cur_w_out) begin
                            ow <= ow + 1;
                            load_c <= 0; load_ky <= 0; load_kx <= 0; load_idx <= 0;
                            state <= S_CONV_LOAD_PATCH;
                        end else begin
                            ow <= 0;
                            if (oh + 1 < cur_h_out) begin
                                oh <= oh + 1;
                                load_c <= 0; load_ky <= 0; load_kx <= 0; load_idx <= 0;
                                state <= S_CONV_LOAD_PATCH;
                            end else begin
                                state <= S_SAVE_FMAP;
                            end
                        end
                    end else if (eng_out_valid) begin
                        // Another batch of results
                        batch_ch_base <= eng_out_ch_base;
                        batch_count   <= eng_out_count;
                        batch_idx     <= 0;
                        state         <= S_CONV_WRITE_BATCH;
                    end
                end

                // =============================================================
                // MAXPOOL
                // =============================================================
                S_POOL_LOAD: begin
                    tmp_base_h = pool_oh * 2;
                    tmp_base_w = pool_ow * 2;
                    tmp_addr00 = pool_ch * cur_h_in * cur_w_in + tmp_base_h * cur_w_in + tmp_base_w;
                    tmp_addr01 = tmp_addr00 + 1;
                    tmp_addr10 = tmp_addr00 + cur_w_in;
                    tmp_addr11 = tmp_addr10 + 1;

                    if (ping_pong == 1'b0) begin
                        pool_a <= fmap_a[tmp_addr00];
                        pool_b <= fmap_a[tmp_addr01];
                        pool_c <= fmap_a[tmp_addr10];
                        pool_d <= fmap_a[tmp_addr11];
                    end else begin
                        pool_a <= fmap_b[tmp_addr00];
                        pool_b <= fmap_b[tmp_addr01];
                        pool_c <= fmap_b[tmp_addr10];
                        pool_d <= fmap_b[tmp_addr11];
                    end
                    state <= S_POOL_COMPUTE;
                end

                S_POOL_COMPUTE: begin
                    pool_valid <= 1'b1;
                    state <= S_POOL_WAIT;
                end

                S_POOL_WAIT: begin
                    if (pool_done) begin
                        out_addr = pool_ch * cur_h_out * cur_w_out + pool_oh * cur_w_out + pool_ow;
                        state <= S_POOL_WRITE;
                    end
                end

                S_POOL_WRITE: begin
                    if (ping_pong == 1'b0)
                        fmap_b[out_addr] <= pool_max;
                    else
                        fmap_a[out_addr] <= pool_max;
                    state <= S_POOL_NEXT;
                end

                S_POOL_NEXT: begin
                    if (pool_ow + 1 < cur_w_out) begin
                        pool_ow <= pool_ow + 1;
                        state <= S_POOL_LOAD;
                    end else begin
                        pool_ow <= 0;
                        if (pool_oh + 1 < cur_h_out) begin
                            pool_oh <= pool_oh + 1;
                            state <= S_POOL_LOAD;
                        end else begin
                            pool_oh <= 0;
                            if (pool_ch + 1 < cur_c_out) begin
                                pool_ch <= pool_ch + 1;
                                state <= S_POOL_LOAD;
                            end else begin
                                state <= S_SAVE_FMAP;
                            end
                        end
                    end
                end

                // =============================================================
                // ROUTE
                // =============================================================
                S_ROUTE_COPY_A: begin
                    if (cur_type == LT_ROUTE_SPLIT) begin
                        tmp_total_a = cur_c_out * cur_h_out * cur_w_out;
                        tmp_src_addr = cur_c_out * cur_h_in * cur_w_in + copy_idx;

                        if (ping_pong == 1'b0)
                            fmap_b[copy_idx] <= fmap_a[tmp_src_addr];
                        else
                            fmap_a[copy_idx] <= fmap_b[tmp_src_addr];

                        if (copy_idx + 1 >= tmp_total_a) begin
                            state <= S_SAVE_FMAP;
                        end else begin
                            copy_idx <= copy_idx + 1;
                        end
                    end else begin
                        // src_a channels = cur_c_in (supports unequal concat like L33: 128+256=384)
                        tmp_total_a = cur_c_in * cur_h_out * cur_w_out;

                        case (cur_src_a)
                            3'd0: begin  // current fmap (previous layer output)
                                if (ping_pong == 1'b0)
                                    fmap_b[copy_idx] <= fmap_a[copy_idx];
                                else
                                    fmap_a[copy_idx] <= fmap_b[copy_idx];
                            end
                            3'd1: begin  // save_l2
                                if (ping_pong == 1'b0)
                                    fmap_b[copy_idx] <= fmap_save_l2[copy_idx];
                                else
                                    fmap_a[copy_idx] <= fmap_save_l2[copy_idx];
                            end
                            3'd3: begin  // save_l10
                                if (ping_pong == 1'b0)
                                    fmap_b[copy_idx] <= fmap_save_l10[copy_idx];
                                else
                                    fmap_a[copy_idx] <= fmap_save_l10[copy_idx];
                            end
                            3'd5: begin  // save_l18
                                if (ping_pong == 1'b0)
                                    fmap_b[copy_idx] <= fmap_save_l18[copy_idx];
                                else
                                    fmap_a[copy_idx] <= fmap_save_l18[copy_idx];
                            end
                            default: begin
                                if (ping_pong == 1'b0)
                                    fmap_b[copy_idx] <= fmap_a[copy_idx];
                                else
                                    fmap_a[copy_idx] <= fmap_b[copy_idx];
                            end
                        endcase

                        if (copy_idx + 1 >= tmp_total_a) begin
                            copy_idx <= 0;
                            copy_total <= tmp_total_a;
                            state <= S_ROUTE_COPY_B;
                        end else begin
                            copy_idx <= copy_idx + 1;
                        end
                    end
                end

                S_ROUTE_COPY_B: begin
                    // src_b channels = cur_c_out - cur_c_in (supports unequal concat)
                    tmp_total_b = (cur_c_out - cur_c_in) * cur_h_out * cur_w_out;
                    tmp_dst_addr = copy_total + copy_idx;

                    case (cur_src_b)
                        3'd0: begin  // current fmap
                            if (ping_pong == 1'b0)
                                fmap_b[tmp_dst_addr] <= fmap_a[copy_idx];
                            else
                                fmap_a[tmp_dst_addr] <= fmap_b[copy_idx];
                        end
                        3'd2: begin  // save_l4
                            if (ping_pong == 1'b0)
                                fmap_b[tmp_dst_addr] <= fmap_save_l4[copy_idx];
                            else
                                fmap_a[tmp_dst_addr] <= fmap_save_l4[copy_idx];
                        end
                        3'd4: begin  // save_l12
                            if (ping_pong == 1'b0)
                                fmap_b[tmp_dst_addr] <= fmap_save_l12[copy_idx];
                            else
                                fmap_a[tmp_dst_addr] <= fmap_save_l12[copy_idx];
                        end
                        3'd5: begin  // save_l20
                            if (ping_pong == 1'b0)
                                fmap_b[tmp_dst_addr] <= fmap_save_l20[copy_idx];
                            else
                                fmap_a[tmp_dst_addr] <= fmap_save_l20[copy_idx];
                        end
                        3'd6: begin  // save_l23
                            if (ping_pong == 1'b0)
                                fmap_b[tmp_dst_addr] <= fmap_save_l23[copy_idx];
                            else
                                fmap_a[tmp_dst_addr] <= fmap_save_l23[copy_idx];
                        end
                        default: begin
                            if (ping_pong == 1'b0)
                                fmap_b[tmp_dst_addr] <= fmap_a[copy_idx];
                            else
                                fmap_a[tmp_dst_addr] <= fmap_b[copy_idx];
                        end
                    endcase

                    if (copy_idx + 1 >= tmp_total_b) begin
                        state <= S_SAVE_FMAP;
                    end else begin
                        copy_idx <= copy_idx + 1;
                    end
                end

                // =============================================================
                // UPSAMPLE 2x (nearest neighbor)
                // Reuses pool_ch, pool_oh, pool_ow for OUTPUT coordinates
                // =============================================================
                S_UPSAMPLE_COPY: begin
                    // Read from input at (pool_oh/2, pool_ow/2)
                    tmp_src_addr = pool_ch * cur_h_in * cur_w_in
                                 + (pool_oh >> 1) * cur_w_in
                                 + (pool_ow >> 1);
                    out_addr = pool_ch * cur_h_out * cur_w_out
                             + pool_oh * cur_w_out
                             + pool_ow;

                    if (ping_pong == 1'b0)
                        fmap_b[out_addr] <= fmap_a[tmp_src_addr];
                    else
                        fmap_a[out_addr] <= fmap_b[tmp_src_addr];

                    if (pool_ow + 1 < cur_w_out) begin
                        pool_ow <= pool_ow + 1;
                    end else begin
                        pool_ow <= 0;
                        if (pool_oh + 1 < cur_h_out) begin
                            pool_oh <= pool_oh + 1;
                        end else begin
                            pool_oh <= 0;
                            if (pool_ch + 1 < cur_c_out) begin
                                pool_ch <= pool_ch + 1;
                            end else begin
                                state <= S_SAVE_FMAP;
                            end
                        end
                    end
                end

                // =============================================================
                // ROUTE_SAVE: copy from a save buffer to output fmap
                // cur_src_a encodes which save buffer: 7=save_l27
                // =============================================================
                S_ROUTE_SAVE_COPY: begin
                    tmp_total_a = cur_c_out * cur_h_out * cur_w_out;

                    case (cur_src_a)
                        3'd5: begin  // save_l18
                            if (ping_pong == 1'b0)
                                fmap_b[copy_idx] <= fmap_save_l18[copy_idx];
                            else
                                fmap_a[copy_idx] <= fmap_save_l18[copy_idx];
                        end
                        3'd7: begin  // save_l27
                            if (ping_pong == 1'b0)
                                fmap_b[copy_idx] <= fmap_save_l27[copy_idx];
                            else
                                fmap_a[copy_idx] <= fmap_save_l27[copy_idx];
                        end
                        default: begin
                            if (ping_pong == 1'b0)
                                fmap_b[copy_idx] <= fmap_a[copy_idx];
                            else
                                fmap_a[copy_idx] <= fmap_b[copy_idx];
                        end
                    endcase

                    if (copy_idx + 1 >= tmp_total_a) begin
                        state <= S_SAVE_FMAP;
                    end else begin
                        copy_idx <= copy_idx + 1;
                    end
                end

                // =============================================================
                // SAVE
                // =============================================================
                S_SAVE_FMAP: begin
                    case (current_layer_reg)
                        6'd2: begin
                            if (save_idx < SAVE_L2_SIZE) begin
                                if (ping_pong == 1'b0)
                                    fmap_save_l2[save_idx] <= fmap_b[save_idx];
                                else
                                    fmap_save_l2[save_idx] <= fmap_a[save_idx];
                                save_idx <= save_idx + 1;
                            end else begin
                                state <= S_LAYER_DONE;
                            end
                        end
                        6'd4: begin
                            if (save_idx < SAVE_L4_SIZE) begin
                                if (ping_pong == 1'b0)
                                    fmap_save_l4[save_idx] <= fmap_b[save_idx];
                                else
                                    fmap_save_l4[save_idx] <= fmap_a[save_idx];
                                save_idx <= save_idx + 1;
                            end else begin
                                state <= S_LAYER_DONE;
                            end
                        end
                        6'd10: begin
                            if (save_idx < SAVE_L10_SIZE) begin
                                if (ping_pong == 1'b0)
                                    fmap_save_l10[save_idx] <= fmap_b[save_idx];
                                else
                                    fmap_save_l10[save_idx] <= fmap_a[save_idx];
                                save_idx <= save_idx + 1;
                            end else begin
                                state <= S_LAYER_DONE;
                            end
                        end
                        6'd12: begin
                            if (save_idx < SAVE_L12_SIZE) begin
                                if (ping_pong == 1'b0)
                                    fmap_save_l12[save_idx] <= fmap_b[save_idx];
                                else
                                    fmap_save_l12[save_idx] <= fmap_a[save_idx];
                                save_idx <= save_idx + 1;
                            end else begin
                                state <= S_LAYER_DONE;
                            end
                        end
                        6'd18: begin
                            if (save_idx < SAVE_L18_SIZE) begin
                                if (ping_pong == 1'b0)
                                    fmap_save_l18[save_idx] <= fmap_b[save_idx];
                                else
                                    fmap_save_l18[save_idx] <= fmap_a[save_idx];
                                save_idx <= save_idx + 1;
                            end else begin
                                state <= S_LAYER_DONE;
                            end
                        end
                        6'd20: begin
                            if (save_idx < SAVE_L20_SIZE) begin
                                if (ping_pong == 1'b0)
                                    fmap_save_l20[save_idx] <= fmap_b[save_idx];
                                else
                                    fmap_save_l20[save_idx] <= fmap_a[save_idx];
                                save_idx <= save_idx + 1;
                            end else begin
                                state <= S_LAYER_DONE;
                            end
                        end
                        6'd23: begin
                            if (save_idx < SAVE_L23_SIZE) begin
                                if (ping_pong == 1'b0)
                                    fmap_save_l23[save_idx] <= fmap_b[save_idx];
                                else
                                    fmap_save_l23[save_idx] <= fmap_a[save_idx];
                                save_idx <= save_idx + 1;
                            end else begin
                                state <= S_LAYER_DONE;
                            end
                        end
                        6'd27: begin
                            if (save_idx < SAVE_L27_SIZE) begin
                                if (ping_pong == 1'b0)
                                    fmap_save_l27[save_idx] <= fmap_b[save_idx];
                                else
                                    fmap_save_l27[save_idx] <= fmap_a[save_idx];
                                save_idx <= save_idx + 1;
                            end else begin
                                state <= S_LAYER_DONE;
                            end
                        end
                        default: state <= S_LAYER_DONE;
                    endcase
                end

                // =============================================================
                S_LAYER_DONE: begin
                    ping_pong <= ~ping_pong;
                    layer_cycles[current_layer_reg] <= cycle_counter - layer_start_cycle;

                    if (run_all_mode) begin
                        if (current_layer_reg == NUM_LAYERS - 1) begin
                            state <= S_ALL_DONE;
                        end else begin
                            current_layer_reg <= current_layer_reg + 6'd1;
                            // Conv layers need weight reload; route/maxpool/upsample auto-advance
                            if (ld_type[current_layer_reg + 6'd1] == LT_CONV3X3 ||
                                ld_type[current_layer_reg + 6'd1] == LT_CONV1X1 ||
                                ld_type[current_layer_reg + 6'd1] == LT_CONV1X1_LIN) begin
                                state <= S_LAYER_RELOAD;
                            end else begin
                                state <= S_LAYER_INIT;
                            end
                        end
                    end else begin
                        state <= S_ALL_DONE;
                    end
                end

                // =============================================================
                // LAYER_RELOAD: Wait for host to load weights, then continue
                // Accepts write_byte(0), write_scale(5), write_layer_desc(6)
                // run_layer(1) continues to next layer
                // =============================================================
                S_LAYER_RELOAD: begin
                    cmd_ready  <= 1'b1;
                    reload_req <= 1'b1;
                    if (cmd_valid) begin
                        case (cmd_type)
                            3'd0: begin // write_byte (weight/bias loading)
                                if (cmd_addr < MAX_WBUF) begin
                                    weight_buf[cmd_addr] <= $signed(cmd_data);
                                end else if (cmd_addr < MAX_WBUF + MAX_CH*4) begin
                                    tmp_boff = cmd_addr - MAX_WBUF;
                                    tmp_bch  = tmp_boff >> 2;
                                    tmp_bsel = tmp_boff & 3;
                                    case (tmp_bsel)
                                        0: bias_buf[tmp_bch][7:0]   <= cmd_data;
                                        1: bias_buf[tmp_bch][15:8]  <= cmd_data;
                                        2: bias_buf[tmp_bch][23:16] <= cmd_data;
                                        3: bias_buf[tmp_bch][31:24] <= cmd_data;
                                    endcase
                                end
                                state <= S_LAYER_RELOAD;
                                cmd_ready <= 1'b0;
                                // Need a 1-cycle ack then back to reload
                            end
                            3'd1: begin // run_layer -> continue to next layer
                                reload_req <= 1'b0;
                                state <= S_LAYER_INIT;
                            end
                            3'd2: begin // read_byte (check outputs during reload)
                                if (ping_pong == 1'b0)
                                    rsp_data <= fmap_a[cmd_addr];
                                else
                                    rsp_data <= fmap_b[cmd_addr];
                                rsp_valid <= 1'b1;
                                // Stay in LAYER_RELOAD after read
                                cmd_ready <= 1'b0;
                            end
                            3'd5: begin // write_scale
                                if (cmd_addr[0] == 1'b0) begin
                                    scale_reg[7:0]  <= cmd_data;
                                    ld_scale[current_layer_reg][7:0] <= cmd_data;
                                end else begin
                                    scale_reg[15:8] <= cmd_data;
                                    ld_scale[current_layer_reg][15:8] <= cmd_data;
                                end
                                state <= S_LAYER_RELOAD;
                                cmd_ready <= 1'b0;
                            end
                            3'd6: begin // write_layer_desc
                                case (cmd_addr[3:0])
                                    4'd0:  ld_type  [cmd_addr[9:4]] <= cmd_data[2:0];
                                    4'd1:  ld_c_in  [cmd_addr[9:4]][7:0]  <= cmd_data;
                                    4'd2:  ld_c_in  [cmd_addr[9:4]][10:8] <= cmd_data[2:0];
                                    4'd3:  ld_c_out [cmd_addr[9:4]][7:0]  <= cmd_data;
                                    4'd4:  ld_c_out [cmd_addr[9:4]][10:8] <= cmd_data[2:0];
                                    4'd5:  ld_h_in  [cmd_addr[9:4]][7:0]  <= cmd_data;
                                    4'd6:  ld_h_in  [cmd_addr[9:4]][10:8] <= cmd_data[2:0];
                                    4'd7:  ld_w_in  [cmd_addr[9:4]][7:0]  <= cmd_data;
                                    4'd8:  ld_w_in  [cmd_addr[9:4]][10:8] <= cmd_data[2:0];
                                    4'd9:  ld_h_out [cmd_addr[9:4]][7:0]  <= cmd_data;
                                    4'd10: ld_h_out [cmd_addr[9:4]][10:8] <= cmd_data[2:0];
                                    4'd11: ld_w_out [cmd_addr[9:4]][7:0]  <= cmd_data;
                                    4'd12: ld_w_out [cmd_addr[9:4]][10:8] <= cmd_data[2:0];
                                    4'd13: ld_stride[cmd_addr[9:4]] <= cmd_data[1:0];
                                    4'd14: ld_scale [cmd_addr[9:4]][7:0]  <= cmd_data;
                                    4'd15: ld_scale [cmd_addr[9:4]][15:8] <= cmd_data;
                                endcase
                                state <= S_LAYER_RELOAD;
                                cmd_ready <= 1'b0;
                            end
                            default: ;
                        endcase
                    end
                end

                S_ALL_DONE: begin
                    busy <= 1'b0;
                    done <= 1'b1;
                    state <= S_IDLE;
                end

                default: state <= S_IDLE;
            endcase
        end
    end

endmodule
`default_nettype wire
