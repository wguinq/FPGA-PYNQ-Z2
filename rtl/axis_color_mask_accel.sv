`timescale 1ns/1ps
// OpenCV-compatible Cb>150 proposal mask for RGBX AXI4-Stream pixels.
// Byte convention: [7:0]=R, [15:8]=G, [23:16]=B, [31:24]=X.
//
// OpenCV's 8-bit BT.601 YCrCb path computes
//   Y  = round((4899*R + 9617*G + 1868*B) / 16384)
//   Cb = 128 + round((B-Y) * 9241 / 16384).
// For integer B-Y, Cb>150 is exactly equivalent to B-Y>=40. This keeps the
// PL decision equivalent to OpenCV while avoiding the second Cb multiplier.
// The conversion coefficients and decision were exhaustively checked for all
// 2^24 RGB values against cv2.COLOR_BGR2YCrCb.
module axis_color_mask_accel (
    input wire [1:0] plate_mode,
    (* X_INTERFACE_INFO = "xilinx.com:signal:clock:1.0 aclk CLK" *)
    (* X_INTERFACE_PARAMETER = "ASSOCIATED_BUSIF S_AXIS:M_AXIS, ASSOCIATED_RESET aresetn" *)
    input  wire        aclk,
    (* X_INTERFACE_INFO = "xilinx.com:signal:reset:1.0 aresetn RST" *)
    (* X_INTERFACE_PARAMETER = "POLARITY ACTIVE_LOW" *)
    input  wire        aresetn,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 S_AXIS TDATA" *)  input  wire [31:0] s_axis_tdata,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 S_AXIS TKEEP" *)  input  wire [3:0]  s_axis_tkeep,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 S_AXIS TVALID" *) input  wire        s_axis_tvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 S_AXIS TREADY" *) output wire        s_axis_tready,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 S_AXIS TLAST" *)
    (* X_INTERFACE_PARAMETER = "XIL_INTERFACENAME S_AXIS, TDATA_NUM_BYTES 4, HAS_TKEEP 1, HAS_TLAST 1" *)
    input  wire        s_axis_tlast,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 M_AXIS TDATA" *)  output reg [31:0] m_axis_tdata,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 M_AXIS TKEEP" *)  output reg [3:0]  m_axis_tkeep,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 M_AXIS TVALID" *) output reg        m_axis_tvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 M_AXIS TREADY" *) input  wire        m_axis_tready,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 M_AXIS TLAST" *)
    (* X_INTERFACE_PARAMETER = "XIL_INTERFACENAME M_AXIS, TDATA_NUM_BYTES 4, HAS_TKEEP 1, HAS_TLAST 1" *)
    output reg         m_axis_tlast
);
    wire [7:0] r = s_axis_tdata[7:0];
    wire [7:0] g = s_axis_tdata[15:8];
    wire [7:0] b = s_axis_tdata[23:16];

    // Three elastic stages split the Q14 conversion across clock cycles.
    // Every sideband travels with its pixel, including during backpressure.
    reg [22:0] product_r, product_g, product_b;
    reg [7:0]  blue_stage0, x_stage0;
    reg [3:0]  keep_stage0;
    reg        last_stage0, valid_stage0;

    reg [22:0] y_rounded_stage1;
    reg [7:0]  blue_stage1, x_stage1;
    reg [3:0]  keep_stage1;
    reg        last_stage1, valid_stage1;

    wire [7:0] y_stage1 = y_rounded_stage1[21:14];
    wire signed [8:0] blue_minus_y =
        $signed({1'b0, blue_stage1}) - $signed({1'b0, y_stage1});
    // Division-free continuous HSV green band H=35..95 degrees in OpenCV
    // units (70..190 degrees), saturation >=35/255, value >=45.
    wire [7:0] vmax = (r > g) ? ((r > b) ? r : b) : ((g > b) ? g : b);
    wire [7:0] vmin = (r < g) ? ((r < b) ? r : b) : ((g < b) ? g : b);
    wire [8:0] delta = {1'b0,vmax} - {1'b0,vmin};
    reg [7:0] vmax_green0;
    reg [8:0] delta_green0, diff_rb0, diff_gr0;
    reg green_sector0, cyan_sector0, r_le_b0;
    wire green_input = vmax_green0 >= 45 && delta_green0 != 0 &&
        delta_green0 * 16'd255 >= vmax_green0 * 16'd35 &&
        ((green_sector0 && (r_le_b0 || diff_rb0 * 16'd6 <= delta_green0 * 16'd5)) ||
         (cyan_sector0 && diff_gr0 * 16'd6 >= delta_green0 * 16'd5));
    reg green_stage1;
    reg [1:0] mode_stage0, mode_stage1;
    wire blue_hit = blue_minus_y >= 9'sd40;
    wire hit = mode_stage1 == 1 ? green_stage1 :
               mode_stage1 == 2 ? (blue_hit || green_stage1) : blue_hit;
    wire [7:0] mask = hit ? 8'hff : 8'h00;

    wire output_ready = !m_axis_tvalid || m_axis_tready;
    wire stage1_ready = !valid_stage1 || output_ready;
    wire stage0_ready = !valid_stage0 || stage1_ready;
    assign s_axis_tready = stage0_ready;

    always @(posedge aclk) begin
        if (!aresetn) begin
            valid_stage0 <= 1'b0;
            valid_stage1 <= 1'b0;
            m_axis_tdata  <= 32'd0;
            m_axis_tkeep  <= 4'd0;
            m_axis_tvalid <= 1'b0;
            m_axis_tlast  <= 1'b0;
        end else begin
            if (output_ready) begin
                m_axis_tvalid <= valid_stage1;
                if (valid_stage1) begin
                    m_axis_tdata <= {x_stage1, mask, mask, mask};
                    m_axis_tkeep <= keep_stage1;
                    m_axis_tlast <= last_stage1;
                end
            end
            if (stage1_ready) begin
                valid_stage1 <= valid_stage0;
                if (valid_stage0) begin
                    y_rounded_stage1 <= product_r + product_g + product_b + 23'd8192;
                    blue_stage1 <= blue_stage0;
                    green_stage1 <= green_input;
                    mode_stage1 <= mode_stage0;
                    x_stage1 <= x_stage0;
                    keep_stage1 <= keep_stage0;
                    last_stage1 <= last_stage0;
                end
            end
            if (stage0_ready) begin
                valid_stage0 <= s_axis_tvalid;
                if (s_axis_tvalid) begin
                    product_r <= r * 23'd4899;
                    product_g <= g * 23'd9617;
                    product_b <= b * 23'd1868;
                    blue_stage0 <= b;
                    vmax_green0 <= vmax;
                    delta_green0 <= delta;
                    diff_rb0 <= {1'b0,r} - {1'b0,b};
                    diff_gr0 <= {1'b0,g} - {1'b0,r};
                    green_sector0 <= g >= r && g >= b;
                    cyan_sector0 <= b > g && g >= r;
                    r_le_b0 <= r <= b;
                    mode_stage0 <= plate_mode;
                    x_stage0 <= s_axis_tdata[31:24];
                    keep_stage0 <= s_axis_tkeep;
                    last_stage0 <= s_axis_tlast;
                end
            end
        end
    end
endmodule
