`timescale 1ns/1ps
// 32-bit RGBX/gray AXI DMA frame -> 3x3 Sobel magnitude, one accepted beat
// per pixel. TLAST marks end of the complete frame, not end of line.
// The causal 3x3 window is reported at its bottom-right pixel, so a valid
// result at (x,y) represents the conventional Sobel center (x-1,y-1).
module axis_sobel_accel #(
    parameter integer IMAGE_WIDTH = 640,
    parameter integer IMAGE_HEIGHT = 480
) (
    (* X_INTERFACE_INFO = "xilinx.com:signal:clock:1.0 aclk CLK" *)
    (* X_INTERFACE_PARAMETER = "ASSOCIATED_BUSIF S_AXIS:M_AXIS, ASSOCIATED_RESET aresetn" *)
    input wire aclk,
    (* X_INTERFACE_INFO = "xilinx.com:signal:reset:1.0 aresetn RST" *)
    (* X_INTERFACE_PARAMETER = "POLARITY ACTIVE_LOW" *)
    input wire aresetn,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 S_AXIS TDATA" *) input wire [31:0] s_axis_tdata,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 S_AXIS TKEEP" *) input wire [3:0] s_axis_tkeep,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 S_AXIS TVALID" *) input wire s_axis_tvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 S_AXIS TREADY" *) output wire s_axis_tready,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 S_AXIS TLAST" *) input wire s_axis_tlast,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 M_AXIS TDATA" *) output reg [31:0] m_axis_tdata,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 M_AXIS TKEEP" *) output reg [3:0] m_axis_tkeep,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 M_AXIS TVALID" *) output reg m_axis_tvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 M_AXIS TREADY" *) input wire m_axis_tready,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 M_AXIS TLAST" *) output reg m_axis_tlast
);
    localparam integer X_BITS = (IMAGE_WIDTH <= 1) ? 1 : $clog2(IMAGE_WIDTH);
    localparam integer Y_BITS = (IMAGE_HEIGHT <= 1) ? 1 : $clog2(IMAGE_HEIGHT);
    reg [X_BITS-1:0] x;
    reg [Y_BITS-1:0] y;
    reg [7:0] line_1 [0:IMAGE_WIDTH-1];
    reg [7:0] line_2 [0:IMAGE_WIDTH-1];
    reg [7:0] top_left, top_mid, mid_left, mid_mid, bot_left, bot_mid;
    wire [7:0] top_right = line_2[x];
    wire [7:0] mid_right = line_1[x];
    wire [7:0] bot_right = s_axis_tdata[7:0];
    integer signed gx, gy, ax, ay, magnitude;
    reg [7:0] edge_value;

    always @* begin
        gx = (top_right + 2*mid_right + bot_right) - (top_left + 2*mid_left + bot_left);
        gy = (bot_left + 2*bot_mid + bot_right) - (top_left + 2*top_mid + top_right);
        ax = (gx < 0) ? -gx : gx;
        ay = (gy < 0) ? -gy : gy;
        magnitude = ax + ay;
        if ((x < 2) || (y < 2)) edge_value = 0;
        else if (magnitude > 255) edge_value = 255;
        else edge_value = magnitude[7:0];
    end

    assign s_axis_tready = !m_axis_tvalid || m_axis_tready;
    always @(posedge aclk) begin
        if (!aresetn) begin
            x <= 0; y <= 0;
            top_left <= 0; top_mid <= 0; mid_left <= 0; mid_mid <= 0;
            bot_left <= 0; bot_mid <= 0;
            m_axis_tvalid <= 0; m_axis_tdata <= 0; m_axis_tkeep <= 0; m_axis_tlast <= 0;
        end else if (s_axis_tready) begin
            m_axis_tvalid <= s_axis_tvalid;
            if (s_axis_tvalid) begin
                line_1[x] <= bot_right;
                line_2[x] <= mid_right;
                top_left <= top_mid; top_mid <= top_right;
                mid_left <= mid_mid; mid_mid <= mid_right;
                bot_left <= bot_mid; bot_mid <= bot_right;
                m_axis_tdata <= {s_axis_tdata[31:24], edge_value, edge_value, edge_value};
                m_axis_tkeep <= s_axis_tkeep;
                m_axis_tlast <= s_axis_tlast;
                if (s_axis_tlast) begin
                    x <= 0; y <= 0;
                end else if (x == IMAGE_WIDTH-1) begin
                    x <= 0;
                    if (y == IMAGE_HEIGHT-1) y <= 0;
                    else y <= y + 1'b1;
                end else x <= x + 1'b1;
            end
        end
    end
endmodule
