`timescale 1ns/1ps
// Four-stage binary morphology for a frame-wide AXI stream:
// dilate(CLOSE_W x CLOSE_H) -> erode(CLOSE_W x CLOSE_H)
// -> erode(OPEN_W x OPEN_H) -> dilate(OPEN_W x OPEN_H).
// Each stage is causal (window result reported at its bottom-right pixel).
module axis_binary_morph_accel #(
    parameter integer IMAGE_WIDTH = 640,
    parameter integer IMAGE_HEIGHT = 480,
    parameter integer CLOSE_W = 25,
    parameter integer CLOSE_H = 9,
    parameter integer OPEN_W = 3,
    parameter integer OPEN_H = 3
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
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 M_AXIS TDATA" *) output wire [31:0] m_axis_tdata,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 M_AXIS TKEEP" *) output wire [3:0] m_axis_tkeep,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 M_AXIS TVALID" *) output wire m_axis_tvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 M_AXIS TREADY" *) input wire m_axis_tready,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 M_AXIS TLAST" *) output wire m_axis_tlast
);
    wire [31:0] d1_data, e1_data, e2_data;
    wire [3:0] d1_keep, e1_keep, e2_keep;
    wire d1_valid, e1_valid, e2_valid;
    wire d1_ready, e1_ready, e2_ready;
    wire d1_last, e1_last, e2_last;

    axis_rect_morph_stage #(.IMAGE_WIDTH(IMAGE_WIDTH), .IMAGE_HEIGHT(IMAGE_HEIGHT),
        .KERNEL_W(CLOSE_W), .KERNEL_H(CLOSE_H), .DILATE(1)) close_dilate (
        .aclk(aclk), .aresetn(aresetn),
        .s_axis_tdata(s_axis_tdata), .s_axis_tkeep(s_axis_tkeep), .s_axis_tvalid(s_axis_tvalid),
        .s_axis_tready(s_axis_tready), .s_axis_tlast(s_axis_tlast),
        .m_axis_tdata(d1_data), .m_axis_tkeep(d1_keep), .m_axis_tvalid(d1_valid),
        .m_axis_tready(d1_ready), .m_axis_tlast(d1_last));

    axis_rect_morph_stage #(.IMAGE_WIDTH(IMAGE_WIDTH), .IMAGE_HEIGHT(IMAGE_HEIGHT),
        .KERNEL_W(CLOSE_W), .KERNEL_H(CLOSE_H), .DILATE(0)) close_erode (
        .aclk(aclk), .aresetn(aresetn),
        .s_axis_tdata(d1_data), .s_axis_tkeep(d1_keep), .s_axis_tvalid(d1_valid),
        .s_axis_tready(d1_ready), .s_axis_tlast(d1_last),
        .m_axis_tdata(e1_data), .m_axis_tkeep(e1_keep), .m_axis_tvalid(e1_valid),
        .m_axis_tready(e1_ready), .m_axis_tlast(e1_last));

    axis_rect_morph_stage #(.IMAGE_WIDTH(IMAGE_WIDTH), .IMAGE_HEIGHT(IMAGE_HEIGHT),
        .KERNEL_W(OPEN_W), .KERNEL_H(OPEN_H), .DILATE(0)) open_erode (
        .aclk(aclk), .aresetn(aresetn),
        .s_axis_tdata(e1_data), .s_axis_tkeep(e1_keep), .s_axis_tvalid(e1_valid),
        .s_axis_tready(e1_ready), .s_axis_tlast(e1_last),
        .m_axis_tdata(e2_data), .m_axis_tkeep(e2_keep), .m_axis_tvalid(e2_valid),
        .m_axis_tready(e2_ready), .m_axis_tlast(e2_last));

    axis_rect_morph_stage #(.IMAGE_WIDTH(IMAGE_WIDTH), .IMAGE_HEIGHT(IMAGE_HEIGHT),
        .KERNEL_W(OPEN_W), .KERNEL_H(OPEN_H), .DILATE(1)) open_dilate (
        .aclk(aclk), .aresetn(aresetn),
        .s_axis_tdata(e2_data), .s_axis_tkeep(e2_keep), .s_axis_tvalid(e2_valid),
        .s_axis_tready(e2_ready), .s_axis_tlast(e2_last),
        .m_axis_tdata(m_axis_tdata), .m_axis_tkeep(m_axis_tkeep), .m_axis_tvalid(m_axis_tvalid),
        .m_axis_tready(m_axis_tready), .m_axis_tlast(m_axis_tlast));
endmodule

// One elastic, one-pixel-per-cycle rectangular binary morphology stage.
module axis_rect_morph_stage #(
    parameter integer IMAGE_WIDTH = 640,
    parameter integer IMAGE_HEIGHT = 480,
    parameter integer KERNEL_W = 3,
    parameter integer KERNEL_H = 3,
    parameter integer DILATE = 1
) (
    input wire aclk, input wire aresetn,
    input wire [31:0] s_axis_tdata, input wire [3:0] s_axis_tkeep,
    input wire s_axis_tvalid, output wire s_axis_tready, input wire s_axis_tlast,
    output reg [31:0] m_axis_tdata, output reg [3:0] m_axis_tkeep,
    output reg m_axis_tvalid, input wire m_axis_tready, output reg m_axis_tlast
);
    localparam integer X_BITS = (IMAGE_WIDTH <= 1) ? 1 : $clog2(IMAGE_WIDTH);
    localparam integer Y_BITS = (IMAGE_HEIGHT <= 1) ? 1 : $clog2(IMAGE_HEIGHT);
    reg [X_BITS-1:0] x;
    reg [Y_BITS-1:0] y;
    reg line_buffer [0:KERNEL_H-2][0:IMAGE_WIDTH-1];
    reg history [0:KERNEL_H-1][0:KERNEL_W-2];
    integer row, col;
    reg morph_value;
    reg column_value;
    wire pixel_in = s_axis_tdata[0];
    wire [7:0] mask_byte = morph_value ? 8'hff : 8'h00;

    always @* begin
        morph_value = DILATE ? 1'b0 : 1'b1;
        for (row = 0; row < KERNEL_H; row = row + 1) begin
            for (col = 0; col < KERNEL_W-1; col = col + 1) begin
                if (DILATE) morph_value = morph_value | history[row][col];
                else        morph_value = morph_value & history[row][col];
            end
            if (row == KERNEL_H-1) column_value = pixel_in;
            else column_value = line_buffer[KERNEL_H-2-row][x];
            if (DILATE) morph_value = morph_value | column_value;
            else        morph_value = morph_value & column_value;
        end
        // Suppress stale/unfilled state at the top and left frame boundaries.
        if ((x < KERNEL_W-1) || (y < KERNEL_H-1)) morph_value = 1'b0;
    end

    assign s_axis_tready = !m_axis_tvalid || m_axis_tready;
    always @(posedge aclk) begin
        if (!aresetn) begin
            x <= 0;
            y <= 0;
            m_axis_tdata <= 0;
            m_axis_tkeep <= 0;
            m_axis_tvalid <= 0;
            m_axis_tlast <= 0;
        end else if (s_axis_tready) begin
            m_axis_tvalid <= s_axis_tvalid;
            if (s_axis_tvalid) begin
                m_axis_tdata <= {s_axis_tdata[31:24], mask_byte, mask_byte, mask_byte};
                m_axis_tkeep <= s_axis_tkeep;
                m_axis_tlast <= s_axis_tlast;
                for (row = KERNEL_H-2; row > 0; row = row - 1)
                    line_buffer[row][x] <= line_buffer[row-1][x];
                line_buffer[0][x] <= pixel_in;
                for (row = 0; row < KERNEL_H; row = row + 1) begin
                    for (col = 0; col < KERNEL_W-2; col = col + 1)
                        history[row][col] <= history[row][col+1];
                    if (row == KERNEL_H-1) history[row][KERNEL_W-2] <= pixel_in;
                    else history[row][KERNEL_W-2] <= line_buffer[KERNEL_H-2-row][x];
                end
                if (s_axis_tlast) begin
                    x <= 0;
                    y <= 0;
                end else if (x == IMAGE_WIDTH-1) begin
                    x <= 0;
                    if (y == IMAGE_HEIGHT-1) y <= 0;
                    else y <= y + 1'b1;
                end else begin
                    x <= x + 1'b1;
                end
            end
        end
    end
endmodule
