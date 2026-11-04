`timescale 1ns/1ps
// SW0 selects a self-contained VGA color bar signal at the TMDS encoder.
// With SW0 low, pass through the normal Video Out pixels. The Xilinx bridge
// exposes active-high sync; the 640x480 DVI/VGA signal uses active-low sync.
module hdmi_diag_mux (
    input  wire        pixel_clk,
    input  wire        resetn,
    input  wire        select_pattern,
    input  wire [23:0] in_data,
    input  wire        in_vde,
    input  wire        in_hsync,
    input  wire        in_vsync,
    output wire [23:0] out_data,
    output wire        out_vde,
    output wire        out_hsync,
    output wire        out_vsync
);
    localparam integer H_ACTIVE = 640;
    localparam integer H_TOTAL  = 800;
    localparam integer V_ACTIVE = 480;
    localparam integer V_TOTAL  = 525;

    reg [9:0] h = 0;
    reg [9:0] v = 0;
    reg select_meta = 0;
    reg select_sync = 0;
    reg select_mode = 0;

    always @(posedge pixel_clk) begin
        if (!resetn) begin
            h <= 0;
            v <= 0;
            select_meta <= 0;
            select_sync <= 0;
            select_mode <= 0;
        end else begin
            select_meta <= select_pattern;
            select_sync <= select_meta;
            if (h == H_TOTAL - 1) begin
                h <= 0;
                if (v == V_TOTAL - 1) begin
                    v <= 0;
                    select_mode <= select_sync;
                end else begin
                    v <= v + 1'b1;
                end
            end else begin
                h <= h + 1'b1;
            end
        end
    end

    wire pattern_vde = h < H_ACTIVE && v < V_ACTIVE;
    wire pattern_hsync_n = !(h >= 656 && h < 752);
    wire pattern_vsync_n = !(v >= 490 && v < 492);
    // rgb2dvi maps input [23:16] to R, [7:0] to G, [15:8] to B.
    wire [23:0] pattern_data = h < 213 ? 24'hff0000 :
                               h < 426 ? 24'h0000ff : 24'h00ff00;

    assign out_data  = select_mode ? pattern_data    : in_data;
    assign out_vde   = select_mode ? pattern_vde     : in_vde;
    assign out_hsync = select_mode ? pattern_hsync_n : ~in_hsync;
    assign out_vsync = select_mode ? pattern_vsync_n : ~in_vsync;
endmodule
