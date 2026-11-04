// Extracts one configured binary ROI from a raster stream.  Feed character
// foreground pixels (not the colour plate mask) to in_pixel.  The bbox should
// be latched only at a frame boundary; its coordinates are normally supplied
// by the previous-frame detector or a VDMA second pass.
module roi_cropper #(
    parameter int IMAGE_WIDTH = 640,
    parameter int IMAGE_HEIGHT = 480
) (
    input  logic clk, rst_n,
    input  logic in_valid, in_sof, in_eol, in_pixel,
    input  logic [$clog2(IMAGE_WIDTH)-1:0] roi_x,
    input  logic [$clog2(IMAGE_HEIGHT)-1:0] roi_y,
    input  logic [$clog2(IMAGE_WIDTH):0] roi_width,
    input  logic [$clog2(IMAGE_HEIGHT):0] roi_height,
    input  logic roi_enable,
    output logic out_valid, out_sof, out_eol, out_eof, out_pixel
);
    localparam XW = $clog2(IMAGE_WIDTH);
    localparam YW = $clog2(IMAGE_HEIGHT);
    logic [XW-1:0] x;
    logic [YW-1:0] y;
    logic [XW:0] right_edge;
    logic [YW:0] bottom_edge;
    logic in_roi;

    always_comb begin
        right_edge = {1'b0, roi_x} + roi_width;
        bottom_edge = {1'b0, roi_y} + roi_height;
        in_roi = roi_enable && ({1'b0, x} >= {1'b0, roi_x}) &&
                 ({1'b0, x} < right_edge) && ({1'b0, y} >= {1'b0, roi_y}) &&
                 ({1'b0, y} < bottom_edge);
    end

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            x <= '0;
            y <= '0;
            out_valid <= 1'b0;
            out_sof <= 1'b0;
            out_eol <= 1'b0;
            out_eof <= 1'b0;
            out_pixel <= 1'b0;
        end else begin
            out_valid <= 1'b0;
            out_sof <= 1'b0;
            out_eol <= 1'b0;
            out_eof <= 1'b0;
            if (in_valid) begin
                if (in_sof) begin
                    x <= '0;
                    y <= '0;
                end
                if (in_roi) begin
                    out_valid <= 1'b1;
                    out_sof <= (x == roi_x) && (y == roi_y);
                    out_eol <= ({1'b0,x} == right_edge - 1'b1);
                    out_eof <= ({1'b0,x} == right_edge - 1'b1) &&
                               ({1'b0,y} == bottom_edge - 1'b1);
                    out_pixel <= in_pixel;
                end
                if (in_eol) begin
                    x <= '0;
                    if (y == IMAGE_HEIGHT-1) y <= '0;
                    else                     y <= y + 1'b1;
                end else begin
                    x <= x + 1'b1;
                end
            end
        end
    end
endmodule
