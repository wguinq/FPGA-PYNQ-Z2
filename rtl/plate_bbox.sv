// Frame-level bounding box statistic.  It intentionally produces one union
// candidate; connected-component labeling is the next scalable refinement.
module plate_bbox #(
    parameter int IMAGE_WIDTH  = 640,
    parameter int IMAGE_HEIGHT = 480,
    parameter int MIN_WIDTH    = 48,
    parameter int MIN_HEIGHT   = 16
) (
    input logic clk, rst_n,
    input logic in_valid, in_sof, in_eol, in_pixel,
    output logic bbox_valid,
    output logic [$clog2(IMAGE_WIDTH)-1:0]  bbox_x,
    output logic [$clog2(IMAGE_HEIGHT)-1:0] bbox_y,
    output logic [$clog2(IMAGE_WIDTH):0]    bbox_w,
    output logic [$clog2(IMAGE_HEIGHT):0]   bbox_h
);
    localparam XW = $clog2(IMAGE_WIDTH);
    localparam YW = $clog2(IMAGE_HEIGHT);
    logic [XW-1:0] x, min_x, max_x;
    logic [YW-1:0] y, min_y, max_y;
    logic seen;
    logic [XW:0] cand_w;
    logic [YW:0] cand_h;

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            x <= '0; y <= '0; seen <= 1'b0; bbox_valid <= 1'b0;
            min_x <= '0; max_x <= '0; min_y <= '0; max_y <= '0;
            bbox_x <= '0; bbox_y <= '0; bbox_w <= '0; bbox_h <= '0;
        end else if (in_valid) begin
            if (in_sof) begin
                x <= '0; y <= '0; seen <= 1'b0; bbox_valid <= 1'b0;
            end
            if (in_pixel) begin
                if (!seen) begin
                    min_x <= x; max_x <= x; min_y <= y; max_y <= y; seen <= 1'b1;
                end else begin
                    if (x < min_x) min_x <= x;
                    if (x > max_x) max_x <= x;
                    if (y < min_y) min_y <= y;
                    if (y > max_y) max_y <= y;
                end
            end
            if (in_eol) begin
                x <= '0;
                if (y == IMAGE_HEIGHT-1) begin
                    cand_w = max_x - min_x + 1'b1;
                    cand_h = max_y - min_y + 1'b1;
                    if (seen && cand_w >= MIN_WIDTH && cand_h >= MIN_HEIGHT) begin
                        bbox_x <= min_x; bbox_y <= min_y;
                        bbox_w <= cand_w; bbox_h <= cand_h;
                        bbox_valid <= 1'b1;
                    end
                    y <= '0;
                end else y <= y + 1'b1;
            end else x <= x + 1'b1;
        end
    end
endmodule
