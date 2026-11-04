// Converts a vertical foreground projection into character x-ranges.  The
// client supplies one projection count per column and pulses start after the
// ROI has been accumulated.  char_valid is a one-cycle event for each range.
module char_segmenter #(
    parameter int MAX_ROI_WIDTH = 192,
    parameter int MAX_ROI_HEIGHT = 64
) (
    input  logic clk, rst_n,
    input  logic start,
    input  logic [$clog2(MAX_ROI_WIDTH):0] roi_width,
    input  logic [$clog2(MAX_ROI_WIDTH)-1:0] projection_addr,
    input  logic [$clog2(MAX_ROI_HEIGHT+1)-1:0] projection_count,
    input  logic [$clog2(MAX_ROI_HEIGHT+1)-1:0] foreground_threshold,
    input  logic [$clog2(MAX_ROI_WIDTH):0] min_char_width,
    output logic char_valid,
    output logic [$clog2(MAX_ROI_WIDTH)-1:0] char_x_start,
    output logic [$clog2(MAX_ROI_WIDTH)-1:0] char_x_end,
    output logic [3:0] char_index,
    output logic done
);
    localparam XW = $clog2(MAX_ROI_WIDTH);
    logic scanning, in_char;
    logic [3:0] next_char_index;
    logic [XW-1:0] x, start_x;
    logic foreground;
    logic [XW:0] width_at_boundary;

    assign projection_addr = x;
    always_comb begin
        foreground = (projection_count >= foreground_threshold);
        width_at_boundary = foreground ? (x - start_x + 1'b1) : (x - start_x);
    end

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            scanning <= 1'b0; in_char <= 1'b0; x <= '0; start_x <= '0;
            char_valid <= 1'b0; char_x_start <= '0; char_x_end <= '0;
            char_index <= '0; next_char_index <= '0; done <= 1'b0;
        end else begin
            char_valid <= 1'b0;
            done <= 1'b0;
            if (start && !scanning) begin
                scanning <= 1'b1;
                in_char <= 1'b0;
                x <= '0;
                next_char_index <= '0;
            end else if (scanning) begin
                if (!in_char && foreground) begin
                    in_char <= 1'b1;
                    start_x <= x;
                end else if (in_char && (!foreground || x == roi_width-1)) begin
                    // At the final foreground column, include that column in
                    // the reported range.  At a gap, the range ends at x-1.
                    if (width_at_boundary >= min_char_width) begin
                        char_valid <= 1'b1;
                        char_x_start <= start_x;
                        char_x_end <= foreground ? x : x - 1'b1;
                        char_index <= next_char_index;
                        next_char_index <= next_char_index + 1'b1;
                    end
                    in_char <= 1'b0;
                end
                if (x == roi_width-1) begin
                    scanning <= 1'b0;
                    done <= 1'b1;
                end else begin
                    x <= x + 1'b1;
                end
            end
        end
    end
endmodule
