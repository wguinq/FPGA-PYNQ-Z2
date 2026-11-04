// Nearest-neighbour hardware resampler.  It reads a previously buffered binary
// ROI and packs the selected character into a fixed 16x32 bit vector for the
// classifier.  source_pixel is combinational here; with synchronous BRAM,
// register source_x/source_y and delay the write-enable by one cycle.
module char_resampler #(
    parameter int MAX_ROI_WIDTH = 192,
    parameter int MAX_ROI_HEIGHT = 64,
    parameter int OUT_WIDTH = 16,
    parameter int OUT_HEIGHT = 32
) (
    input  logic clk, rst_n,
    input  logic start,
    input  logic [$clog2(MAX_ROI_WIDTH)-1:0] char_x_start,
    input  logic [$clog2(MAX_ROI_WIDTH)-1:0] char_x_end,
    input  logic [$clog2(MAX_ROI_HEIGHT):0] roi_height,
    output logic [$clog2(MAX_ROI_WIDTH)-1:0] source_x,
    output logic [$clog2(MAX_ROI_HEIGHT)-1:0] source_y,
    input  logic source_pixel,
    output logic busy,
    output logic done,
    output logic [OUT_WIDTH*OUT_HEIGHT-1:0] sample_bits
);
    localparam XW = $clog2(MAX_ROI_WIDTH);
    localparam YW = $clog2(MAX_ROI_HEIGHT);
    localparam OXW = (OUT_WIDTH <= 1) ? 1 : $clog2(OUT_WIDTH);
    localparam OYW = (OUT_HEIGHT <= 1) ? 1 : $clog2(OUT_HEIGHT);
    localparam OW = OUT_WIDTH * OUT_HEIGHT;
    logic [OXW-1:0] out_x;
    logic [OYW-1:0] out_y;
    logic [XW-1:0] char_x_start_reg, char_x_end_reg;
    logic [YW:0] roi_height_reg;
    logic [XW:0] char_width;
    logic [XW+OXW:0] scaled_x;
    logic [YW+OYW:0] scaled_y;

    always_comb begin
        char_width = char_x_end_reg - char_x_start_reg + 1'b1;
        // OUT_WIDTH/OUT_HEIGHT are powers of two: division becomes a shift.
        scaled_x = out_x * char_width;
        scaled_y = out_y * roi_height_reg;
        source_x = char_x_start_reg + (scaled_x >> $clog2(OUT_WIDTH));
        source_y = scaled_y >> $clog2(OUT_HEIGHT);
    end

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            busy <= 1'b0;
            done <= 1'b0;
            out_x <= '0;
            out_y <= '0;
            char_x_start_reg <= '0;
            char_x_end_reg <= '0;
            roi_height_reg <= '0;
            sample_bits <= '0;
        end else begin
            done <= 1'b0;
            if (start && !busy) begin
                busy <= 1'b1;
                out_x <= '0;
                out_y <= '0;
                char_x_start_reg <= char_x_start;
                char_x_end_reg <= char_x_end;
                roi_height_reg <= roi_height;
                sample_bits <= '0;
            end else if (busy) begin
                sample_bits[out_y*OUT_WIDTH + out_x] <= source_pixel;
                if (out_x == OUT_WIDTH-1) begin
                    out_x <= '0;
                    if (out_y == OUT_HEIGHT-1) begin
                        busy <= 1'b0;
                        done <= 1'b1;
                    end else begin
                        out_y <= out_y + 1'b1;
                    end
                end else begin
                    out_x <= out_x + 1'b1;
                end
            end
        end
    end
endmodule
