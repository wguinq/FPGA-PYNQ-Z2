// Approximate HSV classifier for one RGB888 pixel.  Division is avoided so the
// module maps to comparators/subtractors instead of a costly floating point unit.
module plate_color_mask #(
    parameter int SAT_MIN      = 48,
    parameter int VAL_MIN      = 40,
    parameter int DARK_VAL_MIN = 18
) (
    input  logic       clk,
    input  logic       rst_n,
    input  logic       in_valid,
    input  logic [7:0] r,
    input  logic [7:0] g,
    input  logic [7:0] b,
    input  logic [1:0] color_mode, // bit 0: accept blue, bit 1: accept yellow/green
    input  logic       night_mode,
    output logic       out_valid,
    output logic       plate_pixel
);
    logic [7:0] maxc, minc, chroma, value_floor;
    logic blue, yellow, green;

    always_comb begin
        maxc = r;
        if (g > maxc) maxc = g;
        if (b > maxc) maxc = b;
        minc = r;
        if (g < minc) minc = g;
        if (b < minc) minc = b;
        chroma = maxc - minc;
        value_floor = night_mode ? DARK_VAL_MIN[7:0] : VAL_MIN[7:0];

        // Ratios are replaced by channel separation margins. Thresholds are
        // deliberately AXI-Lite configurable when this module is packaged.
        // Extend operands before adding margins; 8-bit addition would wrap
        // around near 255 and incorrectly classify bright neutral pixels.
        blue   = ({1'b0,b} > {1'b0,g} + 9'd20) && ({1'b0,b} > {1'b0,r} + 9'd28) &&
                 (chroma >= SAT_MIN) && (maxc >= value_floor);
        yellow = ({1'b0,r} > {1'b0,b} + 9'd32) && ({1'b0,g} > {1'b0,b} + 9'd20) &&
                 (chroma >= SAT_MIN) && (maxc >= value_floor);
        green  = ({1'b0,g} > {1'b0,r} + 9'd18) && ({1'b0,g} > {1'b0,b} + 9'd18) &&
                 (chroma >= SAT_MIN) && (maxc >= value_floor);
    end

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            out_valid   <= 1'b0;
            plate_pixel <= 1'b0;
        end else begin
            out_valid   <= in_valid;
            plate_pixel <= (color_mode[0] && blue) ||
                           (color_mode[1] && (yellow || green));
        end
    end
endmodule
