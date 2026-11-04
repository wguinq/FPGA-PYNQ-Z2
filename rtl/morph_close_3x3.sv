// One-pixel-per-clock 3x3 close (dilate then erode).  RAM inference is used for
// the two preceding lines.  It is a causal (top-left anchored) window: its
// output timing remains pixel-for-pixel aligned with the input AXIS frame.
module morph_close_3x3 #(
    parameter int IMAGE_WIDTH = 640
) (
    input  logic clk, rst_n,
    input  logic in_valid, in_sof, in_eol, in_pixel,
    output logic out_valid, out_sof, out_eol, out_pixel
);
    logic line0 [0:IMAGE_WIDTH-1];
    logic line1 [0:IMAGE_WIDTH-1];
    logic [2:0] d0, d1, d2;
    logic [2:0] e0, e1, e2;
    logic [$clog2(IMAGE_WIDTH)-1:0] col;
    logic p0, p1, dilated;

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            col <= '0; d0 <= '0; d1 <= '0; d2 <= '0;
            e0 <= '0; e1 <= '0; e2 <= '0;
            out_valid <= 1'b0; out_sof <= 1'b0; out_eol <= 1'b0; out_pixel <= 1'b0;
        end else if (in_valid) begin
            p0 = line0[col];
            p1 = line1[col];
            line0[col] <= in_pixel;
            line1[col] <= p0;
            d0 <= {d0[1:0], p1};
            d1 <= {d1[1:0], p0};
            d2 <= {d2[1:0], in_pixel};
            dilated = (|d0) | (|d1) | (|d2);
            e0 <= {e0[1:0], dilated};
            e1 <= {e1[1:0], dilated};
            e2 <= {e2[1:0], dilated};
            out_pixel <= (&e0) & (&e1) & (&e2);
            out_valid <= 1'b1;
            out_sof <= in_sof;
            out_eol <= in_eol;
            if (in_eol) col <= '0;
            else        col <= col + 1'b1;
        end else begin
            out_valid <= 1'b0;
            out_sof <= 1'b0;
            out_eol <= 1'b0;
        end
    end
endmodule
