module plate_pipeline #(
    parameter int IMAGE_WIDTH = 640,
    parameter int IMAGE_HEIGHT = 480
) (
    input logic aclk, aresetn,
    input logic [23:0] s_axis_tdata,
    input logic s_axis_tvalid,
    output logic s_axis_tready,
    input logic s_axis_tuser,             // start of frame
    input logic s_axis_tlast,             // end of line
    input logic [1:0] color_mode,
    input logic night_mode,
    output logic m_axis_tvalid,
    input logic m_axis_tready,
    output logic m_axis_tuser,
    output logic m_axis_tlast,
    output logic [7:0] m_axis_tdata,      // 0 or 255 candidate mask
    output logic bbox_valid,
    output logic [$clog2(IMAGE_WIDTH)-1:0] bbox_x,
    output logic [$clog2(IMAGE_HEIGHT)-1:0] bbox_y,
    output logic [$clog2(IMAGE_WIDTH):0] bbox_w,
    output logic [$clog2(IMAGE_HEIGHT):0] bbox_h
);
    logic mask_valid, mask_pixel, mask_sof, mask_eol;
    logic morph_valid, morph_sof, morph_eol, morph_pixel;
    // The compute pipe is deliberately one pixel/clock.  Package an AXIS FIFO
    // after this IP when its consumer may apply backpressure; while directly
    // connected, m_axis_tready must remain asserted for an active frame.
    assign s_axis_tready = 1'b1;

    // plate_color_mask is registered, so the AXIS sideband must be registered
    // with it.  Losing this alignment corrupts frame/line coordinates even
    // though the binary image may still look plausible.
    always_ff @(posedge aclk) begin
        if (!aresetn) begin
            mask_sof <= 1'b0;
            mask_eol <= 1'b0;
        end else begin
            mask_sof <= s_axis_tvalid && s_axis_tready && s_axis_tuser;
            mask_eol <= s_axis_tvalid && s_axis_tready && s_axis_tlast;
        end
    end

    plate_color_mask u_mask (
        .clk(aclk), .rst_n(aresetn), .in_valid(s_axis_tvalid && s_axis_tready),
        .r(s_axis_tdata[23:16]), .g(s_axis_tdata[15:8]), .b(s_axis_tdata[7:0]),
        .color_mode(color_mode), .night_mode(night_mode),
        .out_valid(mask_valid), .plate_pixel(mask_pixel)
    );
    morph_close_3x3 #(.IMAGE_WIDTH(IMAGE_WIDTH)) u_morph (
        .clk(aclk), .rst_n(aresetn), .in_valid(mask_valid), .in_sof(mask_sof),
        .in_eol(mask_eol), .in_pixel(mask_pixel), .out_valid(morph_valid),
        .out_sof(morph_sof), .out_eol(morph_eol), .out_pixel(morph_pixel)
    );
    plate_bbox #(.IMAGE_WIDTH(IMAGE_WIDTH), .IMAGE_HEIGHT(IMAGE_HEIGHT)) u_bbox (
        .clk(aclk), .rst_n(aresetn), .in_valid(morph_valid), .in_sof(morph_sof),
        .in_eol(morph_eol), .in_pixel(morph_pixel), .bbox_valid(bbox_valid),
        .bbox_x(bbox_x), .bbox_y(bbox_y), .bbox_w(bbox_w), .bbox_h(bbox_h)
    );
    always_ff @(posedge aclk) begin
        if (!aresetn) begin
            m_axis_tvalid <= 1'b0; m_axis_tuser <= 1'b0; m_axis_tlast <= 1'b0; m_axis_tdata <= '0;
        end else begin
            m_axis_tvalid <= morph_valid;
            m_axis_tuser <= morph_sof;
            m_axis_tlast <= morph_eol;
            m_axis_tdata <= morph_pixel ? 8'hff : 8'h00;
        end
    end
endmodule
