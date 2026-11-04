`timescale 1ns/1ps
// Observe the encoder input without modifying the video path. Gray counters
// cross into the AXI reference domain; software differences two snapshots.
module hdmi_signal_probe (
    input wire pixel_clk, input wire pixel_resetn,
    input wire hsync, input wire vsync,
    input wire ref_clk, input wire ref_resetn,
    output wire [31:0] pixel_cycles,
    output wire [31:0] sync_edges
);
    reg [31:0] pixels = 0;
    reg [15:0] lines = 0, frames = 0;
    reg hs = 0, vs = 0;
    reg [31:0] pixel_gray = 0;
    reg [15:0] line_gray = 0, frame_gray = 0;
    wire [31:0] next_pixel = pixels + 1'b1;
    wire [15:0] next_line = lines + 1'b1;
    wire [15:0] next_frame = frames + 1'b1;
    always @(posedge pixel_clk) begin
        if (!pixel_resetn) begin
            pixels <= 0; lines <= 0; frames <= 0;
            pixel_gray <= 0; line_gray <= 0; frame_gray <= 0;
            hs <= 0; vs <= 0;
        end else begin
            pixels <= next_pixel;
            pixel_gray <= next_pixel ^ (next_pixel >> 1);
            hs <= hsync; vs <= vsync;
            if (hsync && !hs) begin
                lines <= next_line;
                line_gray <= next_line ^ (next_line >> 1);
            end
            if (vsync && !vs) begin
                frames <= next_frame;
                frame_gray <= next_frame ^ (next_frame >> 1);
            end
        end
    end
    (* ASYNC_REG = "TRUE" *) reg [31:0] p_meta = 0, p_sync = 0;
    (* ASYNC_REG = "TRUE" *) reg [15:0] h_meta = 0, h_sync = 0;
    (* ASYNC_REG = "TRUE" *) reg [15:0] v_meta = 0, v_sync = 0;
    always @(posedge ref_clk) begin
        if (!ref_resetn) begin
            p_meta <= 0; p_sync <= 0; h_meta <= 0; h_sync <= 0;
            v_meta <= 0; v_sync <= 0;
        end else begin
            p_meta <= pixel_gray; p_sync <= p_meta;
            h_meta <= line_gray; h_sync <= h_meta;
            v_meta <= frame_gray; v_sync <= v_meta;
        end
    end
    genvar i;
    generate for (i=0; i<32; i=i+1) begin: decode_pixel
        assign pixel_cycles[i] = ^(p_sync >> i);
    end endgenerate
    generate for (i=0; i<16; i=i+1) begin: decode_sync
        assign sync_edges[i] = ^(h_sync >> i);
        assign sync_edges[16+i] = ^(v_sync >> i);
    end endgenerate
endmodule
