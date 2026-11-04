`timescale 1ns/1ps
// Phase-3A experimental blue-candidate mask.
// Input/output convention: [7:0]=R, [15:8]=G, [23:16]=B, [31:24]=X.
// A blue candidate satisfies all three inexpensive RGB-domain comparisons:
// B >= BLUE_MIN, B-R >= BR_MARGIN, and B-G >= BG_MARGIN.
// Output RGB is 255 for a candidate and 0 otherwise; X is preserved.
module axis_blue_mask_accel #(
    parameter integer BLUE_MIN  = 80,
    parameter integer BR_MARGIN = 20,
    parameter integer BG_MARGIN = 10
) (
    (* X_INTERFACE_INFO = "xilinx.com:signal:clock:1.0 aclk CLK" *)
    (* X_INTERFACE_PARAMETER = "ASSOCIATED_BUSIF S_AXIS:M_AXIS, ASSOCIATED_RESET aresetn" *)
    input  wire        aclk,
    (* X_INTERFACE_INFO = "xilinx.com:signal:reset:1.0 aresetn RST" *)
    (* X_INTERFACE_PARAMETER = "POLARITY ACTIVE_LOW" *)
    input  wire        aresetn,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 S_AXIS TDATA" *)  input  wire [31:0] s_axis_tdata,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 S_AXIS TKEEP" *)  input  wire [3:0]  s_axis_tkeep,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 S_AXIS TVALID" *) input  wire        s_axis_tvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 S_AXIS TREADY" *) output wire        s_axis_tready,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 S_AXIS TLAST" *)
    (* X_INTERFACE_PARAMETER = "XIL_INTERFACENAME S_AXIS, TDATA_NUM_BYTES 4, HAS_TKEEP 1, HAS_TLAST 1" *)
    input  wire        s_axis_tlast,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 M_AXIS TDATA" *)  output reg [31:0] m_axis_tdata,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 M_AXIS TKEEP" *)  output reg [3:0]  m_axis_tkeep,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 M_AXIS TVALID" *) output reg        m_axis_tvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 M_AXIS TREADY" *) input  wire        m_axis_tready,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 M_AXIS TLAST" *)
    (* X_INTERFACE_PARAMETER = "XIL_INTERFACENAME M_AXIS, TDATA_NUM_BYTES 4, HAS_TKEEP 1, HAS_TLAST 1" *)
    output reg         m_axis_tlast
);
    wire [7:0] r = s_axis_tdata[7:0];
    wire [7:0] g = s_axis_tdata[15:8];
    wire [7:0] b = s_axis_tdata[23:16];
    wire blue = (b >= BLUE_MIN) &&
                ({1'b0, b} >= ({1'b0, r} + BR_MARGIN)) &&
                ({1'b0, b} >= ({1'b0, g} + BG_MARGIN));
    wire [7:0] mask = blue ? 8'hff : 8'h00;

    // One-entry elastic register preserves the beat during downstream stalls.
    assign s_axis_tready = !m_axis_tvalid || m_axis_tready;
    always @(posedge aclk) begin
        if (!aresetn) begin
            m_axis_tdata  <= 32'd0;
            m_axis_tkeep  <= 4'd0;
            m_axis_tvalid <= 1'b0;
            m_axis_tlast  <= 1'b0;
        end else if (s_axis_tready) begin
            m_axis_tvalid <= s_axis_tvalid;
            if (s_axis_tvalid) begin
                m_axis_tdata <= {s_axis_tdata[31:24], mask, mask, mask};
                m_axis_tkeep <= s_axis_tkeep;
                m_axis_tlast <= s_axis_tlast;
            end
        end
    end
endmodule
