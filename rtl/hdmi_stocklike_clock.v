`timescale 1ns/1ps
// Fixed 640x480 DVI clocking patterned after the PYNQ base HDMI clock path.
// One 126 MHz MMCM output feeds BUFIO for the OSERDES 5x clock and a BUFR /5
// for the pixel clock. Both serializer clocks therefore share the same source.
module hdmi_stocklike_clock (
    (* X_INTERFACE_INFO = "xilinx.com:signal:clock:1.0 ref_clk_100m CLK",
       X_INTERFACE_PARAMETER = "FREQ_HZ 100000000" *)
    input  wire ref_clk_100m,
    input  wire reset,
    (* X_INTERFACE_INFO = "xilinx.com:signal:clock:1.0 pixel_clk CLK",
       X_INTERFACE_PARAMETER = "FREQ_HZ 25200000" *)
    output wire pixel_clk,
    (* X_INTERFACE_INFO = "xilinx.com:signal:clock:1.0 serial_clk CLK",
       X_INTERFACE_PARAMETER = "FREQ_HZ 126000000" *)
    output wire serial_clk,
    output wire locked
);
    wire mmcm_clk_5x;
    wire mmcm_feedback;

    MMCME2_ADV #(
        .BANDWIDTH("OPTIMIZED"),
        .COMPENSATION("ZHOLD"),
        .DIVCLK_DIVIDE(5),
        .CLKFBOUT_MULT_F(31.5),
        .CLKOUT0_DIVIDE_F(5.0),
        .CLKIN1_PERIOD(10.0),
        .REF_JITTER1(0.010)
    ) mmcm (
        .CLKIN1(ref_clk_100m),
        .CLKIN2(1'b0),
        .CLKINSEL(1'b1),
        // PYNQ axi_dynclk uses direct feedback for its BUFIO/BUFR output.
        .CLKFBIN(mmcm_feedback),
        .CLKFBOUT(mmcm_feedback),
        .CLKOUT0(mmcm_clk_5x),
        .LOCKED(locked),
        .PWRDWN(1'b0),
        .RST(reset),
        .DADDR(7'b0),
        .DCLK(1'b0),
        .DEN(1'b0),
        .DI(16'b0),
        .DWE(1'b0),
        .PSCLK(1'b0),
        .PSEN(1'b0),
        .PSINCDEC(1'b0)
    );

    BUFIO serial_clock (.I(mmcm_clk_5x), .O(serial_clk));
    BUFR #(.BUFR_DIVIDE("5"), .SIM_DEVICE("7SERIES")) pixel_clock (
        .I(mmcm_clk_5x),
        .CE(1'b1),
        .CLR(~locked),
        .O(pixel_clk)
    );
endmodule
