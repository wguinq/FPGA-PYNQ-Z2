`timescale 1ns/1ps
module tb_axis_sobel_accel;
    reg clk = 0, rst_n = 0;
    always #5 clk = ~clk;
    reg [31:0] in_data = 0;
    reg [3:0] in_keep = 4'hf;
    reg in_valid = 0, in_last = 0;
    wire in_ready;
    wire [31:0] out_data;
    wire [3:0] out_keep;
    wire out_valid, out_last;
    reg out_ready = 0;
    integer sent, received = 0, cycles = 0, expected;

    axis_sobel_accel #(.IMAGE_WIDTH(5), .IMAGE_HEIGHT(5)) dut (
        .aclk(clk), .aresetn(rst_n),
        .s_axis_tdata(in_data), .s_axis_tkeep(in_keep), .s_axis_tvalid(in_valid),
        .s_axis_tready(in_ready), .s_axis_tlast(in_last),
        .m_axis_tdata(out_data), .m_axis_tkeep(out_keep), .m_axis_tvalid(out_valid),
        .m_axis_tready(out_ready), .m_axis_tlast(out_last)
    );

    always @(negedge clk) begin
        cycles = cycles + 1;
        out_ready = (cycles % 4 != 0); // force downstream stalls
    end
    always @(posedge clk) begin
        if (rst_n && out_valid && out_ready) begin
            expected = ((received / 5 >= 2) && (received % 5 >= 3)) ? 255 : 0;
            if (out_data !== {8'hff, expected[7:0], expected[7:0], expected[7:0]})
                $fatal(1, "pixel %0d got %h expected gray %0d", received, out_data, expected);
            if (out_keep !== 4'hf || out_last !== (received == 24))
                $fatal(1, "sideband mismatch at pixel %0d", received);
            received = received + 1;
        end
    end
    initial begin
        repeat (3) @(negedge clk);
        rst_n = 1;
        for (sent = 0; sent < 25; sent = sent + 1) begin
            @(negedge clk);
            in_valid = 1;
            in_last = (sent == 24);
            in_data = ((sent % 5) >= 3) ? 32'hffffffff : 32'hff000000;
            do @(posedge clk); while (!in_ready);
        end
        @(negedge clk); in_valid = 0; in_last = 0;
        wait (received == 25);
        $display("PASS: 25 pixels, Sobel vertical edge, frame TLAST and output stalls");
        $finish;
    end
    initial begin
        #100000;
        $fatal(1, "timeout: received %0d", received);
    end
endmodule
