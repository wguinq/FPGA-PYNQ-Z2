`timescale 1ns/1ps
module tb_axis_binary_morph_accel;
    reg clk = 0, rst_n = 0;
    always #5 clk = ~clk;
    reg [31:0] in_data = 0;
    reg [3:0] in_keep = 0;
    reg in_valid = 0, in_last = 0;
    wire in_ready;
    wire [31:0] out_data;
    wire [3:0] out_keep;
    wire out_valid, out_last;
    reg out_ready = 0;
    integer sent, received = 0, cycles = 0;
    integer x, y;
    reg input_bit, expected_bit;

    axis_binary_morph_accel #(
        .IMAGE_WIDTH(10), .IMAGE_HEIGHT(10),
        .CLOSE_W(3), .CLOSE_H(3), .OPEN_W(3), .OPEN_H(3)
    ) dut (
        .aclk(clk), .aresetn(rst_n),
        .s_axis_tdata(in_data), .s_axis_tkeep(in_keep),
        .s_axis_tvalid(in_valid), .s_axis_tready(in_ready), .s_axis_tlast(in_last),
        .m_axis_tdata(out_data), .m_axis_tkeep(out_keep),
        .m_axis_tvalid(out_valid), .m_axis_tready(out_ready), .m_axis_tlast(out_last)
    );

    always @(negedge clk) begin
        cycles = cycles + 1;
        out_ready = (cycles % 6 != 0);
    end

    always @(posedge clk) begin
        if (rst_n && out_valid && out_ready) begin
            x = received % 10;
            y = received / 10;
            expected_bit = (y >= 6) && (x >= 6);
            if (out_data !== {8'ha5, expected_bit ? 24'hffffff : 24'h000000})
                $fatal(1, "pixel (%0d,%0d): got %h expected %0d", x, y, out_data, expected_bit);
            if (out_keep !== ((received == 7) ? 4'b0111 : 4'b1111))
                $fatal(1, "TKEEP mismatch at pixel %0d", received);
            if (out_last !== (received == 99))
                $fatal(1, "TLAST mismatch at pixel %0d", received);
            received = received + 1;
        end
    end

    initial begin
        repeat (3) @(negedge clk);
        rst_n = 1;
        for (sent = 0; sent < 100; sent = sent + 1) begin
            x = sent % 10;
            y = sent / 10;
            input_bit = (y >= 2 && y <= 6 && x >= 2 && x <= 7);
            if (x == 4 && y == 4) input_bit = 0; // hole that closing must fill
            if (x == 1 && y == 8) input_bit = 1; // noise that opening must remove
            @(negedge clk);
            in_data = {8'ha5, input_bit ? 24'hffffff : 24'h000000};
            in_keep = (sent == 7) ? 4'b0111 : 4'b1111;
            in_last = (sent == 99);
            in_valid = 1;
            do @(posedge clk); while (!in_ready);
        end
        @(negedge clk);
        in_valid = 0;
        in_last = 0;
        wait (received == 100);
        $display("PASS: causal close/open morphology, hole fill, noise removal, sidebands and stalls");
        $finish;
    end

    initial begin
        #200000;
        $fatal(1, "timeout: received %0d", received);
    end
endmodule
