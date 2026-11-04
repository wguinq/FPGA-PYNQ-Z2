`timescale 1ns/1ps
module tb_axis_cb150_mask_accel;
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
    reg [31:0] vectors [0:15];
    reg expected_blue [0:15];
    integer sent, received = 0, cycles = 0;
    reg stalled = 0;
    reg [31:0] stalled_data;
    reg [3:0] stalled_keep;
    reg stalled_last;

    axis_cb150_mask_accel dut (
        .aclk(clk), .aresetn(rst_n),
        .s_axis_tdata(in_data), .s_axis_tkeep(in_keep),
        .s_axis_tvalid(in_valid), .s_axis_tready(in_ready), .s_axis_tlast(in_last),
        .m_axis_tdata(out_data), .m_axis_tkeep(out_keep),
        .m_axis_tvalid(out_valid), .m_axis_tready(out_ready), .m_axis_tlast(out_last)
    );

    always @(negedge clk) begin
        cycles = cycles + 1;
        out_ready = (cycles % 19 < 7); // sustained AXIS backpressure
    end

    always @(posedge clk) begin
        if (rst_n && stalled &&
            (!out_valid || out_data !== stalled_data ||
             out_keep !== stalled_keep || out_last !== stalled_last))
            $fatal(1, "output changed while stalled");
        stalled = rst_n && out_valid && !out_ready;
        if (stalled) begin
            stalled_data = out_data;
            stalled_keep = out_keep;
            stalled_last = out_last;
        end
        if (rst_n && out_valid && out_ready) begin
            if (out_data !== {vectors[received][31:24],
                              expected_blue[received] ? 24'hffffff : 24'h000000})
                $fatal(1, "pixel %0d: input=%h output=%h expected_blue=%0d",
                       received, vectors[received], out_data, expected_blue[received]);
            if (out_keep !== ((received == 4) ? 4'b0111 : 4'b1111))
                $fatal(1, "TKEEP mismatch at pixel %0d", received);
            if (out_last !== (received == 15))
                $fatal(1, "TLAST mismatch at pixel %0d", received);
            received = received + 1;
        end
    end

    initial begin
        // Packed bytes are X,B,G,R. For R=G=0, B=44 gives B-Y=39;
        // B=45 gives B-Y=40 and is the exact positive boundary.
        vectors[0]  = 32'hff2c0000; expected_blue[0]  = 0;
        vectors[1]  = 32'hff2d0000; expected_blue[1]  = 1;
        vectors[2]  = 32'hff000000; expected_blue[2]  = 0;
        vectors[3]  = 32'hffff0000; expected_blue[3]  = 1;
        vectors[4]  = 32'hff806060; expected_blue[4]  = 0;
        vectors[5]  = 32'hffb02020; expected_blue[5]  = 1;
        vectors[6]  = 32'hff2080f0; expected_blue[6]  = 0;
        vectors[7]  = 32'hff208020; expected_blue[7]  = 0;
        vectors[8]  = 32'h7f96501e; expected_blue[8]  = 1;
        vectors[9]  = 32'h005a513c; expected_blue[9]  = 0;
        vectors[10] = 32'h005a503c; expected_blue[10] = 0;
        vectors[11] = 32'hff96501e; expected_blue[11] = 1;
        vectors[12] = 32'hff501e3c; expected_blue[12] = 0;
        vectors[13] = 32'hff4f1e3c; expected_blue[13] = 0;
        vectors[14] = 32'hffb4b4b4; expected_blue[14] = 0;
        vectors[15] = 32'h0064a0ff; expected_blue[15] = 0;

        repeat (3) @(negedge clk);
        rst_n = 1;
        for (sent = 0; sent < 16; sent = sent + 1) begin
            @(negedge clk);
            in_data = vectors[sent];
            in_keep = (sent == 4) ? 4'b0111 : 4'b1111;
            in_last = (sent == 15);
            in_valid = 1;
            do @(posedge clk); while (!in_ready);
        end
        @(negedge clk);
        in_valid = 0;
        in_last = 0;
        wait (received == 16);
        $display("PASS: 16 Cb>150 pixels, threshold, sidebands and stalls");
        $finish;
    end

    initial begin
        #100000;
        $fatal(1, "timeout: received %0d", received);
    end
endmodule
