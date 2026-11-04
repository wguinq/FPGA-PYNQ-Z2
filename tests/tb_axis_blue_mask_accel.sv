`timescale 1ns/1ps
module tb_axis_blue_mask_accel;
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
    reg [31:0] vectors [0:11];
    reg expected_blue [0:11];
    integer sent, received = 0, cycles = 0;

    axis_blue_mask_accel dut (
        .aclk(clk), .aresetn(rst_n),
        .s_axis_tdata(in_data), .s_axis_tkeep(in_keep),
        .s_axis_tvalid(in_valid), .s_axis_tready(in_ready), .s_axis_tlast(in_last),
        .m_axis_tdata(out_data), .m_axis_tkeep(out_keep),
        .m_axis_tvalid(out_valid), .m_axis_tready(out_ready), .m_axis_tlast(out_last)
    );

    always @(negedge clk) begin
        cycles = cycles + 1;
        out_ready = (cycles % 5 != 0); // exercise AXIS backpressure
    end

    always @(posedge clk) begin
        if (rst_n && out_valid && out_ready) begin
            if (out_data !== {vectors[received][31:24],
                              expected_blue[received] ? 24'hffffff : 24'h000000})
                $fatal(1, "pixel %0d: input=%h output=%h blue=%0d",
                       received, vectors[received], out_data, expected_blue[received]);
            if (out_keep !== ((received == 4) ? 4'b0111 : 4'b1111))
                $fatal(1, "TKEEP mismatch at pixel %0d", received);
            if (out_last !== (received == 11))
                $fatal(1, "TLAST mismatch at pixel %0d", received);
            received = received + 1;
        end
    end

    initial begin
        // Values are X,B,G,R. Include exact thresholds and near misses.
        vectors[0]  = 32'hff96501e; expected_blue[0]  = 1; // clear blue
        vectors[1]  = 32'hff4f0000; expected_blue[1]  = 0; // B below 80
        vectors[2]  = 32'hff501e3c; expected_blue[2]  = 1; // exact B minimum / BR margin
        vectors[3]  = 32'hff501e3d; expected_blue[3]  = 0; // BR margin is only 19
        vectors[4]  = 32'hff50501e; expected_blue[4]  = 0; // BG margin is zero
        vectors[5]  = 32'hff50503c; expected_blue[5]  = 0; // gray-blue, both margins fail
        vectors[6]  = 32'hff80501e; expected_blue[6]  = 1;
        vectors[7]  = 32'hff2080f0; expected_blue[7]  = 0; // red
        vectors[8]  = 32'hff208020; expected_blue[8]  = 0; // green
        vectors[9]  = 32'h7f5a503c; expected_blue[9]  = 1; // preserve non-FF X
        vectors[10] = 32'h005a513c; expected_blue[10] = 0; // BG margin is 9
        vectors[11] = 32'h005a503c; expected_blue[11] = 1; // exact BG margin

        repeat (3) @(negedge clk);
        rst_n = 1;
        for (sent = 0; sent < 12; sent = sent + 1) begin
            @(negedge clk);
            in_data = vectors[sent];
            in_keep = (sent == 4) ? 4'b0111 : 4'b1111;
            in_last = (sent == 11);
            in_valid = 1;
            do @(posedge clk); while (!in_ready);
        end
        @(negedge clk);
        in_valid = 0;
        in_last = 0;
        wait (received == 12);
        $display("PASS: 12 blue-mask pixels, threshold boundaries, sidebands and stalls");
        $finish;
    end

    initial begin
        #100000;
        $fatal(1, "timeout: received %0d", received);
    end
endmodule
