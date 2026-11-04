`timescale 1ns/1ps
module tb_hdmi_signal_probe;
    reg pixel_clk=0, ref_clk=0, resetn=0;
    always #20 pixel_clk=~pixel_clk;
    always #5 ref_clk=~ref_clk;
    wire [31:0] pixels, edges;
    reg [9:0] h=0, v=0;
    integer expected_pixels=0, expected_lines=0, expected_frames=0;
    reg previous_h=0, previous_v=0;
    wire hs=!(h>=656 && h<752);
    wire vs=!(v>=490 && v<492);
    hdmi_signal_probe dut(pixel_clk,resetn,hs,vs,ref_clk,resetn,pixels,edges);
    always @(posedge pixel_clk) begin
        if (resetn) begin
            expected_pixels=expected_pixels+1;
            if(hs && !previous_h) expected_lines=expected_lines+1;
            if(vs && !previous_v) expected_frames=expected_frames+1;
            previous_h=hs;previous_v=vs;
            if(h==799) begin h<=0;v<=v==524?0:v+1'b1;end
            else h<=h+1'b1;
        end
    end
    initial begin
        #101;resetn=1;
        repeat(900000) @(negedge pixel_clk);
        // Sample far from the next pixel edge after CDC has settled.
        #16;
        if(pixels!==expected_pixels || edges[15:0]!==expected_lines || edges[31:16]!==expected_frames)
            $fatal(1,"Counter mismatch p=%d/%d h=%d/%d v=%d/%d",pixels,expected_pixels,edges[15:0],expected_lines,edges[31:16],expected_frames);
        $display("PASS HDMI Gray CDC counters: pixels=%d lines=%d frames=%d",pixels,edges[15:0],edges[31:16]);
        $finish;
    end
endmodule
