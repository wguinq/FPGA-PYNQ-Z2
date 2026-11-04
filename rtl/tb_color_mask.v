`timescale 1ns/1ps
module tb_color_mask;
reg clk=0; always #5 clk=~clk;
reg resetn=0;
reg [31:0] data=0;
reg [1:0] mode=0;
reg valid=0,last=0,ready=0;
reg advance=1;
reg stalled=0;
reg [31:0] held_data;
reg held_last;
wire sr,ov,ol;
wire [31:0] outdata;
wire [3:0] keep;
reg [32:0] vectors[0:999];
integer sent=0,received=0,cycles=0;
axis_color_mask_accel dut(.aclk(clk),.aresetn(resetn),.plate_mode(mode),
 .s_axis_tdata(data),.s_axis_tkeep(4'hf),.s_axis_tvalid(valid),.s_axis_tready(sr),.s_axis_tlast(last),
 .m_axis_tdata(outdata),.m_axis_tkeep(keep),.m_axis_tvalid(ov),.m_axis_tready(ready),.m_axis_tlast(ol));
initial begin
 $readmemh("vectors.mem",vectors);
 repeat(5) @(negedge clk);
 resetn=1;
end
always @(negedge clk) if(resetn) begin
 ready=$urandom_range(0,1);
 if(advance) begin
  valid=sent<1000 && $urandom_range(0,1);
  if(sent<1000) begin
   data=vectors[sent][31:0];mode=sent%3;last=(sent%37==36);
  end
 end
end
always @(posedge clk) if(resetn) begin
 if(stalled && (!ov || outdata!==held_data || ol!==held_last || keep!==4'hf))
  $fatal(1,"Output changed under backpressure");
 stalled<=ov&&!ready;held_data<=outdata;held_last<=ol;
 advance <= !valid || sr;
 cycles=cycles+1;
 if(valid && sr) sent=sent+1;
 if(ov && ready) begin
  if(outdata !== {vectors[received][31:24],{24{vectors[received][32]}}} ||
     keep!==4'hf || ol!==(received%37==36)) $fatal(1,"Mismatch at %d got=%h expected=%h",received,outdata,{vectors[received][31:24],{24{vectors[received][32]}}});
  received=received+1;
  if(received==1000) begin $display("COLOR_STREAM_PASS 1000 pixels with backpressure");$finish;end
 end
 if(cycles>20000) $fatal(1,"Timeout");
end
endmodule
