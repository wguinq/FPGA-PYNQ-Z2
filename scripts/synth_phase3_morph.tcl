read_verilog -sv [file normalize "../../rtl/phase3/axis_binary_morph_accel.sv"]
synth_design -mode out_of_context -top axis_binary_morph_accel -part xc7z020clg400-1
create_clock -period 10.000 [get_ports aclk]
report_utilization -file utilization_phase3_morph.rpt
report_timing_summary -file timing_phase3_morph.rpt
exit
