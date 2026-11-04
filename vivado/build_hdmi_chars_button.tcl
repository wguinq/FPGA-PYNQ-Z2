set root [file normalize [file join [file dirname [info script]] ..]]
set_param general.maxThreads 2
set dest [file join $root build plate_ocr_hdmi_chars_button]
if {[file exists [file join $dest plate_ocr.xpr]]} {
    open_project [file join $dest plate_ocr.xpr]
} else {
    open_project [file join $root build plate_ocr_hdmi_green_pipeline plate_ocr.xpr]
    save_project_as plate_ocr $dest -force
}
set bd [get_files -filter {NAME =~ */VIP_system.bd}]
open_bd_design $bd
set_property CONFIG.C_GPIO_WIDTH {3} [get_bd_cells axi_gpio_buttons]
set_property LEFT {2} [get_bd_ports buttons]
foreach old [get_files -all -quiet *pynqz2_buttons.xdc] { remove_files $old }
set xdc [file join $root vivado pynqz2_buttons3.xdc]
if {![llength [get_files -quiet $xdc]]} { add_files -fileset constrs_1 -norecurse $xdc }
validate_bd_design
save_bd_design
config_ip_cache -disable_cache
generate_target -force all $bd
set wrapper [make_wrapper -files $bd -top -force]
foreach old [get_files -all -quiet *VIP_system_wrapper.v] {
    if {[string tolower [file normalize $old]] ne [string tolower [file normalize $wrapper]]} { remove_files $old }
}
add_files -norecurse $wrapper
set_property top VIP_system_wrapper [current_fileset]
update_compile_order -fileset sources_1
reset_run synth_1
foreach run [get_runs -quiet *_synth_1] {
    if {[get_property PROGRESS $run] ne "100%"} { reset_run $run }
}
launch_runs impl_1 -to_step write_bitstream -jobs 1
wait_on_run impl_1
if {[get_property PROGRESS [get_runs impl_1]] ne "100%"} { error "Implementation failed" }
open_run impl_1
if {[llength [get_ports -quiet {buttons[*]}]] != 3} { error "Three buttons absent" }
foreach {port pin} {buttons[0] D19 buttons[1] D20 buttons[2] L20} {
    if {[get_property PACKAGE_PIN [get_ports $port]] ne $pin} { error "Wrong pin: $port" }
}
set exports [file join $dest exports]
file mkdir $exports
report_timing_summary -file [file join $exports timing_summary.rpt]
report_drc -file [file join $exports drc.rpt]
report_utilization -file [file join $exports utilization.rpt]
if {[get_property SLACK [get_timing_paths -delay_type max -max_paths 1]] < 0} { error "Setup timing failed" }
if {[get_property SLACK [get_timing_paths -delay_type min -max_paths 1]] < 0} { error "Hold timing failed" }
file copy -force [file join $dest plate_ocr.runs impl_1 VIP_system_wrapper.bit] [file join $exports phase3_hdmi_chars.bit]
file copy -force [file join $dest plate_ocr.gen sources_1 bd VIP_system hw_handoff VIP_system.hwh] [file join $exports phase3_hdmi_chars.hwh]
puts "CHARS_BUTTON_EXPORT_COMPLETE $exports"
