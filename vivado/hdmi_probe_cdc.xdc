# Only the first stage of each explicitly Gray-coded counter synchronizer.
# Bound bit skew/flight time to less than one 25.2 MHz source period. The
# second-stage and decoded counters retain their normal 100 MHz timing.
set gray_src [get_cells -hier -regexp {.*hdmi_signal_probe_0/inst/(pixel_gray|line_gray|frame_gray)_reg\[[0-9]+\]}]
set gray_dst [get_cells -hier -regexp {.*hdmi_signal_probe_0/inst/(p_meta|h_meta|v_meta)_reg\[[0-9]+\]}]
if {![llength $gray_src] || ![llength $gray_dst]} { error "Missing HDMI probe Gray synchronizers" }
set_max_delay -datapath_only 30.0 -from $gray_src -to $gray_dst
set_bus_skew 10.0 -from $gray_src -to $gray_dst
