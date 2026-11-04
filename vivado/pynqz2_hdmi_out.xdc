# PYNQ-Z2 HDMI OUT. Pin map follows the board's TMDS transmitter connector.
set_property -dict {PACKAGE_PIN L17 IOSTANDARD TMDS_33} [get_ports {TMDS_0_clk_n}]
set_property -dict {PACKAGE_PIN L16 IOSTANDARD TMDS_33} [get_ports {TMDS_0_clk_p}]
set_property -dict {PACKAGE_PIN K18 IOSTANDARD TMDS_33} [get_ports {TMDS_0_data_n[0]}]
set_property -dict {PACKAGE_PIN K17 IOSTANDARD TMDS_33} [get_ports {TMDS_0_data_p[0]}]
set_property -dict {PACKAGE_PIN J19 IOSTANDARD TMDS_33} [get_ports {TMDS_0_data_n[1]}]
set_property -dict {PACKAGE_PIN K19 IOSTANDARD TMDS_33} [get_ports {TMDS_0_data_p[1]}]
set_property -dict {PACKAGE_PIN H18 IOSTANDARD TMDS_33} [get_ports {TMDS_0_data_n[2]}]
set_property -dict {PACKAGE_PIN J18 IOSTANDARD TMDS_33} [get_ports {TMDS_0_data_p[2]}]

# HDMI status indicators, LED0..2 (active high on PYNQ-Z2).
set_property -dict {PACKAGE_PIN R14 IOSTANDARD LVCMOS33} [get_ports {hdmi_clk_locked_dbg}]
set_property -dict {PACKAGE_PIN P14 IOSTANDARD LVCMOS33} [get_ports {hdmi_reset_released_dbg[0]}]
set_property -dict {PACKAGE_PIN N16 IOSTANDARD LVCMOS33} [get_ports {hdmi_video_locked_dbg}]
set_property -dict {PACKAGE_PIN M20 IOSTANDARD LVCMOS33} [get_ports {hdmi_diag_select}]
