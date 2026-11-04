# Definitional proc to organize widgets for parameters.
proc init_gui { IPINST } {
  ipgui::add_param $IPINST -name "Component_Name"
  #Adding Page
  set Page_0 [ipgui::add_page $IPINST -name "Page 0"]
  ipgui::add_param $IPINST -name "BG_MARGIN" -parent ${Page_0}
  ipgui::add_param $IPINST -name "BLUE_MIN" -parent ${Page_0}
  ipgui::add_param $IPINST -name "BR_MARGIN" -parent ${Page_0}


}

proc update_PARAM_VALUE.BG_MARGIN { PARAM_VALUE.BG_MARGIN } {
	# Procedure called to update BG_MARGIN when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.BG_MARGIN { PARAM_VALUE.BG_MARGIN } {
	# Procedure called to validate BG_MARGIN
	return true
}

proc update_PARAM_VALUE.BLUE_MIN { PARAM_VALUE.BLUE_MIN } {
	# Procedure called to update BLUE_MIN when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.BLUE_MIN { PARAM_VALUE.BLUE_MIN } {
	# Procedure called to validate BLUE_MIN
	return true
}

proc update_PARAM_VALUE.BR_MARGIN { PARAM_VALUE.BR_MARGIN } {
	# Procedure called to update BR_MARGIN when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.BR_MARGIN { PARAM_VALUE.BR_MARGIN } {
	# Procedure called to validate BR_MARGIN
	return true
}


proc update_MODELPARAM_VALUE.BLUE_MIN { MODELPARAM_VALUE.BLUE_MIN PARAM_VALUE.BLUE_MIN } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.BLUE_MIN}] ${MODELPARAM_VALUE.BLUE_MIN}
}

proc update_MODELPARAM_VALUE.BR_MARGIN { MODELPARAM_VALUE.BR_MARGIN PARAM_VALUE.BR_MARGIN } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.BR_MARGIN}] ${MODELPARAM_VALUE.BR_MARGIN}
}

proc update_MODELPARAM_VALUE.BG_MARGIN { MODELPARAM_VALUE.BG_MARGIN PARAM_VALUE.BG_MARGIN } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.BG_MARGIN}] ${MODELPARAM_VALUE.BG_MARGIN}
}

