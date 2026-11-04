# Definitional proc to organize widgets for parameters.
proc init_gui { IPINST } {
  ipgui::add_param $IPINST -name "Component_Name"
  #Adding Page
  set Page_0 [ipgui::add_page $IPINST -name "Page 0"]
  ipgui::add_param $IPINST -name "CLOSE_H" -parent ${Page_0}
  ipgui::add_param $IPINST -name "CLOSE_W" -parent ${Page_0}
  ipgui::add_param $IPINST -name "IMAGE_HEIGHT" -parent ${Page_0}
  ipgui::add_param $IPINST -name "IMAGE_WIDTH" -parent ${Page_0}
  ipgui::add_param $IPINST -name "OPEN_H" -parent ${Page_0}
  ipgui::add_param $IPINST -name "OPEN_W" -parent ${Page_0}


}

proc update_PARAM_VALUE.CLOSE_H { PARAM_VALUE.CLOSE_H } {
	# Procedure called to update CLOSE_H when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.CLOSE_H { PARAM_VALUE.CLOSE_H } {
	# Procedure called to validate CLOSE_H
	return true
}

proc update_PARAM_VALUE.CLOSE_W { PARAM_VALUE.CLOSE_W } {
	# Procedure called to update CLOSE_W when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.CLOSE_W { PARAM_VALUE.CLOSE_W } {
	# Procedure called to validate CLOSE_W
	return true
}

proc update_PARAM_VALUE.IMAGE_HEIGHT { PARAM_VALUE.IMAGE_HEIGHT } {
	# Procedure called to update IMAGE_HEIGHT when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.IMAGE_HEIGHT { PARAM_VALUE.IMAGE_HEIGHT } {
	# Procedure called to validate IMAGE_HEIGHT
	return true
}

proc update_PARAM_VALUE.IMAGE_WIDTH { PARAM_VALUE.IMAGE_WIDTH } {
	# Procedure called to update IMAGE_WIDTH when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.IMAGE_WIDTH { PARAM_VALUE.IMAGE_WIDTH } {
	# Procedure called to validate IMAGE_WIDTH
	return true
}

proc update_PARAM_VALUE.OPEN_H { PARAM_VALUE.OPEN_H } {
	# Procedure called to update OPEN_H when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.OPEN_H { PARAM_VALUE.OPEN_H } {
	# Procedure called to validate OPEN_H
	return true
}

proc update_PARAM_VALUE.OPEN_W { PARAM_VALUE.OPEN_W } {
	# Procedure called to update OPEN_W when any of the dependent parameters in the arguments change
}

proc validate_PARAM_VALUE.OPEN_W { PARAM_VALUE.OPEN_W } {
	# Procedure called to validate OPEN_W
	return true
}


proc update_MODELPARAM_VALUE.IMAGE_WIDTH { MODELPARAM_VALUE.IMAGE_WIDTH PARAM_VALUE.IMAGE_WIDTH } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.IMAGE_WIDTH}] ${MODELPARAM_VALUE.IMAGE_WIDTH}
}

proc update_MODELPARAM_VALUE.IMAGE_HEIGHT { MODELPARAM_VALUE.IMAGE_HEIGHT PARAM_VALUE.IMAGE_HEIGHT } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.IMAGE_HEIGHT}] ${MODELPARAM_VALUE.IMAGE_HEIGHT}
}

proc update_MODELPARAM_VALUE.CLOSE_W { MODELPARAM_VALUE.CLOSE_W PARAM_VALUE.CLOSE_W } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.CLOSE_W}] ${MODELPARAM_VALUE.CLOSE_W}
}

proc update_MODELPARAM_VALUE.CLOSE_H { MODELPARAM_VALUE.CLOSE_H PARAM_VALUE.CLOSE_H } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.CLOSE_H}] ${MODELPARAM_VALUE.CLOSE_H}
}

proc update_MODELPARAM_VALUE.OPEN_W { MODELPARAM_VALUE.OPEN_W PARAM_VALUE.OPEN_W } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.OPEN_W}] ${MODELPARAM_VALUE.OPEN_W}
}

proc update_MODELPARAM_VALUE.OPEN_H { MODELPARAM_VALUE.OPEN_H PARAM_VALUE.OPEN_H } {
	# Procedure called to set VHDL generic/Verilog parameter value(s) based on TCL parameter value
	set_property value [get_property value ${PARAM_VALUE.OPEN_H}] ${MODELPARAM_VALUE.OPEN_H}
}

