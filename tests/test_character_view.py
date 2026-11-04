import sys
import unittest
from unittest.mock import Mock, patch
from queue import Queue
from pathlib import Path
import cv2
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'ps'))
from board_controls import button_actions, BoardControls
from phase3_hdmi_overlay import character_binary, render_character_preview, packet_predictions


class CharacterViewTests(unittest.TestCase):
    def test_bouncing_button2_press_generates_one_toggle(self):
        controls=BoardControls.__new__(BoardControls)
        controls.button_mask=7
        controls.events=Queue()
        controls.gpio=Mock()
        controls.gpio.read.side_effect=[0,4,0,4,4,4]
        controls.stop=Mock()
        controls.stop.wait.side_effect=[False]*5+[True]
        with patch('board_controls.time.monotonic',side_effect=[0,.01,.02,.03,.08,.12]):
            controls.poll()
        self.assertEqual(controls.events.get_nowait(),'binary')
        self.assertTrue(controls.events.empty())

    def test_button2_is_independent_of_existing_actions(self):
        self.assertEqual(button_actions(1),('capture',))
        self.assertEqual(button_actions(2),('toggle',))
        self.assertEqual(button_actions(4),('binary',))
        self.assertEqual(button_actions(0),())
        self.assertEqual(button_actions(7),('capture','toggle','binary'))

    def test_blue_and_green_glyphs_are_white_without_modifying_input(self):
        for color,background,glyph in [('blue',(180,30,10),(250,250,250)),
                                        ('green',(110,220,140),(10,10,10))]:
            crop=np.full((64,224,3),background,np.uint8)
            crop[12:52,30:45]=glyph
            before=crop.copy()
            mask=character_binary(crop,color)
            np.testing.assert_array_equal(crop,before)
            self.assertEqual(mask.dtype,np.uint8)
            self.assertEqual(mask[24,35],255)
            self.assertEqual(mask[24,100],0)
            self.assertEqual(set(np.unique(mask)),{0,255})

    def test_preview_uses_matching_packet_and_keeps_predictions_serializable(self):
        predictions=[{'rank':1,'color':'blue','text':'test','ocr_seconds':.1}]
        mask=np.zeros((64,224),np.uint8);mask[:,30:45]=255
        item=('ok',125,1.,.2,[],'test',.1,predictions,{'binary':mask,'color':'blue'})
        image=render_character_preview(item)
        self.assertEqual(image.shape,(480,640,3))
        self.assertEqual(packet_predictions(item),predictions)
        self.assertEqual(render_character_preview(None).shape,(480,640,3))
        self.assertEqual(render_character_preview(item[:8]).shape,(480,640,3))


if __name__=='__main__':unittest.main()
