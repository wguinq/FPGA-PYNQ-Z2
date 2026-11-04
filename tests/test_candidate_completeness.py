"""Character-supported expansion must not become a largest-box heuristic."""
import sys
import unittest
from pathlib import Path
import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'ps'))
from plate_locator_adaptive import suppress_candidate_duplicates, blue_extension_has_glyph


def candidate(x, width, score):
    q = np.float32([[x,180],[x+width,180],[x+width,250],[x,250]])
    return (score,q,(x,180,width,70),float(width*70),width/70.,1.)


class CompletenessTests(unittest.TestCase):
    def test_missing_characters_restore_larger_candidate(self):
        frame = np.full((480,640,3), 25, np.uint8)
        cv2.rectangle(frame,(100,180),(380,250),(210,60,20),-1)
        cv2.putText(frame,'AB12345',(105,232),cv2.FONT_HERSHEY_SIMPLEX,
                    1.45,(250,240,230),3,cv2.LINE_AA)
        large, small = candidate(100,280,20), candidate(180,200,37)
        self.assertTrue(blue_extension_has_glyph(frame,large,small))
        result=suppress_candidate_duplicates([small,large],bgr=frame,plate_color='blue')
        self.assertEqual(result[0][2],large[2])

    def test_plain_extension_does_not_promote_larger_candidate(self):
        frame=np.full((480,640,3),25,np.uint8)
        cv2.rectangle(frame,(100,180),(380,250),(210,60,20),-1)
        cv2.putText(frame,'12345',(185,232),cv2.FONT_HERSHEY_SIMPLEX,
                    1.45,(250,240,230),3,cv2.LINE_AA)
        large,small=candidate(100,280,20),candidate(180,200,37)
        self.assertFalse(blue_extension_has_glyph(frame,large,small))

    def test_green_branch_does_not_use_blue_expansion(self):
        large,small=candidate(100,280,20),candidate(180,200,37)
        result=suppress_candidate_duplicates([small,large],bgr=np.zeros((480,640,3),np.uint8),plate_color='green')
        self.assertEqual(result[0][2],small[2])


if __name__=='__main__':
    unittest.main()
