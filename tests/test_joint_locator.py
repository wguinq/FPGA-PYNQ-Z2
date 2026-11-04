import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'ps'))
import plate_locator_adaptive as locator
from plate_ocr_baseline import warp_candidate, order_candidate_quad


class JointLocatorTests(unittest.TestCase):
    def test_mixed_scene_keeps_blue_and_green(self):
        frame = np.full((480, 640, 3), 30, np.uint8)
        cv2.rectangle(frame, (60, 100), (260, 160), (200, 50, 20), -1)
        cv2.putText(frame, 'AB12345', (66, 145), cv2.FONT_HERSHEY_SIMPLEX,
                    .95, (250, 220, 210), 2, cv2.LINE_AA)
        cv2.rectangle(frame, (330, 280), (570, 345), (100, 220, 130), -1)
        cv2.putText(frame, 'AD123456', (334, 328), cv2.FONT_HERSHEY_SIMPLEX,
                    1.2, (15, 15, 15), 2, cv2.LINE_AA)
        candidates = locator.find_candidates_adaptive(frame, plate_color='auto')
        self.assertEqual({c[6] for c in candidates}, {'blue', 'green'})
        self.assertEqual(len(candidates), 2)

    def test_plain_color_regions_are_not_plates(self):
        for color in ((200, 50, 20), (100, 220, 130)):
            frame = np.full((480, 640, 3), 30, np.uint8)
            frame[200:260, 200:430] = color
            self.assertEqual(locator.find_candidates_adaptive(frame, plate_color='auto'), [])

    def test_duplicate_colors_suppressed_but_separate_plates_remain(self):
        quad = np.float32([[10,10],[210,10],[210,60],[10,60]])
        def candidate(score, q, color):
            return (score, q, (10,10,200,50), 10000., 4., 1., color)
        values = [candidate(40, quad, 'green'), candidate(30, quad + 2, 'blue'),
                  candidate(20, quad + [0, 150], 'blue')]
        kept = locator.suppress_candidate_duplicates(values)
        self.assertEqual([c[6] for c in kept], ['green', 'blue'])

    def test_true_four_corners_are_rectified_without_rectangle_fitting(self):
        # Project a known planar pattern, then recover it from unordered corners.
        pattern = np.full((64,224,3), 20, np.uint8)
        cv2.rectangle(pattern, (8,8), (50,55), (0,0,255), -1)
        cv2.rectangle(pattern, (170,8), (215,55), (0,255,0), -1)
        target = np.float32([[0,0],[223,0],[223,63],[0,63]])
        quad = np.float32([[100,100],[380,130],[345,220],[120,200]])
        scene = cv2.warpPerspective(pattern, cv2.getPerspectiveTransform(target,quad), (480,320))
        recovered = warp_candidate(scene, quad[[2,0,3,1]])
        self.assertLess(float(np.abs(recovered[12:50,12:210].astype(float) -
                                     pattern[12:50,12:210]).mean()), 3.)
        np.testing.assert_allclose(order_candidate_quad(quad[[2,0,3,1]]), quad)

    def test_invalid_quad_rejected(self):
        with self.assertRaises(ValueError):
            order_candidate_quad(np.zeros((4,2), np.float32))
        with self.assertRaises(ValueError):
            order_candidate_quad(np.full((4,2), np.nan, np.float32))


if __name__ == '__main__':
    unittest.main()
