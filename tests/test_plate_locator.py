"""Image-level regressions for the offline plate locator and annotation geometry."""
import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'ps'))
from evaluate_ccpd import parse_annotation, polygon_iou, letterbox, transform_points, summarize_rows
from plate_locator_adaptive import find_candidates_adaptive


class PlateLocatorTests(unittest.TestCase):
    def test_invalid_images_are_not_counted_as_successful_rejections(self):
        # One false alarm and one unreadable negative: rate must be 1/1, not 1/2.
        rows = [dict(subset='np', has_plate=0, status='false_positive',
                     candidate_count=1, iou_bbox=0, iou_quad=0),
                dict(subset='np', has_plate=0, status='read_error',
                     candidate_count=0, iou_bbox=0, iou_quad=0)]
        summary = summarize_rows(rows)[0]
        self.assertEqual(summary['false_positive_rate'], 1.0)
        self.assertEqual(summary['evaluable_negatives'], 1)
        self.assertEqual(summary['read_errors'], 1)

    def test_uniform_blue_is_not_text(self):
        frame = np.zeros((480, 640, 3), np.uint8)
        frame[200:245, 220:380] = (200, 40, 20)
        self.assertEqual(find_candidates_adaptive(frame), [])

    def test_tinted_glyphs_survive_brightness_and_rotation(self):
        frame = np.full((480, 640, 3), 30, np.uint8)
        cv2.rectangle(frame, (220, 200), (400, 255), (200, 50, 20), -1)
        # Even the light strokes satisfy the old B-R / B-G blue predicate.
        cv2.putText(frame, 'AB12345', (225, 242), cv2.FONT_HERSHEY_SIMPLEX,
                    .95, (250, 220, 210), 2, cv2.LINE_AA)
        box = np.float32([[220,200],[400,200],[400,255],[220,255]])
        for angle, brightness in ((0, 1.0), (20, .65), (-20, 1.0)):
            with self.subTest(angle=angle, brightness=brightness):
                matrix = cv2.getRotationMatrix2D((310,227), angle, 1)
                rotated = cv2.warpAffine(frame, matrix, (640,480))
                rotated = np.uint8(rotated.astype(float) * brightness)
                expected = cv2.transform(box[None], matrix)[0]
                candidates = find_candidates_adaptive(rotated)
                self.assertTrue(candidates)
                self.assertGreater(polygon_iou(candidates[0][1], expected), .6)

    def test_annotation_and_letterbox_geometry(self):
        parsed = parse_annotation(Path('01-0_0-30&40_150&80-150&80_30&80_30&40_150&40-0_0-1-1.jpg'))
        self.assertEqual(parsed[0], [(30,40), (150,80)])
        self.assertIsNone(parse_annotation(Path('negative.jpg')))
        canvas, scale, left, top = letterbox(np.zeros((1160,720,3), np.uint8))
        self.assertEqual(canvas.shape, (480,640,3))
        points = transform_points(parsed[1], scale, left, top)
        self.assertAlmostEqual(polygon_iou(points, points), 1.0, places=5)
        self.assertEqual(polygon_iou(points, points + 500), 0.0)


if __name__ == '__main__':
    unittest.main()
