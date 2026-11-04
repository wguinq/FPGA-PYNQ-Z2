"""Software color-mask references for the live plate proposal pipeline.

These are algorithm comparisons only; OpenCV's HSV/YCbCr conversion is not a
bit-exact model of a future fixed-point PL implementation.
"""
import cv2
import numpy as np

from phase3_uvc_locator import original_blue_mask


def raw_color_mask(bgr, color='rgb', cb_threshold=150):
    """Create binary RGB, HSV-blue, or YCbCr-Cb-blue mask."""
    if color == 'rgb':
        return original_blue_mask(bgr)
    if color == 'hsv':
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        return cv2.inRange(hsv, np.uint8([95, 70, 45]),
                          np.uint8([135, 255, 255]))
    if color == 'cb':
        ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
        return np.where(ycrcb[:, :, 2] > int(cb_threshold),
                        255, 0).astype(np.uint8)
    raise ValueError('color must be one of: rgb, hsv, cb')


def color_proposal_mask(bgr, color='rgb', cb_threshold=150,
                        close_w=25, close_h=9):
    """Apply a shared close/open stage after the selected color predicate."""
    mask = raw_color_mask(bgr, color=color, cb_threshold=cb_threshold)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                            np.ones((int(close_h), int(close_w)), np.uint8))
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                            np.ones((3, 3), np.uint8))
