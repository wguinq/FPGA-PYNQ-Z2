"""Experimental Cb + Sobel + projection plate proposals inspired by the
License-Plate-Recognition-FPGA project. This is a software prototype, not RTL.
"""
import cv2
import numpy as np

from plate_locator_adaptive import text_features


def _runs(projection, threshold, max_gap=2):
    active = np.flatnonzero(projection >= threshold)
    if active.size == 0:
        return []
    groups = []
    start = previous = int(active[0])
    for value in active[1:]:
        value = int(value)
        if value - previous > max_gap + 1:
            groups.append((start, previous))
            start = value
        previous = value
    groups.append((start, previous))
    return groups


def find_candidates_projection(bgr, cb_threshold=150, debug=False):
    """Return plate proposals from Cb threshold, erosion, Sobel and projections.

    The thresholds intentionally begin close to the reference project's RTL
    values. Projection runs are paired geometrically; chroma and text metrics
    rank the pairs but do not require a full seven-character OCR result.
    """
    height, width = bgr.shape[:2]
    ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    cb = ycrcb[:, :, 2]
    blue = (cb > int(cb_threshold)).astype(np.uint8) * 255

    # Reference project: 3x3 binary erosion, Sobel threshold, 3x3 dilation.
    clean = cv2.erode(blue, np.ones((3, 3), np.uint8), iterations=1)
    gx = cv2.Sobel(clean, cv2.CV_16S, 1, 0, ksize=3)
    gy = cv2.Sobel(clean, cv2.CV_16S, 0, 1, ksize=3)
    magnitude = cv2.magnitude(gx.astype(np.float32), gy.astype(np.float32))
    edges = (magnitude >= 128).astype(np.uint8) * 255
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

    horizontal = np.count_nonzero(edges, axis=1).astype(np.float32)
    vertical = np.count_nonzero(edges, axis=0).astype(np.float32)
    horizontal = np.convolve(horizontal, np.ones(3, np.float32) / 3, mode='same')
    vertical = np.convolve(vertical, np.ones(3, np.float32) / 3, mode='same')
    row_threshold = max(8.0, width * 0.025)
    col_threshold = max(6.0, height * 0.025)
    row_runs = _runs(horizontal, row_threshold)
    col_runs = _runs(vertical, col_threshold)

    # Keep the strongest projection bands to bound pair generation time.
    row_runs = sorted(row_runs,
                      key=lambda run: float(np.max(horizontal[run[0]:run[1] + 1])),
                      reverse=True)[:32]
    col_runs = sorted(col_runs,
                      key=lambda run: float(np.max(vertical[run[0]:run[1] + 1])),
                      reverse=True)[:32]

    row_pairs = []
    for i, first in enumerate(row_runs):
        y1 = (first[0] + first[1]) // 2
        for second in row_runs[i + 1:]:
            y2 = (second[0] + second[1]) // 2
            if y2 < y1:
                y1a, y2a = y2, y1
                top_run, bottom_run = second, first
            else:
                y1a, y2a = y1, y2
                top_run, bottom_run = first, second
            h = y2a - y1a
            if 8 <= h <= min(120, int(height * .25)):
                row_pairs.append((y1a, y2a, top_run, bottom_run))

    col_pairs = []
    for i, first in enumerate(col_runs):
        x1 = (first[0] + first[1]) // 2
        for second in col_runs[i + 1:]:
            x2 = (second[0] + second[1]) // 2
            if x2 < x1:
                x1a, x2a = x2, x1
                left_run, right_run = second, first
            else:
                x1a, x2a = x1, x2
                left_run, right_run = first, second
            w = x2a - x1a
            if 20 <= w <= int(width * .65):
                col_pairs.append((x1a, x2a, left_run, right_run))

    candidates = []
    seen = set()
    for y1, y2, top_run, bottom_run in row_pairs:
        h = y2 - y1
        for x1, x2, left_run, right_run in col_pairs:
            w = x2 - x1
            ratio = w / float(max(h, 1))
            if not 1.7 <= ratio <= 7.5:
                continue
            # Expand from Sobel edge centers to include the plate border.
            xa, ya = max(0, x1 - 3), max(0, y1 - 3)
            xb, yb = min(width, x2 + 4), min(height, y2 + 4)
            w_box, h_box = xb - xa, yb - ya
            if w_box < 24 or h_box < 9 or h_box > height * .25:
                continue
            key = (xa // 3, ya // 3, xb // 3, yb // 3)
            if key in seen:
                continue
            seen.add(key)
            blue_fraction = float(np.mean(cb[ya:yb, xa:xb] > cb_threshold))
            if blue_fraction < .12:
                continue

            row_strength = min(1.0, (float(np.max(horizontal[top_run[0]:top_run[1] + 1])) +
                                     float(np.max(horizontal[bottom_run[0]:bottom_run[1] + 1]))) /
                               max(2.0 * width * .12, 1.0))
            col_strength = min(1.0, (float(np.max(vertical[left_run[0]:left_run[1] + 1])) +
                                     float(np.max(vertical[right_run[0]:right_run[1] + 1]))) /
                               max(2.0 * height * .12, 1.0))
            edge_support = float(np.sqrt(row_strength * col_strength))
            ratio_quality = float(np.exp(-abs(np.log(max(ratio, .01) / 3.2))))

            color_crop = bgr[ya:yb, xa:xb]
            rect = ((xa + xb) * .5, (ya + yb) * .5, (w_box, h_box), 0.0)
            normalized = cv2.resize(color_crop, (160, 48), interpolation=cv2.INTER_LINEAR)
            text_count, text_coverage, alignment, _ = text_features(normalized)
            text_score = min(text_count, 7) / 7.0 * min(text_coverage / .16, 1.0)
            score = 100.0 * (0.48 * blue_fraction + 0.27 * edge_support +
                             0.15 * ratio_quality + 0.10 * text_score)
            box = np.asarray([[xa, ya], [xb - 1, ya], [xb - 1, yb - 1],
                              [xa, yb - 1]], dtype=np.int32)
            candidates.append((score, box, (xa, ya, w_box, h_box),
                               float(w_box * h_box), ratio, blue_fraction))

    candidates.sort(key=lambda item: item[0], reverse=True)
    selected = []
    for candidate in candidates:
        x, y, w, h = candidate[2]
        duplicate = False
        for kept in selected:
            kx, ky, kw, kh = kept[2]
            ix = max(0, min(x + w, kx + kw) - max(x, kx))
            iy = max(0, min(y + h, ky + kh) - max(y, ky))
            inter = ix * iy
            if inter / float(min(w * h, kw * kh)) > .5:
                duplicate = True
                break
        if not duplicate:
            selected.append(candidate)
    if debug:
        print('projection cb_threshold={} row_bands={} col_bands={} row_pairs={} col_pairs={} candidates={}'.format(
            cb_threshold, len(row_runs), len(col_runs), len(row_pairs),
            len(col_pairs), len(selected)), flush=True)
        for score, _, (x, y, w, h), _, ratio, blue_fraction in selected[:5]:
            print('projection candidate x={} y={} w={} h={} ratio={:.2f} blue={:.3f} score={:.1f}'.format(
                x, y, w, h, ratio, blue_fraction, score), flush=True)
    return selected
