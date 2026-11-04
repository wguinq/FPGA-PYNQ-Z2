#!/usr/bin/env python3
"""UVC -> blue mask/morphology overlay -> PS candidate box validation."""
import argparse
import time

import cv2
import numpy as np
try:
    from pynq import Overlay, allocate
except ImportError:  # Permit the vision functions to run in desktop dataset tests.
    Overlay = None
    allocate = None

OFFSET_X, OFFSET_Y = 0, 0
MIN_TEXT_COMPONENTS = 1
MIN_PLATE_RATIO = 1.7
DETECTION_CLOSE_W, DETECTION_CLOSE_H = 7, 3
MIN_CANDIDATE_SCORE = 120.0
MIN_RELATIVE_SCORE = 0.25
NESTED_OVERLAP_FRACTION = 0.60
MIN_PROJECTION_RUNS = 1
MIN_SOBEL_EDGE_DENSITY = 0.25
MIN_REFINED_BLUE_FRACTION = 0.45
PL_CLOSE_W, PL_CLOSE_H = 25, 9


def long_axis_angle(rect):
    """Return the min-area rectangle long-axis angle in [-90, 90] degrees."""
    points = cv2.boxPoints(rect)
    edges = np.roll(points, -1, axis=0) - points
    edge = edges[int(np.argmax(np.sum(edges * edges, axis=1)))]
    angle = float(np.degrees(np.arctan2(edge[1], edge[0])))
    if angle > 90:
        angle -= 180
    elif angle < -90:
        angle += 180
    return angle


def original_blue_mask(bgr):
    """Legacy RGB blue predicate; the current PL proposal uses Cb150."""
    b = bgr[:, :, 0].astype(np.int16)
    g = bgr[:, :, 1].astype(np.int16)
    r = bgr[:, :, 2].astype(np.int16)
    return np.where((b >= 80) & (b >= r + 20) & (b >= g + 10),
                    255, 0).astype(np.uint8)


def candidate_mask_from_bgr(bgr):
    """Build the same small-kernel PS candidate mask used by the live loop."""
    raw_blue = original_blue_mask(bgr)
    close_kernel = np.ones((DETECTION_CLOSE_H, DETECTION_CLOSE_W), np.uint8)
    mask = cv2.morphologyEx(raw_blue, cv2.MORPH_CLOSE, close_kernel)
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))


def align_phase3_mask(mask):
    """Undo the causal close-window offset in the current PL morph IP."""
    height, width = mask.shape[:2]
    shift_x, shift_y = PL_CLOSE_W - 1, PL_CLOSE_H - 1
    aligned = np.zeros_like(mask)
    if width > shift_x and height > shift_y:
        aligned[:height - shift_y, :width - shift_x] = mask[shift_y:, shift_x:]
    return aligned


def simulate_phase3_mask_from_bgr(bgr, close_w=PL_CLOSE_W, close_h=PL_CLOSE_H):
    """CPU reference for the RGB mask plus close/open kernel sizes."""
    raw_blue = original_blue_mask(bgr)
    close = cv2.morphologyEx(raw_blue, cv2.MORPH_CLOSE,
                             np.ones((close_h, close_w), np.uint8))
    return cv2.morphologyEx(close, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))


def contour_blue_purity(contour, blue_mask):
    """Fraction of a morphology contour that was genuinely blue originally."""
    shifted = contour.astype(np.int32).copy()
    shifted[:, 0, 0] -= OFFSET_X
    shifted[:, 0, 1] -= OFFSET_Y
    region = np.zeros_like(blue_mask)
    cv2.drawContours(region, [shifted], -1, 255, cv2.FILLED)
    region_pixels = cv2.countNonZero(region)
    if region_pixels == 0:
        return 0.0
    blue_pixels = cv2.countNonZero(cv2.bitwise_and(blue_mask, region))
    return float(blue_pixels) / float(region_pixels)


def plate_texture_metrics(contour, bgr, blue_mask):
    """Measure bright, character-like components adjacent to blue pixels.

    This is diagnostic only for now.  The contour is rectified to a horizontal
    crop so that tilted plates can be compared with false blue regions.
    """
    rect = cv2.minAreaRect(contour)
    corners = cv2.boxPoints(rect).astype(np.float32)
    corners[:, 0] -= OFFSET_X
    corners[:, 1] -= OFFSET_Y
    edges = np.roll(corners, -1, axis=0) - corners
    lengths = np.sqrt(np.sum(edges * edges, axis=1))
    long_edge = int(np.argmax(lengths))
    short_side = max(1, int(round(min(rect[1]))))
    long_side = max(1, int(round(max(rect[1]))))
    out_w = int(np.clip(long_side, 24, 240))
    out_h = int(np.clip(short_side, 8, 80))

    # The four boxPoints are cyclic. Start with a long edge and map its opposite
    # edge to the bottom of the rectified crop.
    i = long_edge
    src = np.array([corners[i], corners[(i + 1) % 4],
                    corners[(i + 2) % 4], corners[(i + 3) % 4]],
                   dtype=np.float32)
    dst = np.array([[0, 0], [out_w - 1, 0],
                    [out_w - 1, out_h - 1], [0, out_h - 1]],
                   dtype=np.float32)
    transform = cv2.getPerspectiveTransform(src, dst)
    color = cv2.warpPerspective(bgr, transform, (out_w, out_h),
                                flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_REPLICATE)
    blue = cv2.warpPerspective(blue_mask, transform, (out_w, out_h),
                               flags=cv2.INTER_NEAREST,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
    near_blue = cv2.dilate(blue, np.ones((5, 5), np.uint8))
    blue_gray = gray[blue != 0]
    if blue_gray.size:
        # White plate characters can be dim in a phone-screen capture. Use
        # their local contrast against the candidate's blue background instead
        # of an absolute brightness cutoff.
        threshold = max(55, int(np.percentile(blue_gray, 60)) + 22)
    else:
        threshold = 75
    bright_nonblue = ((gray >= threshold) & (blue == 0) &
                      (near_blue != 0)).astype(np.uint8) * 255
    # Join fragmented character strokes without deleting thin low-resolution
    # strokes, which the previous 2x2 opening removed.
    bright_nonblue = cv2.morphologyEx(bright_nonblue, cv2.MORPH_CLOSE,
                                      np.ones((2, 2), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(bright_nonblue, 8)
    char_components = 0
    char_pixels = 0
    # Reject thin glare bands on blue glass. Plate glyph strokes occupy a
    # substantial fraction of the rectified plate height; a 12% cutoff let
    # fragmented horizontal reflections count as several "characters".
    min_height = max(2, int(round(out_h * 0.22)))
    for label in range(1, count):
        _, _, cw, ch, pixels = stats[label]
        if ch >= min_height and cw <= max(3, int(out_w * 0.25)) and pixels >= 2:
            char_components += 1
            char_pixels += int(pixels)
    coverage = float(char_pixels) / float(max(1, out_w * out_h))
    return char_components, coverage, color, bright_nonblue


def candidate_score(area, ratio, fill, mean_sat, blue_purity):
    """Rank plate-like blobs without allowing a huge region to dominate."""
    size_term = min(np.sqrt(area), np.sqrt(1200.0))
    ratio_term = np.exp(-abs(ratio - 3.2))
    return float(size_term * ratio_term * fill * mean_sat * blue_purity)


def projection_character_runs(binary):
    """Count plausible character bands using the reference project's projection idea.

    This is diagnostic for now: low-resolution and tilted samples need to be
    measured before making the projection count a hard accept/reject rule.
    """
    height, width = binary.shape
    if height == 0 or width == 0:
        return 0
    projection = np.count_nonzero(binary, axis=0)
    threshold = max(2, int(round(height * 0.15)))
    active = (projection >= threshold).astype(np.uint8)
    # Bridge one-column gaps caused by broken glyph strokes.
    active = cv2.morphologyEx(active.reshape(1, -1), cv2.MORPH_CLOSE,
                              np.ones((1, 3), np.uint8)).reshape(-1)
    padded = np.pad(active, (1, 1))
    starts = np.flatnonzero((padded[1:] == 1) & (padded[:-1] == 0))
    ends = np.flatnonzero((padded[1:] == 0) & (padded[:-1] == 1))
    min_width = max(2, int(round(height * 0.10)))
    max_width = max(min_width, int(round(height * 1.25)))
    return sum(min_width <= end - start <= max_width
               for start, end in zip(starts, ends))


def sobel_edge_density(color):
    """Return the fraction of rectified pixels carrying a strong Sobel edge."""
    gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(gx, gy)
    return float(np.count_nonzero(magnitude >= 180.0)) / float(gray.size)


def locate_blue_plate_patch(color, blue, text_mask):
    """Find a compact blue rectangle with nearby character texture.

    Coordinates are within an already rectified morphology candidate. The
    result is diagnostic only until its crops are visually confirmed.
    """
    closed = cv2.morphologyEx(blue, cv2.MORPH_CLOSE,
                              np.ones((5, 3), np.uint8))
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    height, width = blue.shape
    options = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = float(cv2.contourArea(contour))
        if h < 5 or w < 16 or area < max(20, width * height * 0.008):
            continue
        ratio = w / float(h)
        if not 2.0 <= ratio <= 7.0:
            continue
        blue_fraction = cv2.countNonZero(blue[y:y + h, x:x + w]) / float(w * h)
        text_pixels = cv2.countNonZero(text_mask[y:y + h, x:x + w])
        if blue_fraction < 0.25:
            continue
        ratio_quality = np.exp(-abs(np.log(max(ratio, 0.01) / 4.0)))
        score = (text_pixels + 1.0) * np.sqrt(area) * ratio_quality
        options.append((score, x, y, w, h, text_pixels, blue_fraction))
    if not options:
        return None
    score, x, y, w, h, text_pixels, blue_fraction = max(options)
    pad_x = max(2, int(round(w * 0.05)))
    pad_y = max(2, int(round(h * 0.12)))
    x0, y0 = max(0, x - pad_x), max(0, y - pad_y)
    x1, y1 = min(width, x + w + pad_x), min(height, y + h + pad_y)
    crop = color[y0:y1, x0:x1].copy()
    marked = color.copy()
    cv2.rectangle(marked, (x0, y0), (x1 - 1, y1 - 1), (0, 255, 0), 1)
    return crop, marked, (x, y, w, h, text_pixels, blue_fraction, score)


def contour_diagnostics(mask, bgr, frame_index):
    """Return elongated contour measurements for threshold debugging."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    blue_mask = original_blue_mask(bgr)
    rows = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        x, y, w, h = cv2.boundingRect(contour)
        rect = cv2.minAreaRect(contour)
        rw, rh = rect[1]
        short_side, long_side = min(rw, rh), max(rw, rh)
        if area < 5 or short_side <= 0 or long_side < 10:
            continue
        ratio = long_side / short_side
        angle = long_axis_angle(rect)
        fill = area / (w * h)
        if ratio < 1.5:
            continue
        x0, y0 = max(0, x - OFFSET_X), max(0, y - OFFSET_Y)
        x1 = min(bgr.shape[1], x0 + w)
        y1 = min(bgr.shape[0], y0 + h)
        mean_sat = float(cv2.mean(
            cv2.cvtColor(bgr[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)[:, :, 1]
        )[0]) if x1 > x0 and y1 > y0 else 0.0
        blue_purity = contour_blue_purity(contour, blue_mask)
        reasons = []
        if short_side < 4:
            reasons.append('short')
        if long_side < 20:
            reasons.append('long')
        if short_side > mask.shape[0] * 0.20:
            reasons.append('too_thick')
        if long_side > mask.shape[1] * 0.50:
            reasons.append('too_long')
        if area < 60:
            reasons.append('area')
        if not MIN_PLATE_RATIO <= ratio <= 6.0:
            reasons.append('ratio')
        if fill < 0.30:
            reasons.append('fill')
        if mean_sat < 35:
            reasons.append('sat')
        if blue_purity < 0.12:
            reasons.append('blue')
        if abs(angle) > 50:
            reasons.append('tilt')
        texture_count, texture_coverage, texture_crop, texture_mask = plate_texture_metrics(
            contour, bgr, blue_mask)
        patch = locate_blue_plate_patch(texture_crop,
                                       original_blue_mask(texture_crop),
                                       texture_mask)
        if texture_count < MIN_TEXT_COMPONENTS:
            reasons.append('text')
        if (area >= 60 and short_side >= 4 and long_side >= 20
                and MIN_PLATE_RATIO <= ratio <= 6.0 and abs(angle) <= 50):
            tag = '{:04d}_{}_{}'.format(frame_index, x, y)
            cv2.imwrite('/home/xilinx/phase3_texture_{}.png'.format(tag), texture_crop)
            cv2.imwrite('/home/xilinx/phase3_texture_mask_{}.png'.format(tag), texture_mask)
            if patch is not None:
                patch_crop, patch_marked, patch_info = patch
                cv2.imwrite('/home/xilinx/phase3_refine_{}.png'.format(tag), patch_marked)
                cv2.imwrite('/home/xilinx/phase3_plate_roi_{}.png'.format(tag), patch_crop)
                px, py, pw, ph, text_pixels, blue_fraction, patch_score = patch_info
                print('refine {}: local_box=({}, {}, {}, {}) text_pixels={} blue_fraction={:.3f} score={:.1f}'.format(
                    tag, px, py, pw, ph, text_pixels, blue_fraction, patch_score),
                    flush=True)
        score = candidate_score(area, ratio, fill, mean_sat, blue_purity)
        score *= min(texture_count / float(MIN_TEXT_COMPONENTS), 2.0)
        rows.append((area, x, y, w, h, short_side, long_side,
                     ratio, fill, mean_sat, blue_purity, angle,
                     texture_count, texture_coverage, score,
                     ','.join(reasons) or 'PASS'))
    return sorted(rows, reverse=True)[:20]


def find_candidates(mask, bgr, debug=False):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    blue_mask = original_blue_mask(bgr)
    candidates = []

    def reject(reason, x, y):
        if debug:
            print('candidate reject x={} y={} reason={}'.format(x, y, reason),
                  flush=True)

    for contour in contours:
        area = float(cv2.contourArea(contour))
        x, y, w, h = cv2.boundingRect(contour)
        rect = cv2.minAreaRect(contour)
        rw, rh = rect[1]
        short_side, long_side = min(rw, rh), max(rw, rh)
        # Keep small, distant plates.  Geometry and fill checks below still
        # reject most short line/noise contours.
        if short_side < 4 or long_side < 20:
            reject('minimum_size', x, y)
            continue
        # A plate may be rotated, so constrain min-area-rectangle side lengths
        # rather than axis-aligned width/height.  This rejects large blue bags,
        # screens and illumination regions while retaining a close plate.
        if short_side > mask.shape[0] * 0.20 or long_side > mask.shape[1] * 0.50:
            reject('maximum_size', x, y)
            continue
        ratio = long_side / short_side
        fill = area / (w * h)
        if area < 60 or not MIN_PLATE_RATIO <= ratio <= 6.0 or fill < 0.30:
            reason = 'area' if area < 60 else ('ratio' if not MIN_PLATE_RATIO <= ratio <= 6.0 else 'fill')
            reject(reason, x, y)
            continue
        if abs(long_axis_angle(rect)) > 50:
            reject('tilt', x, y)
            continue
        box = cv2.boxPoints(rect).astype(np.int32)
        box[:, 0] -= OFFSET_X
        box[:, 1] -= OFFSET_Y
        x0, y0 = max(0, x - OFFSET_X), max(0, y - OFFSET_Y)
        x1 = min(bgr.shape[1], x0 + w)
        y1 = min(bgr.shape[0], y0 + h)
        mean_sat = float(cv2.mean(hsv[y0:y1, x0:x1, 1])[0])
        if mean_sat < 35:
            reject('sat', x, y)
            continue
        blue_purity = contour_blue_purity(contour, blue_mask)
        if blue_purity < 0.12:
            reject('blue', x, y)
            continue
        texture_count, texture_coverage, texture_crop, texture_mask = plate_texture_metrics(
            contour, bgr, blue_mask)
        projection_runs = projection_character_runs(texture_mask)
        edge_density = sobel_edge_density(texture_crop)
        if debug:
            print('candidate features x={} y={} text_n={} projection_runs={} sobel_edge={:.3f}'.format(
                x, y, texture_count, projection_runs, edge_density), flush=True)
        if (projection_runs < MIN_PROJECTION_RUNS or
                edge_density < MIN_SOBEL_EDGE_DENSITY):
            reject('character_structure', x, y)
            continue
        if texture_count < MIN_TEXT_COMPONENTS:
            reject('text', x, y)
            continue
        refined = locate_blue_plate_patch(
            texture_crop, original_blue_mask(texture_crop), texture_mask)
        if refined is None:
            reject('blue_patch', x, y)
            continue
        refined_info = refined[2]
        _, _, _, _, refined_text_pixels, refined_blue_fraction, _ = refined_info
        if (refined_text_pixels < 40 or
                refined_blue_fraction < MIN_REFINED_BLUE_FRACTION):
            reject('blue_patch', x, y)
            continue
        score = candidate_score(area, ratio, fill, mean_sat, blue_purity)
        score *= min(texture_count / float(MIN_TEXT_COMPONENTS), 2.0)
        if score < MIN_CANDIDATE_SCORE:
            reject('low_score', x, y)
            continue
        candidates.append((score, box, (x - OFFSET_X, y - OFFSET_Y, w, h),
                           area, ratio, fill))
        if debug:
            print('candidate accept x={} y={} ratio={:.2f} text_n={} score={:.1f}'.format(
                x, y, ratio, texture_count, score), flush=True)
    candidates.sort(key=lambda item: item[0], reverse=True)
    if not candidates:
        return candidates

    # Keep plate-like regions that are competitive with the best candidate;
    # reject weak isolated noise and larger contours that merely contain the
    # already-selected plate (for example, the whole grille/bumper).
    best_score = candidates[0][0]
    selected = []
    for candidate in candidates:
        score, box = candidate[0], candidate[1]
        if score < best_score * MIN_RELATIVE_SCORE:
            if debug:
                print('candidate suppress x={} y={} reason=low_relative_score'.format(
                    candidate[2][0], candidate[2][1]), flush=True)
            continue
        area = max(1.0, abs(cv2.contourArea(box.astype(np.float32))))
        nested = False
        for kept in selected:
            kept_box = kept[1].astype(np.float32)
            kept_area = max(1.0, abs(cv2.contourArea(kept_box)))
            intersection, _ = cv2.intersectConvexConvex(
                box.astype(np.float32), kept_box)
            if intersection / min(area, kept_area) >= NESTED_OVERLAP_FRACTION:
                nested = True
                break
        if nested:
            if debug:
                print('candidate suppress x={} y={} reason=overlaps_better_candidate'.format(
                    candidate[2][0], candidate[2][1]), flush=True)
            continue
        selected.append(candidate)
    return selected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('bitfile')
    parser.add_argument('--camera', type=int, default=0)
    parser.add_argument('--width', type=int, default=640)
    parser.add_argument('--height', type=int, default=480)
    parser.add_argument('--frames', type=int, default=300)
    parser.add_argument('--detector', choices=('baseline', 'adaptive', 'adaptive_pl'), default='baseline',
                        help='baseline: 原定位器；adaptive: CPU HSV proposals；adaptive_pl: PL proposals + PS character filter')
    args = parser.parse_args()
    if Overlay is None or allocate is None:
        raise SystemExit('实时采集需要 PYNQ 环境；桌面环境请使用 evaluate_ccpd.py')
    if (args.width, args.height) != (640, 480):
        raise SystemExit('当前硬件仅按 640x480 验证')

    overlay = Overlay(args.bitfile)
    dma = overlay.axi_dma_0
    cap = cv2.VideoCapture(args.camera, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    if not cap.isOpened():
        raise RuntimeError('无法打开 USB UVC 摄像头')

    in_buf = allocate(shape=(args.height, args.width), dtype=np.uint32)
    out_buf = allocate(shape=(args.height, args.width), dtype=np.uint32)
    run_started = time.perf_counter()
    try:
        for index in range(args.frames):
            ok, bgr = cap.read()
            if not ok or bgr.shape[:2] != (args.height, args.width):
                raise RuntimeError('摄像头未返回固定尺寸画面')
            rgba = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGBA)
            in_buf[:] = rgba.view(np.uint32).reshape(args.height, args.width)
            dma.recvchannel.transfer(out_buf)
            dma.sendchannel.transfer(in_buf)
            dma.sendchannel.wait()
            dma.recvchannel.wait()
            raw = out_buf.view(np.uint8).reshape(args.height, args.width, 4)
            pl_mask = np.where(raw[:, :, 0] != 0, 255, 0).astype(np.uint8)
            aligned_pl_mask = align_phase3_mask(pl_mask)
            debug_frame = index % 30 == 0
            # Keep legacy candidate-mask generation on every frame only for
            # the baseline detector. The other detectors do not consume it;
            # generate it only when saving a diagnostic snapshot.
            if args.detector == 'baseline':
                candidate_mask = candidate_mask_from_bgr(bgr)
            elif debug_frame:
                candidate_mask = candidate_mask_from_bgr(bgr)
            else:
                candidate_mask = None
            if args.detector == 'adaptive':
                from plate_locator_adaptive import find_candidates_adaptive
                candidates = find_candidates_adaptive(bgr, debug=(index % 30 == 0))
            elif args.detector == 'adaptive_pl':
                from plate_locator_adaptive import find_candidates_adaptive
                candidates = find_candidates_adaptive(
                    bgr, debug=(index % 30 == 0), proposal_mask=aligned_pl_mask)
            else:
                candidates = find_candidates(candidate_mask, bgr,
                                             debug=(index % 30 == 0))

            if debug_frame:
                raw_blue = original_blue_mask(bgr)
                elapsed = max(time.perf_counter() - run_started, 1e-6)
                print('frame {} timing: average_fps={:.2f} pl_mask_fill={:.4f} candidates={}'.format(
                    index, (index + 1) / elapsed,
                    cv2.countNonZero(pl_mask) / float(args.width * args.height),
                    len(candidates)), flush=True)
                marked = bgr.copy()
                for rank, (_, box, _, _, _, _) in enumerate(candidates[:3], 1):
                    box[:, 0] = np.clip(box[:, 0], 0, args.width - 1)
                    box[:, 1] = np.clip(box[:, 1], 0, args.height - 1)
                    cv2.polylines(marked, [box], True, (0, 255, 0), 2)
                    cv2.putText(marked, str(rank), tuple(box[0]),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.imwrite('/home/xilinx/phase3_mask_{:04d}.png'.format(index), pl_mask)
                cv2.imwrite('/home/xilinx/phase3_blue_raw_{:04d}.png'.format(index), raw_blue)
                cv2.imwrite('/home/xilinx/phase3_candidate_mask_{:04d}.png'.format(index),
                            candidate_mask)
                cv2.imwrite('/home/xilinx/phase3_boxes_{:04d}.png'.format(index), marked)
                if candidates:
                    _, _, (x, y, w, h), area, ratio, fill = candidates[0]
                    x0, y0 = max(0, x), max(0, y)
                    x1, y1 = min(args.width, x + w), min(args.height, y + h)
                    cv2.imwrite('/home/xilinx/phase3_roi_{:04d}.png'.format(index),
                                bgr[y0:y1, x0:x1])
                    print('frame {}: candidate x={} y={} w={} h={} area={:.0f} ratio={:.2f} fill={:.3f}'.format(
                        index, x, y, w, h, area, ratio, fill), flush=True)
                else:
                    print('frame {}: no candidate'.format(index), flush=True)
                if args.detector in ('adaptive', 'adaptive_pl'):
                    print('detector={} ({} proposals); saved phase3_mask is the raw PL output'.format(
                        args.detector, 'CPU HSV' if args.detector == 'adaptive' else 'PL Cb150/morph'),
                          flush=True)
                    continue
                print('diagnostic: area x y w h short long ratio fill sat blue angle text_n text_cov score result', flush=True)
                for row in contour_diagnostics(candidate_mask, bgr, index):
                    (area, x, y, w, h, short_side, long_side, ratio, fill,
                     mean_sat, blue_purity, angle, texture_count,
                     texture_coverage, score, result) = row
                    print('{:.0f} {} {} {} {} {:.1f} {:.1f} {:.2f} {:.3f} {:.1f} {:.3f} {:.1f} {} {:.3f} {:.1f} {}'.format(
                        area, x, y, w, h, short_side, long_side, ratio, fill,
                        mean_sat, blue_purity, angle, texture_count,
                        texture_coverage, score, result),
                        flush=True)
    finally:
        cap.release()
        in_buf.freebuffer()
        out_buf.freebuffer()


if __name__ == '__main__':
    main()
