"""CPU blue-plate locator with saturation proposals and relative text contrast.

No dataset annotations or filename information enter this detector. The legacy
RGB mask remains available in phase3_uvc_locator for the existing RTL pipeline.
"""
import cv2
import numpy as np


def rectify(bgr, rect):
    corners = cv2.boxPoints(rect).astype(np.float32)
    edges = np.roll(corners, -1, axis=0) - corners
    start = int(np.argmax(np.sum(edges * edges, axis=1)))
    source = np.roll(corners, -start, axis=0)
    # Choose the left-to-right long edge to avoid upside-down rectification.
    if source[1, 0] < source[0, 0]:
        source = np.roll(source, -2, axis=0)
    width = 160
    height = int(np.clip(width * min(rect[1]) / max(rect[1]), 24, 80))
    target = np.float32([[0, 0], [width-1, 0], [width-1, height-1], [0, height-1]])
    transform = cv2.getPerspectiveTransform(source, target)
    return cv2.warpPerspective(bgr, transform, (width, height),
                               borderMode=cv2.BORDER_REPLICATE)


def text_features(color, dark=False):
    """Extract bright strokes relative to each crop, even if glyphs are bluish."""
    gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
    if dark:
        gray = 255 - gray
    height, width = gray.shape
    margin_y = max(2, int(height * .12))
    inner = gray[margin_y:-margin_y, 4:-4]
    low, high = np.percentile(inner, [15, 90])
    contrast = float(high - low)
    if contrast < 18:
        return 0, 0.0, 0.0, np.zeros_like(gray)
    threshold, _ = cv2.threshold(inner, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    threshold = max(threshold, low + contrast * .35)
    bright = (gray > threshold).astype(np.uint8) * 255
    bright[:margin_y] = 0
    bright[-margin_y:] = 0
    bright[:, :4] = 0
    bright[:, -4:] = 0
    count, labels, stats, centers = cv2.connectedComponentsWithStats(bright, 8)
    accepted = []
    for idx in range(1, count):
        x, y, w, h, area = stats[idx]
        if (height * .28 <= h <= height * .82 and
                1 <= w <= width * .19 and area >= height * .12):
            accepted.append(idx)
    cleaned = np.isin(labels, accepted).astype(np.uint8) * 255
    coverage = float(np.count_nonzero(cleaned)) / gray.size
    if len(accepted) < 2:
        return len(accepted), coverage, 0.0, cleaned
    # A row of glyphs must have similar vertical centers; reflections scattered
    # over a window should not count as a text line.
    ys = centers[accepted, 1]
    alignment = float(np.clip(1.0 - np.std(ys) / (height * .22), 0, 1))
    return len(accepted), coverage, alignment, cleaned


def find_candidates_adaptive(bgr, debug=False, proposal_mask=None,
                             min_edge_density=0.0, edge_score_weight=0.0,
                             enable_large_blue_fallback=True,
                             experimental_small_target=False, plate_color='blue',
                             blue_value_floor=45):
    if plate_color == 'auto':
        if proposal_mask is not None:
            raise ValueError('Auto color requires separate PS color proposals')
        joint = []
        for color in ('blue', 'green'):
            if debug:
                print('adaptive color={}'.format(color), flush=True)
            branch = find_candidates_adaptive(
                bgr, debug=debug, min_edge_density=min_edge_density,
                edge_score_weight=edge_score_weight,
                enable_large_blue_fallback=enable_large_blue_fallback,
                experimental_small_target=experimental_small_target,
                plate_color=color)
            joint.extend(tuple(candidate) + (color,) for candidate in branch)
        return suppress_candidate_duplicates(joint)
    if plate_color not in ('blue', 'green'):
        raise ValueError('plate_color must be blue, green or auto')
    green = plate_color == 'green'
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    height, width = bgr.shape[:2]
    candidates = []
    if proposal_mask is not None:
        if proposal_mask.shape[:2] != (height, width):
            raise ValueError('PL proposal mask must match the camera frame size')
        proposal_masks = [('PL', proposal_mask, cv2.RETR_EXTERNAL)]
    else:
        proposal_masks = []
        # Screen recapture can tint dark bodywork green. Higher saturation
        # proposals separate the plate from that connected background; retain
        # low thresholds for pale plates. Blue proposals remain unchanged.
        for saturation in ((35, 65, 100, 130, 160) if green else (70, 110, 150)):
            blue = cv2.inRange(hsv, np.uint8([35 if green else 95, saturation,
                                            45 if green else blue_value_floor]),
                              np.uint8([95 if green else 135, 255, 255]))
            mask = cv2.morphologyEx(blue, cv2.MORPH_CLOSE,
                                    np.ones((3, 7), np.uint8))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                                    np.ones((3, 3), np.uint8))
            proposal_masks.append(('HSV_S{}'.format(saturation), mask,
                                   cv2.RETR_EXTERNAL))

    # In PL-assisted mode the Cb150 mask and morphology IP produce the
    # proposals. PS still performs perspective rectification and character
    # structure scoring on those proposals.
    for proposal_name, mask, retrieval_mode in proposal_masks:
        # A large blue vehicle can swallow its plate in the PL close/open mask.
        # If the normal PL contours find nothing, search for the bright plate
        # border and characters within that large blue component.
        if proposal_name == 'PL_TEXT' and candidates:
            continue
        contours, _ = cv2.findContours(mask, retrieval_mode,
                                       cv2.CHAIN_APPROX_SIMPLE)
        rejection_counts = {}
        accepted_count = 0
        oversized_pl_blob = False
        for contour in contours:
            rect = cv2.minAreaRect(contour)
            short, long = sorted(rect[1])
            area = float(cv2.contourArea(contour))
            box = cv2.boxPoints(rect).astype(np.int32)
            x, y, w, h = cv2.boundingRect(box)

            def reject(reason, details=''):
                rejection_counts[reason] = rejection_counts.get(reason, 0) + 1
                if debug:
                    # Noise components below minimum dimensions are common;
                    # aggregate them instead of flooding the UART console.
                    if reason != 'minimum_or_maximum_size':
                        print('adaptive reject proposal={} box=({}, {}, {}, {}) reason={} {}'.format(
                            proposal_name, x, y, w, h, reason, details).rstrip(), flush=True)

            if short < 5 or long < 22 or short > height * .25 or long > width * .65:
                if proposal_name == 'PL' and (short > height * .25 or
                                               long > width * .65):
                    oversized_pl_blob = True
                reject('minimum_or_maximum_size', 'short={:.1f} long={:.1f} area={:.0f}'.format(
                    short, long, area))
                continue
            ratio = long / short
            fill = area / (short * long)
            if area < 70:
                reject('area', 'area={:.0f}'.format(area))
                continue
            if not 2.0 <= ratio <= 5.8:
                reject('ratio', 'ratio={:.2f}'.format(ratio))
                continue
            min_fill = .40 if proposal_name == 'PL_TEXT' else .45
            if fill < min_fill:
                reject('fill', 'fill={:.3f}'.format(fill))
                continue
            color = rectify(bgr, rect)
            n, coverage, alignment, _ = text_features(color, dark=green)
            relaxed = (experimental_small_target and n == 2 and long <= 80 and
                       short <= 25 and coverage >= .05 and alignment >= .65)
            if n < 3 and not relaxed:
                reject('text_components', 'n={} coverage={:.3f}'.format(n, coverage))
                continue
            if coverage < .035 or coverage > .5:
                reject('text_coverage', 'coverage={:.3f}'.format(coverage))
                continue
            if alignment < .45:
                reject('text_alignment', 'alignment={:.3f}'.format(alignment))
                continue
            crop_hsv = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)
            blue_fraction = float(np.mean(cv2.inRange(
                crop_hsv, np.uint8([35 if green else 95, 35 if green else 70, 45]),
                np.uint8([95 if green else 135, 255, 255])) != 0))
            if blue_fraction < .3:
                reject('blue_fraction', 'blue={:.3f}'.format(blue_fraction))
                continue
            if relaxed and blue_fraction < .5:
                reject('small_blue_fraction', 'blue={:.3f}'.format(blue_fraction))
                continue
            # Optional grayscale/Sobel feature gate. Keep the color mask as
            # the region proposal; use edges only to assess plate texture.
            gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
            gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
            edge_mag = np.abs(gx) + np.abs(gy)
            top = max(1, int(edge_mag.shape[0] * .12))
            bottom = max(top + 1, int(edge_mag.shape[0] * .88))
            inner_edges = edge_mag[top:bottom, 4:-4]
            edge_density = (float(np.mean(inner_edges >= 320.0))
                            if inner_edges.size else 0.0)
            if edge_density < min_edge_density:
                reject('edge_density', 'edge={:.3f} min={:.3f}'.format(
                    edge_density, min_edge_density))
                continue
            ratio_quality = float(np.exp(-abs(np.log(ratio / 3.2))))
            coverage_target = .30 if proposal_name == 'PL_TEXT' else .16
            text_quality = min(n, 7) / 7.0 * min(coverage / coverage_target, 1.0)
            score = 100 * ratio_quality * text_quality * alignment * np.sqrt(blue_fraction * fill)
            if edge_score_weight > 0:
                score *= 1.0 + edge_score_weight * min(edge_density, .5)
            # Experimental low-text candidates are retained as alternatives,
            # not promoted to reliable plates. Normal behavior stays unchanged.
            if score < (6 if relaxed else 15):
                reject('score', 'score={:.1f}'.format(score))
                continue
            candidates.append((float(score), box, (x,y,w,h), area, ratio, fill))
            accepted_count += 1
            if debug:
                print('adaptive accept proposal={} x={} y={} w={} h={} text={} coverage={:.3f} alignment={:.3f} blue={:.3f} edge={:.3f} score={:.1f}'.format(
                    proposal_name, x, y, w, h, n, coverage, alignment,
                    blue_fraction, edge_density, score), flush=True)
        if debug:
            summary = ','.join('{}={}'.format(key, value)
                               for key, value in sorted(rejection_counts.items()))
            print('adaptive summary proposal={} contours={} accepted={} rejects={}'.format(
                proposal_name, len(contours), accepted_count, summary or 'none'),
                flush=True)
        if (not green and proposal_name == 'PL' and not candidates and oversized_pl_blob and
                enable_large_blue_fallback):
            bright = cv2.inRange(hsv, np.uint8([0, 0, 175]),
                                 np.uint8([179, 120, 255]))
            blue_neighborhood = cv2.dilate(
                proposal_mask, np.ones((7, 7), np.uint8))
            bright = cv2.bitwise_and(bright, blue_neighborhood)
            bright = cv2.morphologyEx(
                bright, cv2.MORPH_CLOSE, np.ones((3, 7), np.uint8))
            proposal_masks.append(('PL_TEXT', bright, cv2.RETR_LIST))
    if not green and proposal_mask is None and not candidates and blue_value_floor == 45:
        # Preserve the normal low-light path. Only retry brighter blue pixels
        # when dark blue bodywork has prevented any qualified plate proposal.
        if debug:
            print('adaptive fallback=blue_brightness value_floor=110', flush=True)
        return find_candidates_adaptive(
            bgr, debug=debug, min_edge_density=min_edge_density,
            edge_score_weight=edge_score_weight,
            enable_large_blue_fallback=enable_large_blue_fallback,
            experimental_small_target=experimental_small_target,
            plate_color='blue', blue_value_floor=110)
    return suppress_candidate_duplicates(candidates, bgr=bgr,
                                         plate_color=plate_color, debug=debug)


def blue_extension_has_glyph(bgr, larger, smaller):
    """Prefer observed extra glyphs, not simply a larger colored rectangle."""
    from plate_ocr_baseline import warp_candidate, order_candidate_quad
    large = larger[1].astype(np.float32)
    small = smaller[1].astype(np.float32)
    large_area = max(1., cv2.contourArea(large))
    small_area = max(1., cv2.contourArea(small))
    if not 1.12 <= large_area / small_area <= 1.8:
        return False
    overlap, _ = cv2.intersectConvexConvex(large, small)
    if overlap / small_area < .9 or larger[0] < smaller[0] * .45:
        return False
    a = sorted(cv2.minAreaRect(large)[1])
    b = sorted(cv2.minAreaRect(small)[1])
    # Restrict recovery to substantial horizontal truncation. Slight growth
    # often just captures borders and can distort an already complete crop.
    if a[1] / max(1., b[1]) < 1.3:
        return False
    if not .85 <= a[0] / max(1., b[0]) <= 1.4:
        return False
    crop = warp_candidate(bgr, large)
    n, coverage, alignment, strokes = text_features(crop)
    inner_crop = warp_candidate(bgr, small)
    small_n = text_features(inner_crop)[0]
    if n < max(4, small_n - 1) or alignment < .55 or coverage < .06:
        return False
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    if np.mean(cv2.inRange(hsv, np.uint8([95,70,45]),
                          np.uint8([135,255,255])) != 0) < .5:
        return False
    height, width = crop.shape[:2]
    target = np.float32([[0,0],[width-1,0],[width-1,height-1],[0,height-1]])
    matrix = cv2.getPerspectiveTransform(order_candidate_quad(large), target)
    inner = cv2.perspectiveTransform(small[None], matrix)[0]
    occupied = np.zeros((height,width), np.uint8)
    cv2.fillConvexPoly(occupied, np.rint(inner).astype(np.int32), 255)
    occupied = cv2.dilate(occupied, np.ones((3,3),np.uint8))
    count, labels, stats, centers = cv2.connectedComponentsWithStats(strokes, 8)
    for i in range(1, count):
        cx, cy = centers[i]
        # Horizontal extension must add a glyph in the same text band.
        if not .25 * height <= cy <= .78 * height:
            continue
        component = labels == i
        outside = np.count_nonzero(component & (occupied == 0))
        if outside / max(1, stats[i,cv2.CC_STAT_AREA]) >= .65:
            return True
    return False


def suppress_candidate_duplicates(candidates, bgr=None, plate_color=None, debug=False):
    """Merge ranked candidates, retaining optional color metadata."""
    candidates = sorted(candidates, key=lambda c: c[0], reverse=True)
    selected = []
    for candidate in candidates:
        box = candidate[1].astype(np.float32)
        area = max(1.0, cv2.contourArea(box))
        duplicate = False
        for index, kept in enumerate(selected):
            other = kept[1].astype(np.float32)
            overlap, _ = cv2.intersectConvexConvex(box, other)
            if overlap / min(area, max(1.0, cv2.contourArea(other))) > .5:
                if (plate_color == 'blue' and bgr is not None and
                        blue_extension_has_glyph(bgr, candidate, kept)):
                    selected[index] = candidate
                    if debug:
                        print('adaptive completeness: kept={} replaced_by={} reason=extra_glyph'.format(
                            kept[2], candidate[2]), flush=True)
                duplicate = True
                break
        if not duplicate:
            selected.append(candidate)
    return selected
