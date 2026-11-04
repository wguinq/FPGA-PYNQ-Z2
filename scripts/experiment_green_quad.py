#!/usr/bin/env python3
"""Offline green-plate corner experiment; does not modify the live detector.

Rank local quadrilaterals using geometry/color/texture only, before running
OCR. A relaxed color seed is diagnostic only and is not a recovered detection.
The explicit margin sweep is an ablation, never selected by the supplied truth.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'ps'))
from plate_ctc_infer import PlateCTCReader
from plate_locator_adaptive import find_candidates_adaptive, text_features
from plate_ocr_baseline import warp_candidate


TARGET = np.float32([[0, 0], [223, 0], [223, 63], [0, 63]])


def order_quad(points):
    points = np.asarray(points, np.float32).reshape(4, 2)
    center = points.mean(axis=0)
    points = points[np.argsort(np.arctan2(points[:, 1] - center[1],
                                        points[:, 0] - center[0]))]
    # Start with the left endpoint of the upper long edge. Keeping actual
    # vertices avoids converting a trapezoid back into a min-area rectangle.
    edges = np.roll(points, -1, axis=0) - points
    pairs = [(i, (points[i, 1] + points[(i+1) % 4, 1]) / 2)
             for i in range(4) if abs(edges[i, 0]) > abs(edges[i, 1])]
    if not pairs:
        return None
    start = min(pairs, key=lambda value: value[1])[0]
    points = np.roll(points, -start, axis=0)
    if points[1, 0] < points[0, 0]:
        points = points[[1, 0, 3, 2]]
    return points


def warp_quad(image, points, vertical_margin=0):
    points = points.copy().astype(np.float32)
    # Pixel margins are an explicit diagnostic ablation, not a selected fix.
    points[:2, 1] -= vertical_margin
    points[2:, 1] += vertical_margin
    return cv2.warpPerspective(image, cv2.getPerspectiveTransform(points, TARGET),
                               (224, 64), borderMode=cv2.BORDER_REPLICATE)


def color_seed(image):
    """Relax only the proposal ratio, without declaring this a detection."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.uint8([35, 100, 45]), np.uint8([95, 255, 255]))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 7), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    possible = []
    height, width = image.shape[:2]
    for contour in contours:
        rect = cv2.minAreaRect(contour)
        short, long = sorted(rect[1])
        area = cv2.contourArea(contour)
        if short < 8 or long < 40 or short > height*.25 or long > width*.65:
            continue
        ratio = long / short
        fill = area / (short*long)
        if not 2 <= ratio <= 8 or fill < .35 or area < 150:
            continue
        score = area * fill * np.exp(-abs(np.log(ratio / 4.5)))
        possible.append((float(score), cv2.boxPoints(rect).astype(np.float32)))
    return max(possible, key=lambda item: item[0])[1] if possible else None


def score_quad(image, quad, seed):
    area = cv2.contourArea(quad)
    seed_area = max(1., cv2.contourArea(seed))
    area_ratio = area / seed_area
    if not .65 <= area_ratio <= 1.8:
        return None
    overlap = cv2.intersectConvexConvex(quad, seed)[0] / seed_area
    if overlap < .65:
        return None
    width = (np.linalg.norm(quad[1]-quad[0]) + np.linalg.norm(quad[2]-quad[3])) / 2
    height = (np.linalg.norm(quad[3]-quad[0]) + np.linalg.norm(quad[2]-quad[1])) / 2
    if not 2 <= width / max(1., height) <= 8:
        return None
    rect = cv2.minAreaRect(quad)
    rectangularity = area / max(1., rect[1][0] * rect[1][1])
    if rectangularity < .7:
        return None
    crop = warp_quad(image, quad)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    green_fraction = float(np.mean(cv2.inRange(hsv, np.uint8([35, 35, 45]),
                                              np.uint8([95, 255, 255])) != 0))
    count, coverage, alignment, _ = text_features(crop, dark=True)
    if green_fraction < .3 or count < 3 or coverage < .03 or alignment < .3:
        return None
    # This ranking deliberately cannot access the OCR model or truth string.
    score = (green_fraction * min(overlap, 1.) * rectangularity *
             np.exp(-abs(np.log(area_ratio))))
    return {'score': float(score), 'area_ratio': float(area_ratio),
            'seed_overlap': float(overlap), 'rectangularity': float(rectangularity),
            'green_fraction': green_fraction, 'text_n': count,
            'text_coverage': coverage, 'text_alignment': alignment}


def quadrilaterals(image, seed):
    x, y, width, height = cv2.boundingRect(seed.astype(np.float32))
    left = max(0, int(x-width*.15))
    top = max(0, int(y-height*.6))
    right = min(image.shape[1], int(x+width*1.15))
    bottom = min(image.shape[0], int(y+height*1.6))
    roi = image[top:bottom, left:right]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    sources = []
    # The edge path is reported even when no closed quadrilateral exists.
    edges = cv2.Canny(cv2.GaussianBlur(gray, (3, 3), 0), 40, 120)
    sources.append(('canny', edges))
    kh = max(3, int(round(height*.1)) | 1)
    kw = max(3, int(round(width*.07)) | 1)
    for saturation in (10, 15, 20, 35):
        for value in (80, 100, 120):
            mask = cv2.inRange(hsv, np.uint8([30, saturation, value]),
                               np.uint8([95, 255, 255]))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                                    np.ones((kh, kw), np.uint8))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                                    np.ones((3, 3), np.uint8))
            sources.append(('HSV_S{}_V{}'.format(saturation, value), mask))
    candidates = []
    seen = set()
    for name, mask in sources:
        contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if len(contour) < 4:
                continue
            # Reject ROI-cut components: they do not supply observed borders.
            points = contour.reshape(-1, 2)
            if (points[:, 0].min() <= 1 or points[:, 1].min() <= 1 or
                    points[:, 0].max() >= roi.shape[1]-2 or
                    points[:, 1].max() >= roi.shape[0]-2):
                continue
            hull = cv2.convexHull(contour)
            approx = cv2.approxPolyDP(hull, .025*cv2.arcLength(hull, True), True)
            if len(approx) != 4 or not cv2.isContourConvex(approx):
                continue
            quad = order_quad(approx.reshape(4, 2) + [left, top])
            if quad is None:
                continue
            key = tuple(quad.astype(int).ravel())
            if key in seen:
                continue
            seen.add(key)
            quality = score_quad(image, quad, seed)
            if quality is not None:
                candidates.append(dict(source=name, quad=quad.tolist(), **quality))
    return sorted(candidates, key=lambda item: item['score'], reverse=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=ROOT/'build/green_ocr_static_review/debug/matched')
    parser.add_argument('--output', type=Path, default=ROOT/'build/green_ocr_static_review/quad_experiment')
    parser.add_argument('--model', type=Path, default=ROOT/'build/green_pl_bundle/best.onnx')
    parser.add_argument('--truth', default='皖ADB8617', help='Evaluation only; never used in geometry ranking')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    cv2.setNumThreads(1)
    reader = PlateCTCReader(args.model)
    records = []
    for path in sorted(args.input.glob('*_camera.png')):
        image = cv2.imread(str(path))
        accepted = find_candidates_adaptive(image, plate_color='green')
        seed = accepted[0][1].astype(np.float32) if accepted else color_seed(image)
        record = {'frame': int(path.name.split('_')[1]),
                  'live_candidate_count': len(accepted),
                  'seed_kind': 'accepted' if accepted else 'relaxed_color_diagnostic_only',
                  'source_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
        stem = path.name.replace('_camera.png', '')
        if seed is None:
            record['status'] = 'no_seed'
            records.append(record)
            continue
        record['seed'] = seed.tolist()
        baseline = warp_candidate(image, seed)
        cv2.imwrite(str(args.output/(stem+'_baseline.png')), baseline)
        record['baseline_ocr'] = reader.recognize_details(baseline)
        options = quadrilaterals(image, seed)
        record['geometry_candidates'] = options
        if options:
            # Freeze geometry choice before any OCR or truth comparison.
            chosen = options[0]
            quad = np.float32(chosen['quad'])
            record['selected'] = chosen.copy()
            record['margin_ablation'] = []
            for margin in (0, 2, 4):
                crop = warp_quad(image, quad, margin)
                detail = reader.recognize_details(crop)
                record['margin_ablation'].append({'vertical_pixels': margin,
                                                  'ocr': detail,
                                                  'exact': detail['text'] == args.truth})
                cv2.imwrite(str(args.output/(stem+'_quad_margin{}.png'.format(margin))), crop)
            boxes = image.copy()
            cv2.polylines(boxes, [seed.astype(np.int32)], True, (0, 0, 255), 1)
            cv2.polylines(boxes, [quad.astype(np.int32)], True, (0, 255, 0), 1)
            cv2.imwrite(str(args.output/(stem+'_boxes.png')), boxes)
        record['baseline_exact'] = record['baseline_ocr']['text'] == args.truth
        records.append(record)
    report = {'opencv': cv2.__version__, 'truth_for_evaluation_only': args.truth,
              'model_sha256': hashlib.sha256(args.model.read_bytes()).hexdigest(),
              'warning': 'Five same-plate diagnostic frames, not an accuracy benchmark. '
                         'Relaxed color seeds are not accepted live detections. '
                         'Margin sweep must not be selected by expected text.',
              'records': records}
    (args.output/'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                          encoding='utf-8')
    for record in records:
        print(json.dumps({'frame': record['frame'], 'seed': record['seed_kind'],
                          'baseline': record.get('baseline_ocr', {}).get('text'),
                          'geometry': record.get('selected', {}).get('source'),
                          'margins': [(r['vertical_pixels'], r['ocr']['text'])
                                      for r in record.get('margin_ablation', [])]}, ensure_ascii=True))


if __name__ == '__main__':
    main()
