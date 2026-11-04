"""Attribute baseline misses to the best overlapping contour on a saved manifest."""
import contextlib
import csv
import io
import re
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from evaluate_ccpd import letterbox, parse_annotation, transform_points, bbox_polygon
from phase3_uvc_locator import original_blue_mask, candidate_mask_from_bgr, find_candidates


def box_iou(first, second):
    a, b = np.min(first, axis=0), np.max(first, axis=0)
    c, d = np.min(second, axis=0), np.max(second, axis=0)
    inter = float(np.prod(np.maximum(0, np.minimum(b, d) - np.maximum(a, c))))
    union = float(np.prod(b-a) + np.prod(d-c)) - inter
    return inter / union if union else 0.0


def analyze(root, manifest, output):
    rows, tiles = [], []
    with Path(manifest).open(encoding='utf-8-sig') as handle:
        samples = list(csv.DictReader(handle))
    for sample in samples:
        if sample['has_plate'] != '1':
            continue
        path = Path(root) / sample['subset'] / sample['filename']
        frame, scale, left, top = letterbox(cv2.imread(str(path)))
        bbox, _ = parse_annotation(path)
        gt = transform_points(bbox_polygon(bbox), scale, left, top)
        mask = candidate_mask_from_bgr(frame)
        capture = io.StringIO()
        with contextlib.redirect_stdout(capture):
            find_candidates(mask, frame, debug=True)
        reasons = {(int(x), int(y)): reason for x, y, reason in re.findall(
            r'candidate reject x=(\d+) y=(\d+) reason=(\w+)', capture.getvalue())}
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        matches = [(box_iou(cv2.boxPoints(cv2.minAreaRect(c)), gt), i)
                   for i, c in enumerate(contours)]
        overlap, idx = max(matches, default=(0.0, 0))
        reason = 'no_matching_contour'
        if overlap >= 0.5:
            x, y, _, _ = cv2.boundingRect(contours[idx])
            reason = reasons.get((x, y), 'accepted_or_suppressed')
        x0, y0 = np.floor(gt.min(axis=0)).astype(int)
        x1, y1 = np.ceil(gt.max(axis=0)).astype(int)
        crop = frame[max(0,y0):y1+1, max(0,x0):x1+1]
        blue = original_blue_mask(crop)
        row = dict(filename=path.name, width=x1-x0, height=y1-y0,
                   best_contour_iou=overlap, reason=reason,
                   blue_fraction=float(np.mean(blue != 0)),
                   baseline_iou=float(sample['iou_bbox']))
        rows.append(row)
        if row['baseline_iou'] < 0.5 and len(tiles) < 24:
            tile = np.full((120, 300, 3), 30, np.uint8)
            tile[25:85, :150] = cv2.resize(crop, (150, 60))
            tile[25:85, 150:] = cv2.cvtColor(cv2.resize(blue, (150,60)), cv2.COLOR_GRAY2BGR)
            cv2.putText(tile, reason, (4, 17), cv2.FONT_HERSHEY_SIMPLEX, .43, (255,255,255), 1)
            cv2.putText(tile, '{}x{} blue={:.2f} match={:.2f}'.format(
                x1-x0, y1-y0, row['blue_fraction'], overlap), (4,105),
                cv2.FONT_HERSHEY_SIMPLEX, .40, (255,255,255), 1)
            tiles.append(tile)
    output = Path(output)
    with output.with_suffix('.csv').open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    if tiles:
        while len(tiles) % 4:
            tiles.append(np.zeros_like(tiles[0]))
        cv2.imwrite(str(output.with_suffix('.jpg')), np.vstack([
            np.hstack(tiles[i:i+4]) for i in range(0,len(tiles),4)]))
    print('failures:', Counter(r['reason'] for r in rows if r['baseline_iou'] < .5))
    print('all:', Counter(r['reason'] for r in rows))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root')
    parser.add_argument('manifest')
    parser.add_argument('output')
    args = parser.parse_args()
    analyze(args.root, args.manifest, args.output)
