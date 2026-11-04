#!/usr/bin/env python3
"""Export real CCPD train images cropped by the current adaptive detector.

Only detector proposals with quadrilateral IoU >= the requested threshold are
saved, so a wrong plate proposal does not contaminate the OCR training labels.
"""
import argparse
import csv
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from plate_locator_adaptive import find_candidates_adaptive
from plate_ocr_baseline import (letterbox, parse_ccpd, quad_iou,
                                warp_candidate)


def main():
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ccpd-root', type=Path,
                        default=Path('F:/data/CCPD-Dataset/data/CCPD2019'))
    parser.add_argument('--data', type=Path,
                        default=Path('build/ocr_real_crops'))
    parser.add_argument('--max-images', type=int, default=8000)
    parser.add_argument('--min-iou', type=float, default=.7)
    parser.add_argument('--seed', type=int, default=20261030)
    args = parser.parse_args()
    with (args.data / 'manifest.csv').open(encoding='utf-8-sig',
                                           newline='') as handle:
        rows = [row for row in csv.DictReader(handle)
                if row['dataset'] == 'ccpd2019_base' and
                row['split'] == 'train']
    if args.max_images > 0 and len(rows) > args.max_images:
        rows = random.Random(args.seed).sample(rows, args.max_images)
    rows.sort(key=lambda row: row['source_image'])
    destination = args.data / 'detector7' / 'train'
    destination.mkdir(parents=True, exist_ok=True)
    counts = Counter()
    accepted = []
    for index, row in enumerate(rows, 1):
        source = row['source_image']
        parsed = parse_ccpd(source)
        if parsed is None:
            counts['bad_annotation'] += 1
            continue
        corners, _ = parsed
        image = cv2.imread(str(args.ccpd_root.joinpath(*source.split('/'))),
                           cv2.IMREAD_COLOR)
        if image is None:
            counts['read_error'] += 1
            continue
        frame, scale, left, top = letterbox(image)
        proposals = find_candidates_adaptive(frame)
        if not proposals:
            counts['no_candidate'] += 1
            continue
        mapped = corners * scale + np.float32([left, top])
        iou = quad_iou(mapped, proposals[0][1])
        if iou < args.min_iou:
            counts['low_iou'] += 1
            continue
        crop = warp_candidate(frame, proposals[0][1])
        identifier = hashlib.sha1(source.encode('utf-8')).hexdigest()[:20]
        relative = 'detector7/train/{}.jpg'.format(identifier)
        if not cv2.imwrite(str(args.data / relative), crop,
                           [cv2.IMWRITE_JPEG_QUALITY, 92]):
            counts['write_error'] += 1
            continue
        accepted.append((relative, row['label'], source, round(iou, 4)))
        counts['accepted'] += 1
        if index % 2000 == 0:
            print('processed={}/{} accepted={}'.format(
                index, len(rows), counts['accepted']), flush=True)
    with (args.data / 'detector7_train.tsv').open(
            'w', encoding='utf-8') as handle:
        for relative, label, _, _ in accepted:
            handle.write('{}\t{}\n'.format(relative, label))
    with (args.data / 'detector7_train_details.csv').open(
            'w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.writer(handle)
        writer.writerow(('crop_path', 'label', 'source_image', 'quad_iou'))
        writer.writerows(accepted)
    summary = {'selected_originals': len(rows),
               'minimum_quad_iou': args.min_iou,
               'counts': dict(counts),
               'output_tsv': str(args.data / 'detector7_train.tsv')}
    (args.data / 'detector7_summary.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
