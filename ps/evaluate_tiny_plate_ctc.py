#!/usr/bin/env python3
"""Evaluate tiny CTC OCR on CCPD originals, including 640x480 detector crops."""
import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

from plate_ocr_baseline import (letterbox, parse_ccpd, plate_text, quad_iou,
                                read_split, warp_candidate)
from prepare_real_plate_crops import rectify
from train_tiny_plate_ctc import (HEIGHT, WIDTH, VOCAB, TinyPlateCTC,
                                  greedy_decode)


def main():
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path,
                        default=Path('F:/data/CCPD-Dataset/data/CCPD2019'))
    parser.add_argument('--checkpoint', type=Path,
                        default=Path('build/ocr_tiny_ctc_disjoint/best.pt'))
    parser.add_argument('--split', default='val')
    parser.add_argument('--source', choices=('gt_live', 'detector'),
                        default='gt_live')
    parser.add_argument('--limit', type=int, default=500)
    parser.add_argument('--seed', type=int, default=20261030)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--output', type=Path,
                        default=Path('build/ocr_tiny_ctc_disjoint/eval'))
    args = parser.parse_args()
    if args.source == 'detector':
        from plate_locator_adaptive import find_candidates_adaptive
    torch.set_num_threads(min(4, torch.get_num_threads()))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    saved = torch.load(args.checkpoint, map_location=device,
                       weights_only=False)
    if saved['vocab'] != VOCAB:
        raise ValueError('Checkpoint vocabulary differs from code')
    model = TinyPlateCTC().to(device).eval()
    model.load_state_dict(saved['state_dict'])
    paths = read_split(args.root, args.split, args.limit, args.seed)
    crops, records = [], []
    for relative in paths:
        parsed = parse_ccpd(relative)
        if parsed is None:
            continue
        corners, codes = parsed
        image = cv2.imread(str(args.root.joinpath(*relative.split('/'))),
                           cv2.IMREAD_COLOR)
        if image is None:
            continue
        frame, scale, left, top = letterbox(image)
        mapped = corners * scale + np.float32([left, top])
        truth = plate_text(codes)
        iou = 1.0
        crop = None
        if args.source == 'detector':
            proposals = find_candidates_adaptive(frame)
            if proposals:
                iou = quad_iou(mapped, proposals[0][1])
                crop = warp_candidate(frame, proposals[0][1])
        else:
            crop = rectify(frame, mapped)
        record = {'path': relative, 'truth': truth, 'prediction': '',
                  'detector_quad_iou': round(iou, 4),
                  'no_candidate': crop is None}
        records.append(record)
        if crop is not None:
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            crops.append((len(records) - 1, cv2.resize(
                gray, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)))
    with torch.no_grad():
        for start in range(0, len(crops), args.batch_size):
            batch = crops[start:start + args.batch_size]
            images = np.stack([image for _, image in batch])
            tensor = torch.from_numpy(images).float().to(device).unsqueeze(1)
            guesses = greedy_decode(model(tensor / 255.0))
            for (index, _), guess in zip(batch, guesses):
                records[index]['prediction'] = guess
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / 'predictions.csv').open(
            'w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    found = [r for r in records if not r['no_candidate']]
    correct = [r for r in records if r['prediction'] == r['truth']]
    high_iou = [r for r in found if r['detector_quad_iou'] > .7]
    summary = {'split': args.split, 'source': args.source,
               'total': len(records), 'found': len(found),
               'no_candidate': len(records) - len(found),
               'exact': len(correct),
               'exact_rate_all': len(correct) / len(records),
               'exact_rate_found': len(correct) / len(found) if found else 0,
               'iou_gt_070': len(high_iou),
               'exact_rate_iou_gt_070': (sum(r['prediction'] == r['truth']
                                             for r in high_iou) / len(high_iou)
                                         if high_iou else 0)}
    (args.output / 'summary.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
