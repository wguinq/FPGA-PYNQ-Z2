#!/usr/bin/env python3
"""Evaluate the current classical plate locator on CCPD filename labels."""
import argparse
import csv
import hashlib
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from phase3_uvc_locator import (candidate_mask_from_bgr, find_candidates,
                                simulate_phase3_mask_from_bgr)


def parse_annotation(path):
    """Read CCPD bbox and four plate vertices embedded in an image filename."""
    fields = path.stem.split('-')
    if len(fields) < 7:
        return None
    try:
        bbox = [tuple(map(int, pair.split('&'))) for pair in fields[2].split('_')]
        polygon = [tuple(map(int, pair.split('&'))) for pair in fields[3].split('_')]
    except (TypeError, ValueError):
        return None
    if len(bbox) != 2 or len(polygon) != 4:
        return None
    x1, y1 = bbox[0]
    x2, y2 = bbox[1]
    if x2 <= x1 or y2 <= y1:
        return None
    return bbox, polygon


def letterbox(image, width=640, height=480):
    """Fit CCPD's portrait frames to the live 640x480 raster without distortion."""
    src_h, src_w = image.shape[:2]
    scale = min(width / float(src_w), height / float(src_h))
    out_w = max(1, int(round(src_w * scale)))
    out_h = max(1, int(round(src_h * scale)))
    resized = cv2.resize(image, (out_w, out_h), interpolation=cv2.INTER_AREA)
    left = (width - out_w) // 2
    top = (height - out_h) // 2
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    canvas[top:top + out_h, left:left + out_w] = resized
    return canvas, scale, left, top


def transform_points(points, scale, left, top):
    return np.asarray([(x * scale + left, y * scale + top)
                       for x, y in points], dtype=np.float32)


def polygon_iou(first, second):
    area_a = abs(float(cv2.contourArea(first)))
    area_b = abs(float(cv2.contourArea(second)))
    if area_a <= 0 or area_b <= 0:
        return 0.0
    try:
        intersection, _ = cv2.intersectConvexConvex(
            first.astype(np.float32), second.astype(np.float32))
    except cv2.error:
        return 0.0
    union = area_a + area_b - float(intersection)
    return float(intersection) / union if union > 0 else 0.0


def bbox_polygon(bbox):
    (x1, y1), (x2, y2) = bbox
    return np.asarray([[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
                      dtype=np.float32)


def choose_samples(root, split, max_per_subset, seed, include_np, excluded=None):
    excluded = excluded or set()
    split_file = root / 'splits' / (split + '.txt')
    if not split_file.is_file():
        raise FileNotFoundError('找不到 CCPD 划分文件: {}'.format(split_file))
    grouped = defaultdict(list)
    with split_file.open('r', encoding='utf-8') as handle:
        for line in handle:
            relative = line.strip().replace('\\', '/')
            if not relative:
                continue
            if relative in excluded:
                continue
            grouped[relative.split('/')[0]].append(relative)
    rng = random.Random(seed)
    selected = []
    for subset, relatives in sorted(grouped.items()):
        relatives.sort()
        if max_per_subset > 0 and len(relatives) > max_per_subset:
            relatives = sorted(rng.sample(relatives, max_per_subset))
        for relative in relatives:
            path = root.joinpath(*relative.split('/'))
            selected.append((subset, path, True))
    if include_np > 0:
        negatives = sorted(p for p in (root / 'ccpd_np').glob('*.jpg')
                           if 'ccpd_np/' + p.name not in excluded)
        if negatives:
            negatives = sorted(rng.sample(negatives, min(include_np, len(negatives))))
            selected.extend(('ccpd_np', path, False) for path in negatives)
    return selected


def read_manifest(root, filename):
    with Path(filename).open(encoding='utf-8-sig') as handle:
        rows = list(csv.DictReader(handle))
    return [(r['subset'], root / r['subset'] / r['filename'],
             bool(int(r['has_plate']))) for r in rows]


def evaluate_one(path, has_plate, detector='baseline', cb_threshold=150,
                 proposal_color='rgb', close_w=25, close_h=9,
                 min_edge_density=0.0):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        return {'status': 'read_error', 'iou_bbox': 0.0, 'iou_quad': 0.0,
                'candidate_count': 0}
    frame, scale, left, top = letterbox(image)
    if detector in ('adaptive', 'adaptive_pl', 'adaptive_pl7'):
        from plate_locator_adaptive import find_candidates_adaptive
        if detector == 'adaptive_pl':
            proposal = simulate_phase3_mask_from_bgr(frame)
        elif detector == 'adaptive_pl7':
            proposal = simulate_phase3_mask_from_bgr(frame, close_w=7, close_h=3)
        else:
            proposal = None
        candidates = find_candidates_adaptive(frame, proposal_mask=proposal)
    elif detector == 'projection':
        from plate_locator_projection import find_candidates_projection
        candidates = find_candidates_projection(frame, cb_threshold=cb_threshold)
    elif detector in ('adaptive_color', 'adaptive_color_edge',
                      'adaptive_color_edge_score'):
        from plate_mask_proposals import color_proposal_mask
        from plate_locator_adaptive import find_candidates_adaptive
        proposal = color_proposal_mask(
            frame, color=proposal_color, cb_threshold=cb_threshold,
            close_w=close_w, close_h=close_h)
        candidates = find_candidates_adaptive(
            frame, proposal_mask=proposal,
            min_edge_density=(min_edge_density if detector == 'adaptive_color_edge'
                              else 0.0),
            edge_score_weight=(min_edge_density
                               if detector == 'adaptive_color_edge_score'
                               else 0.0))
    else:
        candidates = find_candidates(candidate_mask_from_bgr(frame), frame, debug=False)
    result = {'status': 'miss' if has_plate else 'false_positive',
              'iou_bbox': 0.0, 'iou_quad': 0.0,
              'candidate_count': len(candidates)}
    if not has_plate:
        if not candidates:
            result['status'] = 'true_negative'
        return result
    annotation = parse_annotation(path)
    if annotation is None:
        result['status'] = 'annotation_error'
        return result
    if not candidates:
        return result

    bbox, polygon = annotation
    gt_bbox = transform_points(bbox_polygon(bbox), scale, left, top)
    gt_quad = transform_points(polygon, scale, left, top)
    predicted = candidates[0][1].astype(np.float32)
    x1, y1 = np.min(gt_bbox, axis=0)
    x2, y2 = np.max(gt_bbox, axis=0)
    p1, q1 = np.min(predicted, axis=0)
    p2, q2 = np.max(predicted, axis=0)
    intersection = max(0.0, min(x2, p2) - max(x1, p1)) * max(
        0.0, min(y2, q2) - max(y1, q1))
    union = max(0.0, (x2 - x1) * (y2 - y1)) + max(
        0.0, (p2 - p1) * (q2 - q1)) - intersection
    result['iou_bbox'] = float(intersection / union) if union > 0 else 0.0
    result['iou_quad'] = polygon_iou(predicted, gt_quad)
    result['status'] = 'detected'
    return result


def summarize_rows(rows):
    """Exclude unreadable/invalid-label images from metric denominators."""
    summary = []
    for subset in sorted(set(row['subset'] for row in rows)):
        subset_rows = [row for row in rows if row['subset'] == subset]
        valid = [row for row in subset_rows
                 if row['status'] not in ('read_error', 'annotation_error')]
        positives = [row for row in valid if int(row['has_plate'])]
        negatives = [row for row in valid if not int(row['has_plate'])]
        summary.append({
            'subset': subset,
            'images': len(subset_rows),
            'positives': sum(int(r['has_plate']) for r in subset_rows),
            'negatives': sum(not int(r['has_plate']) for r in subset_rows),
            'evaluable_positives': len(positives),
            'evaluable_negatives': len(negatives),
            'detection_rate': (sum(int(r['candidate_count']) > 0 for r in positives) /
                               len(positives)) if positives else '',
            'recall_iou50': (sum(float(r['iou_bbox']) >= .5 for r in positives) /
                             len(positives)) if positives else '',
            'recall_iou70': (sum(float(r['iou_bbox']) > .7 for r in positives) /
                             len(positives)) if positives else '',
            'mean_iou_bbox': (sum(float(r['iou_bbox']) for r in positives) /
                              len(positives)) if positives else '',
            'mean_iou_quad': (sum(float(r['iou_quad']) for r in positives) /
                              len(positives)) if positives else '',
            'false_positive_rate': (sum(int(r['candidate_count']) > 0 for r in negatives) /
                                    len(negatives)) if negatives else '',
            'annotation_errors': sum(r['status'] == 'annotation_error' for r in subset_rows),
            'read_errors': sum(r['status'] == 'read_error' for r in subset_rows),
        })
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset_root', help='CCPD2019 根目录，含 splits/ 与 ccpd_* 子目录')
    parser.add_argument('--split', choices=('val', 'test'), default='test')
    parser.add_argument('--max-per-subset', type=int, default=200,
                        help='每个 split 子集抽样数；0 表示该 split 全量')
    parser.add_argument('--include-np', type=int, default=200,
                        help='额外抽取多少张无牌负样本')
    parser.add_argument('--seed', type=int, default=20261025)
    parser.add_argument('--detector', choices=('baseline', 'adaptive', 'adaptive_pl', 'adaptive_pl7', 'projection', 'adaptive_color', 'adaptive_color_edge', 'adaptive_color_edge_score'), default='baseline')
    parser.add_argument('--cb-threshold', type=int, default=150,
                        help='Cb threshold for projection/color-mask experiments')
    parser.add_argument('--proposal-color', choices=('rgb', 'hsv', 'cb'), default='rgb',
                        help='color predicate for adaptive_color detector')
    parser.add_argument('--close-w', type=int, default=25,
                        help='close-kernel width for adaptive_color detector')
    parser.add_argument('--close-h', type=int, default=9,
                        help='close-kernel height for adaptive_color detector')
    parser.add_argument('--min-edge-density', type=float, default=0.0,
                        help='Sobel edge-density gate, or edge-score weight for adaptive_color_edge_score')
    parser.add_argument('--manifest', help='重放已有 detail CSV 的完全相同样本')
    parser.add_argument('--exclude-manifest', action='append', default=[],
                        help='抽样前排除这些 detail CSV 中的图片，可重复指定')
    parser.add_argument('--output', default='build/ccpd_eval')
    args = parser.parse_args()

    root = Path(args.dataset_root).resolve()
    if not (root / 'splits').is_dir():
        raise SystemExit('dataset_root 应指向包含 splits/ 的 CCPD2019 目录')
    excluded = set()
    for filename in args.exclude_manifest:
        excluded.update(subset + '/' + path.name
                        for subset, path, _ in read_manifest(root, filename))
    if args.manifest:
        samples = read_manifest(root, args.manifest)
    else:
        samples = choose_samples(root, args.split, args.max_per_subset,
                                 args.seed, args.include_np, excluded)
    if not samples:
        raise SystemExit('划分文件中没有找到可读取的样本')

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    started = time.perf_counter()
    for index, (subset, path, has_plate) in enumerate(samples, 1):
        result = evaluate_one(path, has_plate, args.detector, args.cb_threshold,
                              args.proposal_color, args.close_w, args.close_h,
                              args.min_edge_density)
        rows.append({'subset': subset, 'filename': path.name,
                     'has_plate': int(has_plate), **result})
        if index % 100 == 0 or index == len(samples):
            print('processed {}/{}'.format(index, len(samples)), flush=True)

    detail_path = output.with_name(output.name + '_detail.csv')
    summary_path = output.with_name(output.name + '_summary.csv')
    with detail_path.open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = summarize_rows(rows)
    with summary_path.open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)

    metadata = {
        'arguments': vars(args), 'dataset_root': str(root),
        'sample_count': len(samples), 'elapsed_seconds': time.perf_counter() - started,
        'preprocessing': '640x480 aspect-preserving letterbox, INTER_AREA',
        'metric': 'top-1 enclosing axis-aligned bbox IoU; iou70 uses strictly >0.70',
        'invalid_samples': 'read_error and annotation_error excluded from metric denominators',
        'opencv_version': cv2.__version__, 'numpy_version': np.__version__,
        'sample_sha256': hashlib.sha256('\n'.join(
            subset + '/' + path.name for subset, path, _ in samples).encode()).hexdigest(),
        'source_sha256': {name: hashlib.sha256(
            Path(__file__).with_name(name).read_bytes()).hexdigest() for name in (
                'evaluate_ccpd.py', 'phase3_uvc_locator.py', 'plate_locator_adaptive.py',
                'plate_locator_projection.py', 'plate_mask_proposals.py')},
    }
    with output.with_name(output.name + '_metadata.json').open('w', encoding='utf-8') as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)

    print('subset images pos neg detect recall@0.5 recall@0.7 meanIoU FP_rate')
    for row in summary:
        print('{} {} {} {} {} {} {} {} {}'.format(
            row['subset'], row['images'], row['positives'], row['negatives'],
            '{:.3f}'.format(row['detection_rate']) if row['detection_rate'] != '' else '-',
            '{:.3f}'.format(row['recall_iou50']) if row['recall_iou50'] != '' else '-',
            '{:.3f}'.format(row['recall_iou70']) if row['recall_iou70'] != '' else '-',
            '{:.3f}'.format(row['mean_iou_bbox']) if row['mean_iou_bbox'] != '' else '-',
            '{:.3f}'.format(row['false_positive_rate']) if row['false_positive_rate'] != '' else '-'))
    print('detail:', detail_path)
    print('summary:', summary_path)


if __name__ == '__main__':
    main()
