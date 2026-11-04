#!/usr/bin/env python3
"""Frozen development-image comparison of blue/green locator changes.

Manual corners and text are used only after candidate generation, for scoring.
These repeated screen recaptures are development data, not an independent test.
"""
import argparse
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics
import sys
import time

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'ps'))


def module_from_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def records():
    manual = json.loads((ROOT / 'build/green_ocr_static_review/manual_comparison.json').read_text(encoding='utf-8'))
    corners = {int(row['frame']): row['manual_quad'] for row in manual['rows']}
    sources = (
        ('green_static5', 'green', ROOT / 'build/green_ocr_static_review/debug/matched', '*camera.png'),
        ('green_previous101', 'green', ROOT / 'build/green_ps_review/green_ps_debug/matched', '*camera.png'),
        ('blue_previous69', 'blue', ROOT / 'build/refine_all_frames', '*camera.jpg'),
    )
    for group, color, folder, pattern in sources:
        for path in sorted(folder.glob(pattern)):
            row = {'id': group + '/' + path.name, 'group': group, 'color': color,
                   'path': path.relative_to(ROOT).as_posix()}
            if group == 'green_static5':
                row.update(truth=manual['truth'], manual_quad=corners[int(path.name.split('_')[1])])
            yield row


def serialize(candidate):
    return {'score': float(candidate[0]),
            'quad': np.asarray(candidate[1]).astype(float).tolist(),
            'bbox': [int(x) for x in candidate[2]], 'area': float(candidate[3]),
            'ratio': float(candidate[4]), 'fill': float(candidate[5]),
            'color': candidate[6] if len(candidate) > 6 else None}


def geometry(candidate, truth):
    predicted = cv2.convexHull(np.float32(candidate['quad'])).reshape(-1, 2)
    truth = cv2.convexHull(np.float32(truth)).reshape(-1, 2)
    predicted_area, truth_area = cv2.contourArea(predicted), cv2.contourArea(truth)
    intersection = float(cv2.intersectConvexConvex(predicted, truth)[0])
    return {'iou': intersection / max(1., predicted_area + truth_area - intersection),
            'coverage': intersection / max(1., truth_area),
            'predicted_to_truth_area': predicted_area / max(1., truth_area)}


def summarize(rows):
    groups = {}
    for group in sorted(set(row['group'] for row in rows)):
        subset = [row for row in rows if row['group'] == group]
        summary = {'frames': len(subset),
                   'with_candidates': sum(bool(row['candidates']) for row in subset),
                   'candidate_total': sum(len(row['candidates']) for row in subset),
                   'candidate_count_distribution': dict(Counter(len(row['candidates']) for row in subset)),
                   'mean_locator_ms': statistics.mean(row['locator_ms'] for row in subset)}
        if group == 'green_static5':
            summary['top1_iou_mean_including_misses'] = statistics.mean(row.get('top1_geometry', {}).get('iou', 0.) for row in subset)
            summary['top1_coverage_mean_including_misses'] = statistics.mean(row.get('top1_geometry', {}).get('coverage', 0.) for row in subset)
            summary['top1_coverage_at_least_90pct'] = sum(row.get('top1_geometry', {}).get('coverage', 0.) >= .9 for row in subset)
            summary['best_candidate_iou_at_least_50pct'] = sum(max((item['iou'] for item in row['all_geometry']), default=0.) >= .5 for row in subset)
            if any('top1_ocr' in row for row in subset):
                summary['top1_ocr_exact_frames'] = sum(row.get('top1_ocr') == row['truth'] for row in subset)
        groups[group] = summary
    return groups


def evaluate(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if args.variant == 'baseline':
        locator_path = output / 'baseline_plate_locator_adaptive.py'
        warp_path = output / 'baseline_plate_ocr_baseline.py'
        if not locator_path.exists() or not warp_path.exists():
            raise SystemExit('Freeze baseline sources in output directory before evaluation')
    else:
        locator_path, warp_path = ROOT / 'ps/plate_locator_adaptive.py', ROOT / 'ps/plate_ocr_baseline.py'
    locator = module_from_file('_joint_eval_locator_' + args.variant, locator_path)
    warper = module_from_file('_joint_eval_warp_' + args.variant, warp_path)
    reader = None
    if args.ocr:
        from plate_ctc_infer import PlateCTCReader
        reader = PlateCTCReader(ROOT / 'build/green_pl_bundle/best.onnx')
    rows = []
    for source in records():
        image = cv2.imread(str(ROOT / source['path']))
        if image is None:
            raise RuntimeError('Cannot read ' + source['path'])
        started = time.perf_counter()
        color = 'auto' if args.auto else source['color']
        candidates = locator.find_candidates_adaptive(image, plate_color=color)
        row = dict(source, locator_ms=1000 * (time.perf_counter() - started),
                   input_sha256=sha(ROOT / source['path']),
                   candidates=[serialize(candidate) for candidate in candidates])
        if 'manual_quad' in source:
            row['all_geometry'] = [geometry(candidate, source['manual_quad']) for candidate in row['candidates']]
            if candidates:
                row['top1_geometry'] = row['all_geometry'][0]
                if reader:
                    row['top1_ocr'] = reader.recognize_crop(warper.warp_candidate(image, candidates[0][1]))
        rows.append(row)
    # Synthetic color rectangles have no text. These are regression fixtures,
    # not a real-scene false-positive benchmark.
    negatives = []
    for name, fill in [('black', None), ('blue_rectangle', (200, 60, 20)), ('green_rectangle', (70, 180, 80))]:
        image = np.zeros((480, 640, 3), np.uint8)
        if fill:
            cv2.rectangle(image, (180, 200), (420, 255), fill, -1)
        for color in (('auto',) if args.auto else ('blue', 'green')):
            candidates = locator.find_candidates_adaptive(image, plate_color=color)
            negatives.append({'fixture': name, 'color': color, 'candidate_count': len(candidates)})
    report = {'variant': args.variant, 'mode': 'auto' if args.auto else 'separate_color',
              'opencv': cv2.__version__, 'threads': 1,
              'source_hashes': {str(locator_path.relative_to(ROOT)): sha(locator_path),
                                str(warp_path.relative_to(ROOT)): sha(warp_path)},
              'limitations': ['Repeated screen recaptures are development data, not independent accuracy testing.',
                             'Only five frames of one plate have manual corners and text truth.',
                             'Candidate counts on unannotated sets are not accuracy.',
                             'Runtime timings are desktop measurements, not board performance.',
                             'Best-candidate geometry uses truth only for evaluation, never for detector selection.'],
              'summary': summarize(rows), 'synthetic_negatives': negatives, 'rows': rows}
    filename = args.variant + ('_auto' if args.auto else '') + '.json'
    (output / filename).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'report': str(output / filename), 'summary': report['summary'],
                      'synthetic_negatives': negatives}, ensure_ascii=True, indent=2))


def compare(args):
    baseline = json.loads((args.output / 'baseline.json').read_text(encoding='utf-8'))
    current_name = 'current_auto.json' if args.auto else 'current.json'
    current = json.loads((args.output / current_name).read_text(encoding='utf-8'))
    old = {row['id']: row for row in baseline['rows']}
    changed = []
    for row in current['rows']:
        before = old[row['id']]
        if before['input_sha256'] != row['input_sha256']:
            raise RuntimeError('Input changed: ' + row['id'])
        if before['candidates'] != row['candidates']:
            changed.append({'id': row['id'], 'old_count': len(before['candidates']),
                            'new_count': len(row['candidates']),
                            'old_top1_geometry': before.get('top1_geometry'),
                            'new_top1_geometry': row.get('top1_geometry'),
                            'old_top1_ocr': before.get('top1_ocr'), 'new_top1_ocr': row.get('top1_ocr')})
    result = {'baseline_summary': baseline['summary'], 'current_summary': current['summary'],
              'changed_frames': len(changed), 'changes': changed,
              'limitations': baseline['limitations']}
    path = args.output / ('comparison_auto.json' if args.auto else 'comparison.json')
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'report': str(path), 'changed_frames': len(changed)}, ensure_ascii=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant', choices=('baseline', 'current', 'compare'), required=True)
    parser.add_argument('--output', type=Path, default=ROOT / 'build/joint_locator_review')
    parser.add_argument('--auto', action='store_true')
    parser.add_argument('--ocr', action='store_true', help='OCR only the five annotated development frames')
    args = parser.parse_args()
    if args.variant == 'baseline' and args.auto:
        parser.error('Baseline has no auto mode')
    cv2.setNumThreads(1)
    compare(args) if args.variant == 'compare' else evaluate(args)


if __name__ == '__main__':
    main()
