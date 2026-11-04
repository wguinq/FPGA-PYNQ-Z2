#!/usr/bin/env python3
"""Smoke-test and time an exported plate OCR model on a PYNQ board.

Place this file, plate_ctc_infer.py, best.onnx, model_info.json,
known_plate.jpg, and expected.txt in one directory. No PYNQ overlay or
camera is loaded by this test.
"""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from plate_ctc_infer import PlateCTCReader


def main():
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, default=here / 'best.onnx')
    parser.add_argument('--sample', type=Path, default=here / 'known_plate.jpg')
    parser.add_argument('--expected-file', type=Path, default=here / 'expected.txt')
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--runs', type=int, default=50)
    args = parser.parse_args()
    if args.warmup < 0 or args.runs <= 0:
        parser.error('warmup must be nonnegative and runs must be positive')
    expected = args.expected_file.read_text(encoding='utf-8').strip()
    if not expected:
        raise ValueError('expected.txt is empty')
    image = cv2.imdecode(np.fromfile(str(args.sample), dtype=np.uint8),
                         cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(args.sample)
    cv2.setNumThreads(1)
    started = time.perf_counter()
    try:
        reader = PlateCTCReader(args.model)
    except Exception as exc:
        print(json.dumps({'status': 'model_load_failed',
                          'opencv': cv2.__version__, 'error': str(exc)},
                         ensure_ascii=False), flush=True)
        return 2
    load_ms = (time.perf_counter() - started) * 1000.0
    try:
        for _ in range(args.warmup):
            reader.recognize_crop(image)
        times = []
        prediction = None
        for _ in range(args.runs):
            started = time.perf_counter()
            prediction = reader.recognize_crop(image)
            times.append((time.perf_counter() - started) * 1000.0)
    except Exception as exc:
        print(json.dumps({'status': 'inference_failed',
                          'opencv': cv2.__version__, 'load_ms': load_ms,
                          'error': str(exc)}, ensure_ascii=False), flush=True)
        return 3
    times.sort()
    report = {'status': 'pass' if prediction == expected else 'wrong_text',
              'opencv': cv2.__version__, 'image_size': list(image.shape[:2]),
              'expected': expected, 'predicted': prediction,
              'load_ms': round(load_ms, 2),
              'median_inference_ms': round(statistics.median(times), 2),
              'p95_inference_ms': round(times[int(.95 * (len(times) - 1))], 2),
              'runs': args.runs}
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 0 if prediction == expected else 4


if __name__ == '__main__':
    sys.exit(main())
