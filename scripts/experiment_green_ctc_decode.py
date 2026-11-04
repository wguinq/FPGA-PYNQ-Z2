#!/usr/bin/env python3
"""Offline CTC decoder comparison; never changes the deployed OCR runtime.

The optional length constraint is exactly eight emitted characters, without
hardcoded character corrections or ground-truth-assisted hypothesis selection.
It is a diagnostic prior, not a calibrated acceptance rule.
"""
import argparse
import json
import math
from pathlib import Path
import sys
import time

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'ps'))
from plate_ctc_infer import PlateCTCReader

NEG_INF = float('-inf')


def logadd(a, b):
    if a == NEG_INF:
        return b
    if b == NEG_INF:
        return a
    high, low = (a, b) if a >= b else (b, a)
    return high + math.log1p(math.exp(low - high))


def prefix_beam(log_probs, beam_width=10, target_length=None):
    """Return label prefixes with summed CTC alignment log probabilities.

    All model classes are visited at each timestep. Beam pruning remains an
    approximation; accumulated masses are not exact full CTC posteriors.
    """
    beam = {(): (0.0, NEG_INF)}
    steps, classes = log_probs.shape
    for t, row in enumerate(log_probs):
        following = {}

        def update(prefix, index, value):
            if value == NEG_INF:
                return
            remaining = steps - t - 1
            if target_length is not None and not (
                    len(prefix) <= target_length <= len(prefix) + remaining):
                return
            state = following.setdefault(prefix, [NEG_INF, NEG_INF])
            state[index] = logadd(state[index], value)

        for prefix, (blank, nonblank) in beam.items():
            total = logadd(blank, nonblank)
            update(prefix, 0, total + float(row[0]))
            last = prefix[-1] if prefix else None
            for char in range(1, classes):
                probability = float(row[char])
                if char == last:
                    # Staying at the same prefix consumes a repeated raw label;
                    # emitting the same label again requires an intervening blank.
                    update(prefix, 1, nonblank + probability)
                    update(prefix + (char,), 1, blank + probability)
                else:
                    update(prefix + (char,), 1, total + probability)
        beam = dict(sorted(following.items(),
                           key=lambda item: logadd(*item[1]), reverse=True)[:beam_width])
    ranked = sorted(((prefix, logadd(*mass)) for prefix, mass in beam.items()),
                    key=lambda item: item[1], reverse=True)
    if target_length is not None:
        ranked = [item for item in ranked if len(item[0]) == target_length]
    return ranked


def greedy(log_probs):
    result, previous = [], 0
    for char in log_probs.argmax(axis=1):
        char = int(char)
        if char and char != previous:
            result.append(char)
        previous = char
    return tuple(result)


def predict_log_probs(reader, crop):
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(gray, (reader.width, reader.height), interpolation=cv2.INTER_AREA)
    reader.net.setInput(np.ascontiguousarray(image.astype(np.float32)[None, None] / 255.0))
    output = reader.net.forward()
    if output.ndim == 4 and output.shape[2] == 1:
        logits = output[0, :, 0, :].T
    elif output.ndim == 3:
        logits = output[0]
    else:
        raise ValueError('Unexpected output {}'.format(output.shape))
    shifted = logits.astype(np.float64) - logits.max(axis=1, keepdims=True)
    return shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, default=ROOT / 'build/green_pl_bundle/best.onnx')
    parser.add_argument('--crops', type=Path, default=ROOT / 'build/green_ocr_static_review/manual_crops')
    parser.add_argument('--output', type=Path, default=ROOT / 'build/green_ocr_static_review/ctc_decode_experiment.json')
    parser.add_argument('--beam-width', type=int, default=10)
    parser.add_argument('--truth', default='皖ADB8617', help='Scoring only; never passed to the decoder')
    args = parser.parse_args()
    if args.beam_width <= 0:
        parser.error('beam width must be positive')
    cv2.setNumThreads(1)
    reader = PlateCTCReader(args.model)
    reader.net.setInput(np.zeros((1, 1, reader.height, reader.width), np.float32))
    reader.net.forward()
    rows = []
    for path in sorted(args.crops.glob('*.png')):
        crop = cv2.imread(str(path))
        if crop is None:
            raise RuntimeError('Cannot read {}'.format(path))
        started = time.perf_counter()
        log_probs = predict_log_probs(reader, crop)
        inference_ms = (time.perf_counter() - started) * 1000
        decode = lambda indices: ''.join(reader.vocab[index - 1] for index in indices)
        result = {'crop': str(path.relative_to(ROOT)), 'inference_ms': inference_ms}
        started = time.perf_counter()
        result['greedy'] = decode(greedy(log_probs))
        result['greedy_ms'] = (time.perf_counter() - started) * 1000
        for name, length in [('beam', None), ('beam_length8', 8)]:
            started = time.perf_counter()
            ranked = prefix_beam(log_probs, args.beam_width, length)
            result[name + '_ms'] = (time.perf_counter() - started) * 1000
            result[name] = decode(ranked[0][0]) if ranked else None
            result[name + '_top5'] = [
                {'text': decode(prefix), 'pruned_log_mass': mass} for prefix, mass in ranked[:5]]
        rows.append(result)
        print(json.dumps(result, ensure_ascii=True))
    report = {'model': str(args.model), 'opencv': cv2.__version__, 'cv_threads': 1,
              'beam_width': args.beam_width, 'class_pruning': False, 'truth': args.truth,
              'note': 'Five recaptured frames of one plate with manually supplied corners; diagnostic only, not an independent test. CPU decoding timing is not board timing. Length8 prior can force a plausible wrong plate.',
              'matches': {name: sum(row[name] == args.truth for row in rows)
                          for name in ('greedy', 'beam', 'beam_length8')}, 'rows': rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report['matches']))


if __name__ == '__main__':
    main()
