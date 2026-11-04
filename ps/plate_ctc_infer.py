#!/usr/bin/env python3
"""Run exported tiny plate OCR with NumPy and OpenCV DNN, without PyTorch."""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np


class PlateCTCReader:
    def __init__(self, onnx_path, info_path=None):
        onnx_path = Path(onnx_path)
        if info_path is None:
            info_path = onnx_path.parent / 'model_info.json'
        info = json.loads(Path(info_path).read_text(encoding='utf-8'))
        self.vocab = info['vocab']
        self.width = int(info['input']['shape'][3])
        self.height = int(info['input']['shape'][2])
        self.net = cv2.dnn.readNetFromONNX(str(onnx_path))

    def recognize_crop(self, bgr):
        return self.recognize_details(bgr)['text']

    def recognize_details(self, bgr):
        """Return greedy-path scores, not calibrated correctness probabilities."""
        if bgr is None or bgr.size == 0:
            raise ValueError('Plate crop is empty')
        if bgr.ndim == 3:
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        else:
            gray = bgr
        image = cv2.resize(gray, (self.width, self.height),
                           interpolation=cv2.INTER_AREA)
        input_array = np.ascontiguousarray(
            image.astype(np.float32)[None, None] / 255.0)
        self.net.setInput(input_array)
        output = self.net.forward()
        if output.ndim == 4 and output.shape[0] == 1 and output.shape[2] == 1:
            logits = output[0, :, 0, :].T  # OpenCV 4.2 graph: class, 1, time.
        elif output.ndim == 3 and output.shape[0] == 1:
            logits = output[0]  # Earlier graph: time, class.
        else:
            raise ValueError('Unexpected ONNX output shape {}'.format(output.shape))
        kept = []
        scores = []
        shifted = logits - logits.max(axis=1, keepdims=True)
        log_probs = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
        previous = 0
        for step, value in enumerate(np.argmax(logits, axis=1)):
            index = int(value)
            if index != 0 and index != previous:
                kept.append(self.vocab[index - 1])
                scores.append(float(np.exp(log_probs[step, index])))
            previous = index
        return {'text': ''.join(kept), 'char_scores': scores,
                'score': float(np.exp(np.mean(np.log(np.maximum(scores, 1e-9))))) if scores else 0.0}


def main():
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('crop', type=Path)
    parser.add_argument('--model', type=Path,
                        default=Path('build/ocr_tiny_ctc_detector_ft/best.onnx'))
    args = parser.parse_args()
    image = cv2.imdecode(np.fromfile(str(args.crop), dtype=np.uint8),
                         cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(args.crop)
    reader = PlateCTCReader(args.model)
    print(json.dumps({'plate': reader.recognize_crop(image),
                      'crop': str(args.crop)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
