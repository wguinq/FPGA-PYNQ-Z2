#!/usr/bin/env python3
"""Export trained TinyPlateCTC to ONNX and verify numerical equivalence."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import onnx
import onnxruntime as ort
import torch
from torch import nn

from train_tiny_plate_ctc import HEIGHT, WIDTH, VOCAB, TinyPlateCTC


class OpenCV42PlateCTC(nn.Module):
    """Convert trained weights to a graph of only 2-D Conv, Relu and Add."""

    def __init__(self, trained):
        super().__init__()
        self.features = trained.features
        self.height_average = nn.Conv2d(128, 128, (2, 1),
                                         groups=128, bias=False)
        self.sequence = nn.ModuleList()
        with torch.no_grad():
            self.height_average.weight.fill_(0.5)
            for source in trained.sequence:
                depthwise, pointwise, norm, _ = source.net
                depth_2d = nn.Conv2d(
                    128, 128, (1, depthwise.kernel_size[0]),
                    padding=(0, depthwise.padding[0]),
                    dilation=(1, depthwise.dilation[0]),
                    groups=depthwise.groups, bias=False)
                depth_2d.weight.copy_(depthwise.weight.unsqueeze(2))
                point_2d = nn.Conv2d(128, 128, 1, bias=True)
                scale = norm.weight / torch.sqrt(norm.running_var + norm.eps)
                point_2d.weight.copy_(pointwise.weight.unsqueeze(2) *
                                      scale[:, None, None, None])
                point_2d.bias.copy_(norm.bias - norm.running_mean * scale)
                self.sequence.append(nn.Sequential(depth_2d, point_2d,
                                                   nn.ReLU()))
            source_head = trained.classifier
            self.classifier = nn.Conv2d(128, source_head.out_channels, 1)
            self.classifier.weight.copy_(source_head.weight.unsqueeze(2))
            self.classifier.bias.copy_(source_head.bias)

    def forward(self, images):
        x = self.height_average(self.features(images))
        for block in self.sequence:
            x = x + block(x)
        return self.classifier(x)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path,
                        default=Path('build/ocr_tiny_ctc_detector_ft/best.pt'))
    parser.add_argument('--output', type=Path,
                        default=Path('build/ocr_tiny_ctc_detector_ft/best.onnx'))
    parser.add_argument('--sample', type=Path,
                        default=Path('build/ocr_real_crops/base7/val'))
    args = parser.parse_args()
    torch.set_num_threads(min(4, torch.get_num_threads()))
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    if saved['vocab'] != VOCAB:
        raise ValueError('Checkpoint vocabulary differs from code')
    model = TinyPlateCTC().eval()
    model.load_state_dict(saved['state_dict'])
    portable = OpenCV42PlateCTC(model).eval()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.zeros(1, 1, HEIGHT, WIDTH)
    torch.onnx.export(portable, dummy, str(args.output),
                      input_names=['plate'], output_names=['logits'],
                      opset_version=11, dynamo=False)
    exported = onnx.load(str(args.output))
    onnx.checker.check_model(exported)
    allowed_ops = {'Conv', 'Relu', 'Add'}
    unexpected = {node.op_type for node in exported.graph.node} - allowed_ops
    if unexpected:
        raise RuntimeError('Unexpected ONNX ops for OpenCV 4.2: {}'.format(
            sorted(unexpected)))
    for node in exported.graph.node:
        if node.op_type == 'Conv':
            kernel = next((attribute.ints for attribute in node.attribute
                           if attribute.name == 'kernel_shape'), None)
            if kernel is None or len(kernel) != 2:
                raise RuntimeError('Export still contains a non-2-D convolution')
    sample = args.sample
    if sample.is_dir():
        sample = next(sample.glob('*.jpg'))
    image = cv2.imread(str(sample), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(sample)
    image = cv2.resize(image, (WIDTH, HEIGHT),
                       interpolation=cv2.INTER_AREA)
    input_array = image.astype(np.float32)[None, None] / 255.0
    with torch.no_grad():
        reference = model(torch.from_numpy(input_array)).numpy()
        portable_result = portable(torch.from_numpy(input_array)).numpy()
    portable_canonical = np.transpose(portable_result[:, :, 0, :], (0, 2, 1))
    wrapper_difference = float(np.max(np.abs(reference - portable_canonical)))
    if wrapper_difference > 1e-4:
        raise RuntimeError('2-D wrapper differs from trained model: {}'.format(
            wrapper_difference))
    session = ort.InferenceSession(str(args.output),
                                   providers=['CPUExecutionProvider'])
    actual = session.run(['logits'], {'plate': input_array})[0]
    actual_canonical = np.transpose(actual[:, :, 0, :], (0, 2, 1))
    difference = float(np.max(np.abs(reference - actual_canonical)))
    if difference > 1e-3:
        raise RuntimeError('ONNX output differs from PyTorch: {}'.format(
            difference))
    metadata = {'vocab': VOCAB, 'blank_id': 0,
                'input': {'name': 'plate', 'shape': [1, 1, HEIGHT, WIDTH],
                          'dtype': 'float32', 'range': '0..1',
                          'color': 'grayscale'},
                'output': {'name': 'logits', 'shape': list(actual.shape),
                           'layout': 'NCHW, height=1; decode width as time'},
                'parameters': saved['parameters'],
                'checkpoint_epoch': saved['epoch'],
                'max_abs_diff_2d_wrapper_vs_torch': wrapper_difference,
                'max_abs_diff_onnx_vs_torch': difference}
    (args.output.parent / 'model_info.json').write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding='utf-8')
    print(json.dumps({'onnx': str(args.output),
                      'bytes': args.output.stat().st_size,
                      'max_abs_diff': difference,
                      'sample': str(sample)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
