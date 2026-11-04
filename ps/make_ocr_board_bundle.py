#!/usr/bin/env python3
"""Pack a known-answer OCR smoke test for transfer to the PYNQ board."""
import argparse
import csv
import hashlib
import shutil
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', type=Path,
                        default=Path('build/ocr_tiny_ctc_detector_ft'))
    parser.add_argument('--onnx', type=Path,
                        default=Path('build/ocr_tiny_ctc_detector_ft/best_opencv42.onnx'),
                        help='OpenCV 4.2 compatible ONNX export')
    parser.add_argument('--data', type=Path,
                        default=Path('build/ocr_real_crops'))
    parser.add_argument('--sample', type=Path,
                        default=Path('build/ocr_real_crops/base7/val/'
                                     '000824f17aff97f41680.jpg'))
    parser.add_argument('--output', type=Path,
                        default=Path('build/ocr_board_bundle'))
    args = parser.parse_args()
    sample_path = args.sample.resolve()
    with (args.data / 'manifest.csv').open(encoding='utf-8-sig',
                                           newline='') as handle:
        labels = [row['label'] for row in csv.DictReader(handle)
                  if (args.data / row['crop_path']).resolve() == sample_path]
    if len(labels) != 1:
        raise ValueError('Expected exactly one manifest label for {}'.format(
            sample_path))
    args.output.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parent
    sources = {
        'best.onnx': args.onnx,
        'model_info.json': args.model_dir / 'model_info.json',
        'known_plate.jpg': args.sample,
        'plate_ctc_infer.py': root / 'plate_ctc_infer.py',
        'verify_plate_ctc_board.py': root / 'verify_plate_ctc_board.py',
        'phase3_hdmi_overlay.py': root / 'phase3_hdmi_overlay.py',
        'plate_glyphs.npz': root / 'plate_glyphs.npz',
        'plate_ocr_baseline.py': root / 'plate_ocr_baseline.py',
        'plate_locator_adaptive.py': root / 'plate_locator_adaptive.py',
        'phase3_uvc_locator.py': root / 'phase3_uvc_locator.py',
    }
    for name, source in sources.items():
        shutil.copy2(source, args.output / name)
    with (args.output / 'expected.txt').open('w', encoding='utf-8',
                                            newline='\n') as handle:
        handle.write(labels[0] + '\n')
    names = sorted(sources) + ['expected.txt']
    checksums = []
    for name in names:
        digest = hashlib.sha256((args.output / name).read_bytes()).hexdigest()
        checksums.append('{}  {}'.format(digest, name))
    # sha256sum on Linux treats CRLF's carriage return as part of each name.
    with (args.output / 'SHA256SUMS').open('w', encoding='ascii',
                                         newline='\n') as handle:
        handle.write('\n'.join(checksums) + '\n')
    print('bundle={} files={} expected={}'.format(args.output, len(names),
                                                   labels[0]))


if __name__ == '__main__':
    main()
