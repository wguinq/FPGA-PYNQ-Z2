#!/usr/bin/env python3
"""Short board launcher: python3 run_plate.py [blue|preview]."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', nargs='?', default='blue',
                        choices=('blue', 'preview'))
    parser.add_argument('--debug', action='store_true', help='Save matched diagnostic images (slower)')
    parser.add_argument('--frames', type=int, default=0, help='0: run until Ctrl-C')
    parser.add_argument('--buttons', action='store_true', help='Enable capture, detection and (3-button overlay) character-view toggle')
    parser.add_argument('--dry-run', action='store_true', help='Print command without running')
    args = parser.parse_args()
    if args.frames < 0:
        parser.error('--frames must be nonnegative')
    root = Path(__file__).resolve().parent
    bitfile = root / ('phase3_hdmi_chars.bit' if (root / 'phase3_hdmi_chars.bit').exists()
                      else 'phase3_hdmi_green.bit')
    python = Path('/usr/local/share/pynq-venv/bin/python3')
    interpreter = str(python) if python.exists() else sys.executable
    detector = 'adaptive_pl' if args.mode == 'blue' else 'off'
    result_dir = root / 'runs' / (datetime.now().strftime('%Y%m%d_%H%M%S_%f') + '_' + args.mode)
    command = [interpreter, str(root / 'phase3_hdmi_overlay.py'),
               str(bitfile), '--detector', detector,
               '--camera-format', 'YUYV', '--camera-buffers', '4',
               '--max-result-age', '3', '--frames', str(args.frames)]
    if args.mode != 'preview':
        command += ['--max-ocr-plates', '2', '--ocr-model', str(root / 'best.onnx'),
                    '--ocr-log', str(result_dir / 'live.csv')]
        if args.debug:
            command += ['--ocr-debug-dir', str(result_dir / 'debug')]
    elif args.debug:
        parser.error('--debug requires detection')
    command += ['--buttons']
    if args.dry_run:
        print(shlex.join(command))
        return 0
    # Re-enter with the PYNQ interpreter and root before creating result files.
    if hasattr(os, 'geteuid') and os.geteuid() != 0:
        os.execvp('sudo', ['sudo', '-E', interpreter, str(Path(__file__).resolve())] + sys.argv[1:])
    for name in ('phase3_hdmi_overlay.py', bitfile.name, bitfile.with_suffix('.hwh').name):
        if not (root / name).is_file():
            parser.error('Missing file beside launcher: ' + name)
    if args.mode != 'preview':
        for name in ('best.onnx', 'model_info.json'):
            if not (root / name).is_file():
                parser.error('Missing file beside launcher: ' + name)
    result_dir.mkdir(parents=True, exist_ok=False)
    (result_dir / 'command.json').write_text(
        json.dumps(command, ensure_ascii=False, indent=2), encoding='utf-8')
    print('Results: {}'.format(result_dir), flush=True)
    print('Mode: {} ({})'.format(args.mode, detector), flush=True)
    try:
        return subprocess.call(command, cwd=str(root))
    except KeyboardInterrupt:
        return 130


if __name__ == '__main__':
    sys.exit(main())
