"""Copy a small, reproducible subset of evaluated CCPD images for board testing."""
import argparse
import csv
import hashlib
import json
import random
import shutil
from pathlib import Path

import cv2

from evaluate_ccpd import summarize_rows


def read_rows(path):
    with path.open(encoding='utf-8-sig') as handle:
        return list(csv.DictReader(handle))


def write_rows(path, rows):
    with path.open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset_root', type=Path)
    parser.add_argument('--output', type=Path, default=Path('ccpd_board_sample'))
    parser.add_argument('--per-subset', type=int, default=5)
    parser.add_argument('--seed', type=int, default=20261027)
    args = parser.parse_args()
    if args.per_subset < 1:
        parser.error('--per-subset must be positive')
    project = Path(__file__).resolve().parents[1]
    baseline, adaptive = {}, {}
    for group in ('holdout', 'test'):
        for detector, table in (('baseline', baseline), ('adaptive', adaptive)):
            name = ('ccpd_test_sample_detail.csv' if group == 'test' and detector == 'baseline'
                    else 'ccpd_{}_{}_detail.csv'.format(group, detector))
            for row in read_rows(project / 'build' / name):
                if group == 'test' and row['subset'] == 'ccpd_np':
                    continue
                table[(row['subset'], row['filename'])] = row
    rng = random.Random(args.seed)
    selected = []
    for subset in sorted({key[0] for key in adaptive}):
        pool = sorted(key for key in adaptive if key[0] == subset and key in baseline
                      and adaptive[key]['status'] not in ('read_error', 'annotation_error')
                      and baseline[key]['status'] not in ('read_error', 'annotation_error'))
        if len(pool) < args.per_subset:
            raise ValueError('Not enough evaluated images in ' + subset)
        selected.extend(sorted(rng.sample(pool, args.per_subset)))

    # Refuse to overwrite an existing user folder or earlier sample package.
    args.output.mkdir(parents=True, exist_ok=False)
    inventory = []
    for subset, filename in selected:
        source = args.dataset_root / subset / filename
        target = args.output / subset / filename
        target.parent.mkdir(exist_ok=True)
        shutil.copy2(source, target)
        checksum = digest(source)
        if digest(target) != checksum:
            raise RuntimeError('Copy checksum mismatch: ' + str(target))
        decoded = cv2.imread(str(target))
        if decoded is None:
            raise RuntimeError('Unreadable image: ' + str(target))
        inventory.append(dict(path=subset + '/' + filename, sha256=checksum,
                              bytes=target.stat().st_size, width=decoded.shape[1],
                              height=decoded.shape[0]))

    write_rows(args.output / 'manifest.csv', [adaptive[key] for key in selected])
    for detector, table in (('baseline', baseline), ('adaptive', adaptive)):
        rows = [table[key] for key in selected]
        write_rows(args.output / ('pc_' + detector + '_detail.csv'), rows)
        write_rows(args.output / ('pc_' + detector + '_summary.csv'), summarize_rows(rows))
    split = args.output / 'splits'
    split.mkdir()
    (split / 'test.txt').write_text(''.join(
        subset + '/' + filename + '\n' for subset, filename in selected
        if int(adaptive[(subset, filename)]['has_plate'])), encoding='utf-8')
    metadata = dict(seed=args.seed, per_subset=args.per_subset,
                    source_root=str(args.dataset_root.resolve()),
                    selection='Random within previously evaluated subsets; independent of detection outcome',
                    images=inventory)
    (args.output / 'checksums.json').write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding='utf-8')
    print('folder:', args.output.resolve())
    print('images:', len(selected))
    print('image_bytes:', sum(row['bytes'] for row in inventory))
    print('all copies decoded and SHA256 matched')


if __name__ == '__main__':
    main()
