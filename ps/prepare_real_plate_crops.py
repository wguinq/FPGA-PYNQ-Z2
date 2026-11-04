#!/usr/bin/env python3
"""Export labelled, perspective-corrected real CCPD plate crops.

Only the official CCPD2019 Base train/val and CCPD2020 Green train/val
partitions are read. No difficult 2019 test subset or 2020 test image is
included. Seven- and eight-character plates are written separately.
"""
import argparse
import csv
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from plate_ocr_baseline import ADS, ALPHABETS, PROVINCES


def parse_ccpd_name(path, expected_length):
    parts = Path(path).stem.split('-')
    if len(parts) < 5:
        return None
    try:
        vertices = [tuple(map(int, item.split('&')))
                    for item in parts[3].split('_')]
        codes = [int(item) for item in parts[4].split('_')]
    except ValueError:
        return None
    if len(vertices) not in (4, 6) or len(codes) != expected_length:
        return None
    if (not 0 <= codes[0] < len(PROVINCES) - 1 or
            not 0 <= codes[1] < len(ALPHABETS) - 1 or
            any(not 0 <= item < len(ADS) - 1 for item in codes[2:])):
        return None
    # CCPD2020 Green adds two bounding-box points before the four plate
    # vertices. The final four retain the RB, LB, LT, RT order of CCPD2019.
    quad = np.asarray(vertices[-4:], dtype=np.float32)
    label = (PROVINCES[codes[0]] + ALPHABETS[codes[1]] +
             ''.join(ADS[item] for item in codes[2:]))
    return quad, label


def rectify(image, quad, width=256, height=64, margin=.02):
    # Source order is RB, LB, LT, RT; OpenCV target order is LT, RT, RB, LB.
    source = quad[[2, 3, 0, 1]].copy()
    center = source.mean(axis=0)
    source = center + (source - center) * (1.0 + margin)
    if abs(float(cv2.contourArea(source))) < 80:
        return None
    target = np.float32([[0, 0], [width - 1, 0],
                         [width - 1, height - 1], [0, height - 1]])
    transform = cv2.getPerspectiveTransform(source, target)
    return cv2.warpPerspective(image, transform, (width, height),
                               borderMode=cv2.BORDER_REPLICATE)


def select_base(root, anhui_limit, val_limit, seed):
    train_file = root / 'splits' / 'train.txt'
    val_file = root / 'splits' / 'val.txt'
    if not train_file.is_file() or not val_file.is_file():
        raise FileNotFoundError('CCPD2019 train.txt/val.txt not found')
    rng = random.Random(seed)
    anhui, others = [], []
    with train_file.open(encoding='utf-8') as handle:
        for line in handle:
            relative = line.strip().replace('\\', '/')
            parsed = parse_ccpd_name(relative, 7)
            if not relative.startswith('ccpd_base/') or parsed is None:
                continue
            (anhui if parsed[1][0] == '皖' else others).append(relative)
    official_train_count = len(anhui) + len(others)
    if anhui_limit > 0 and len(anhui) > anhui_limit:
        anhui = rng.sample(anhui, anhui_limit)
    with val_file.open(encoding='utf-8') as handle:
        valid = [line.strip().replace('\\', '/') for line in handle]
    valid = [name for name in valid if name.startswith('ccpd_base/')
             and parse_ccpd_name(name, 7) is not None]
    if val_limit > 0 and len(valid) > val_limit:
        valid = rng.sample(valid, val_limit)
    return sorted(anhui + others), sorted(valid), {
        'train_total_official': official_train_count,
        'train_anhui_selected': len(anhui),
        'train_other_selected': len(others),
    }


def select_green(root, train_limit, val_limit, seed):
    rng = random.Random(seed + 1)
    selected = {}
    for split, limit in (('train', train_limit), ('val', val_limit)):
        folder = root / 'ccpd_green' / split
        if not folder.is_dir():
            raise FileNotFoundError(folder)
        names = [path.name for path in folder.iterdir()
                 if path.is_file() and path.suffix.lower() in ('.jpg', '.jpeg')
                 and parse_ccpd_name(path.name, 8) is not None]
        if limit > 0 and len(names) > limit:
            names = rng.sample(names, limit)
        selected[split] = sorted(names)
    return selected


def export_group(source_root, names, source_kind, split, output,
                 manifest, counts, errors):
    category = 'base7' if source_kind == 'ccpd2019_base' else 'green8'
    dest_dir = output / category / split
    dest_dir.mkdir(parents=True, exist_ok=True)
    expected = 7 if category == 'base7' else 8
    for index, name in enumerate(names, 1):
        if category == 'base7':
            image_path = source_root.joinpath(*name.split('/'))
            source_relative = name
        else:
            image_path = source_root / 'ccpd_green' / split / name
            source_relative = 'ccpd_green/{}/{}'.format(split, name)
        parsed = parse_ccpd_name(name, expected)
        if parsed is None:
            errors['annotation'] += 1
            continue
        quad, label = parsed
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            errors['image_read'] += 1
            continue
        crop = rectify(image, quad)
        if crop is None:
            errors['degenerate_quad'] += 1
            continue
        identifier = hashlib.sha1((source_kind + '/' + source_relative).encode(
            'utf-8')).hexdigest()[:20]
        relative = '{}/{}/{}.jpg'.format(category, split, identifier)
        destination = output / relative
        if not cv2.imwrite(str(destination), crop,
                           [cv2.IMWRITE_JPEG_QUALITY, 92]):
            errors['image_write'] += 1
            continue
        manifest.append((relative, label, source_kind, split, source_relative))
        counts[(category, split)] += 1
        if index % 2000 == 0:
            print('{} {}: {}/{} source images'.format(category, split,
                                                     index, len(names)), flush=True)


def write_manifests(output, records, counts, errors, seed):
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'manifest.csv').open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.writer(handle)
        writer.writerow(('crop_path', 'label', 'dataset', 'split', 'source_image'))
        writer.writerows(records)
    for category in ('base7', 'green8'):
        for split in ('train', 'val'):
            with (output / '{}_{}.tsv'.format(category, split)).open(
                    'w', encoding='utf-8') as handle:
                for relative, label, _, row_split, _ in records:
                    if relative.startswith(category + '/') and row_split == split:
                        handle.write('{}\t{}\n'.format(relative, label))
    summary = {
        'seed': seed, 'crop_size': [256, 64],
        'counts': {'{}_{}'.format(*key): value for key, value in counts.items()},
        'errors': dict(errors),
        'train_provinces': dict(Counter(label[0] for _, label, _, split, _
                                        in records if split == 'train')),
        'sources': {
            'ccpd2019': 'https://github.com/detectRecog/CCPD',
            'ccpd2020_green': 'https://github.com/detectRecog/CCPD',
        },
    }
    (output / 'summary.json').write_text(json.dumps(
        summary, ensure_ascii=False, indent=2), encoding='utf-8')
    return summary


def main():
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ccpd2019', type=Path,
                        default=Path('F:/data/CCPD-Dataset/data/CCPD2019'))
    parser.add_argument('--ccpd2020', type=Path,
                        default=Path('F:/data/CCPD-Dataset/data/CCPD2020'))
    parser.add_argument('--output', type=Path,
                        default=Path('build/ocr_real_crops'))
    parser.add_argument('--anhui-train-limit', type=int, default=12000)
    parser.add_argument('--base-val-limit', type=int, default=2000)
    parser.add_argument('--green-train-limit', type=int, default=0,
                        help='0 means all official Green train images')
    parser.add_argument('--green-val-limit', type=int, default=0,
                        help='0 means all official Green val images')
    parser.add_argument('--seed', type=int, default=20261030)
    args = parser.parse_args()
    if min(args.anhui_train_limit, args.base_val_limit,
           args.green_train_limit, args.green_val_limit) < 0:
        parser.error('limits must be non-negative')
    base_train, base_val, selection = select_base(
        args.ccpd2019, args.anhui_train_limit, args.base_val_limit, args.seed)
    green = select_green(args.ccpd2020, args.green_train_limit,
                         args.green_val_limit, args.seed)
    print(json.dumps({'selection': selection,
                      'base_val': len(base_val),
                      'green_train': len(green['train']),
                      'green_val': len(green['val'])}, ensure_ascii=False),
          flush=True)
    manifest, counts, errors = [], Counter(), Counter()
    for root, names, source, split in (
            (args.ccpd2019, base_train, 'ccpd2019_base', 'train'),
            (args.ccpd2019, base_val, 'ccpd2019_base', 'val'),
            (args.ccpd2020, green['train'], 'ccpd2020_green', 'train'),
            (args.ccpd2020, green['val'], 'ccpd2020_green', 'val')):
        export_group(root, names, source, split, args.output,
                     manifest, counts, errors)
    print(json.dumps(write_manifests(args.output, manifest, counts,
                                     errors, args.seed), ensure_ascii=False,
                     indent=2))


if __name__ == '__main__':
    main()
