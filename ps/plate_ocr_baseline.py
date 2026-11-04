#!/usr/bin/env python3
"""Small CCPD OCR baseline: 7 fixed glyph slots, HOG, linear classifiers.

Train on CCPD train.txt, evaluate on val.txt. Filename labels are used only
for training and scoring, never for model input. The exported NPZ needs only
NumPy and OpenCV at inference time on the PYNQ board.
"""
import argparse
import csv
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np


PROVINCES = list('皖沪津渝冀晋蒙辽吉黑苏浙京闽赣鲁豫鄂湘粤桂琼川贵云藏陕甘青宁新警学O')
ALPHABETS = list('ABCDEFGHJKLMNPQRSTUVWXYZ') + ['O']
ADS = list('ABCDEFGHJKLMNPQRSTUVWXYZ0123456789') + ['O']
ALPHABET = (PROVINCES, ALPHABETS, ADS)
SLOTS = ((.02, .19), (.16, .33), (.31, .48), (.44, .61),
         (.57, .74), (.70, .87), (.83, 1.00))
PLATE_SIZE = (224, 64)

def glyph_features(glyph):
    """Portable 4x8-cell gradient histogram plus a small intensity image."""
    gx = cv2.Sobel(glyph, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(glyph, cv2.CV_32F, 0, 1, ksize=3)
    magnitude, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)
    bins = np.minimum((angle % 180 / 22.5).astype(np.int32), 7)
    cell_x = np.broadcast_to(np.arange(24, dtype=np.int32)[None, :] // 6,
                             (40, 24))
    cell_y = np.broadcast_to(np.arange(40, dtype=np.int32)[:, None] // 5,
                             (40, 24))
    index = (cell_y * 4 + cell_x) * 8 + bins
    histogram = np.bincount(index.reshape(-1), weights=magnitude.reshape(-1),
                            minlength=256).reshape(32, 8).astype(np.float32)
    histogram /= np.linalg.norm(histogram, axis=1, keepdims=True) + 1e-6
    intensity = cv2.resize(glyph, (12, 20), interpolation=cv2.INTER_AREA)
    return np.concatenate((histogram.reshape(-1),
                           intensity.reshape(-1).astype(np.float32) / 255.0))


def parse_ccpd(path):
    fields = Path(path).stem.split('-')
    if len(fields) < 7:
        return None
    try:
        corners = [tuple(map(int, point.split('&')))
                   for point in fields[3].split('_')]
        codes = tuple(map(int, fields[4].split('_')))
    except ValueError:
        return None
    if len(corners) != 4 or len(codes) != 7:
        return None
    if (not 0 <= codes[0] < len(PROVINCES) - 1 or
            not 0 <= codes[1] < len(ALPHABETS) - 1 or
            any(not 0 <= code < len(ADS) - 1 for code in codes[2:])):
        return None  # CCPD uses the final O entry for missing characters.
    return np.asarray(corners, np.float32), codes


def warp_ccpd(image, corners):
    """CCPD corner order is right-bottom, left-bottom, left-top, right-top."""
    source = corners[[2, 3, 0, 1]]
    width, height = PLATE_SIZE
    target = np.float32([[0, 0], [width - 1, 0],
                         [width - 1, height - 1], [0, height - 1]])
    matrix = cv2.getPerspectiveTransform(source, target)
    return cv2.warpPerspective(image, matrix, PLATE_SIZE,
                               flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REPLICATE)


def order_candidate_quad(box):
    """Order observed corners TL,TR,BR,BL without replacing a trapezoid."""
    points = np.asarray(box, dtype=np.float32)
    if points.shape != (4, 2) or not np.all(np.isfinite(points)):
        raise ValueError('Expected four finite plate corners')
    corners = cv2.convexHull(points).reshape(-1, 2)
    if len(corners) != 4 or cv2.contourArea(corners) < 1:
        raise ValueError('Plate corners must form a nondegenerate convex quadrilateral')
    center = corners.mean(axis=0)
    corners = corners[np.argsort(np.arctan2(corners[:, 1] - center[1],
                                           corners[:, 0] - center[0]))]
    lengths = np.linalg.norm(np.roll(corners, -1, axis=0) - corners, axis=1)
    pair = (0, 2) if lengths[0] + lengths[2] >= lengths[1] + lengths[3] else (1, 3)
    start = min(pair, key=lambda i: (corners[i, 1] + corners[(i + 1) % 4, 1]) / 2)
    source = np.roll(corners, -start, axis=0)
    if source[1, 0] < source[0, 0]:
        source = source[[1, 0, 3, 2]]
    return source


def warp_candidate(image, box, margin_x=0.0, margin_y=0.0):
    """Rectify all four observed corners, retaining real perspective geometry."""
    source = order_candidate_quad(box)
    if margin_x < 0 or margin_y < 0:
        raise ValueError('Crop margins must be nonnegative')
    horizontal = (source[1] - source[0]) * margin_x
    vertical = (source[3] - source[0]) * margin_y
    source = source + np.asarray([-horizontal - vertical,
                                  horizontal - vertical,
                                  horizontal + vertical,
                                 -horizontal + vertical], dtype=np.float32)
    width, height = PLATE_SIZE
    target = np.float32([[0, 0], [width - 1, 0],
                         [width - 1, height - 1], [0, height - 1]])
    return cv2.warpPerspective(image,
                               cv2.getPerspectiveTransform(source, target),
                               PLATE_SIZE, borderMode=cv2.BORDER_REPLICATE)


def letterbox(image, width=640, height=480):
    source_h, source_w = image.shape[:2]
    scale = min(width / source_w, height / source_h)
    resized_w = max(1, int(round(source_w * scale)))
    resized_h = max(1, int(round(source_h * scale)))
    resized = cv2.resize(image, (resized_w, resized_h),
                         interpolation=cv2.INTER_AREA)
    canvas = np.zeros((height, width, 3), np.uint8)
    left = (width - resized_w) // 2
    top = (height - resized_h) // 2
    canvas[top:top + resized_h, left:left + resized_w] = resized
    return canvas, scale, left, top


def quad_iou(first, second):
    a = np.asarray(first, np.float32)
    b = np.asarray(second, np.float32)
    area_a = abs(float(cv2.contourArea(a)))
    area_b = abs(float(cv2.contourArea(b)))
    intersection, _ = cv2.intersectConvexConvex(a, b)
    union = area_a + area_b - float(intersection)
    return float(intersection) / union if union > 0 else 0.0


def features(plate):
    if plate.shape[1::-1] != PLATE_SIZE:
        plate = cv2.resize(plate, PLATE_SIZE, interpolation=cv2.INTER_LINEAR)
    gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    width, height = PLATE_SIZE
    rows = []
    for left, right in SLOTS:
        x0 = int(round(left * width))
        x1 = min(width, int(round(right * width)))
        glyph = gray[int(.13 * height):int(.91 * height), x0:x1]
        glyph = cv2.resize(glyph, (24, 40), interpolation=cv2.INTER_AREA)
        rows.append(glyph_features(glyph))
    return np.stack(rows).astype(np.float32)


def read_split(root, split, limit, seed):
    split_path = root / 'splits' / (split + '.txt')
    if not split_path.is_file():
        raise FileNotFoundError(split_path)
    rng = random.Random(seed)
    selected = []
    count = 0
    with split_path.open(encoding='utf-8') as handle:
        for line in handle:
            relative = line.strip().replace('\\', '/')
            subset = 'ccpd_base' if split in ('train', 'val') else split
            if not relative.startswith(subset + '/'):
                continue
            if parse_ccpd(relative) is None:
                continue
            count += 1
            if limit <= 0:
                selected.append(relative)
            elif len(selected) < limit:
                selected.append(relative)
            else:
                index = rng.randrange(count)
                if index < limit:
                    selected[index] = relative
    return sorted(selected)


def load_samples(root, paths, source='gt', debug_dir=None, save_crops=0,
                 min_iou=0.0):
    if source == 'detector':
        from plate_locator_adaptive import find_candidates_adaptive
    xs, ys, used, ious = [], [], [], []
    missed = 0
    for relative in paths:
        annotation = parse_ccpd(relative)
        if annotation is None:
            continue
        corners, codes = annotation
        image_path = root.joinpath(*relative.split('/'))
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        if source in ('gt_live', 'detector'):
            camera_frame, scale, left, top = letterbox(image)
        if source == 'detector':
            proposals = find_candidates_adaptive(camera_frame)
            if not proposals:
                missed += 1
                continue
            plate = warp_candidate(camera_frame, proposals[0][1])
            mapped = corners * scale + np.float32([left, top])
            iou = quad_iou(mapped, proposals[0][1])
            if iou < min_iou:
                missed += 1
                continue
        elif source == 'gt_live':
            mapped = corners * scale + np.float32([left, top])
            plate = warp_ccpd(camera_frame, mapped)
            iou = 1.0
        else:
            plate = warp_ccpd(image, corners)
            iou = 1.0
        if debug_dir is not None and len(used) < save_crops:
            debug_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(debug_dir / '{:04d}.png'.format(len(used))), plate)
        xs.append(features(plate))
        ys.append(codes)
        used.append(relative)
        ious.append(iou)
    if not xs:
        raise RuntimeError('No valid CCPD plate images found')
    return np.stack(xs), np.asarray(ys, np.int16), used, missed, np.asarray(ious)


def train_model(xs, ys):
    from sklearn.linear_model import SGDClassifier
    payload = {'format_version': np.int16(1)}
    for name, data, labels in (
            ('province', xs[:, 0, :], ys[:, 0]),
            ('letter', xs[:, 1, :], ys[:, 1]),
            ('alnum', xs[:, 2:, :].reshape(-1, xs.shape[-1]),
             ys[:, 2:].reshape(-1))):
        classes = np.unique(labels)
        if len(classes) == 1:
            payload[name + '_weights'] = np.zeros((1, data.shape[1]), np.float32)
            payload[name + '_bias'] = np.zeros(1, np.float32)
        else:
            model = SGDClassifier(loss='hinge', alpha=1e-4, max_iter=30,
                                  tol=1e-3, random_state=20261030)
            model.fit(data, labels)
            payload[name + '_weights'] = model.coef_.astype(np.float32)
            payload[name + '_bias'] = model.intercept_.astype(np.float32)
        payload[name + '_classes'] = classes.astype(np.int16)
    return payload


def predict(xs, model):
    result = np.empty((len(xs), 7), np.int16)
    for name, positions in (('province', [0]), ('letter', [1]),
                            ('alnum', [2, 3, 4, 5, 6])):
        weights = model[name + '_weights']
        bias = model[name + '_bias']
        classes = model[name + '_classes']
        for position in positions:
            if len(classes) == 1:
                result[:, position] = classes[0]
            else:
                scores = xs[:, position, :] @ weights.T + bias
                if len(classes) == 2:
                    result[:, position] = classes[(scores[:, 0] > 0).astype(int)]
                else:
                    result[:, position] = classes[np.argmax(scores, axis=1)]
    return result


def plate_text(codes):
    return (PROVINCES[int(codes[0])] + ALPHABETS[int(codes[1])] +
            ''.join(ADS[int(code)] for code in codes[2:]))


def load_model(path):
    with np.load(str(path), allow_pickle=False) as loaded:
        if int(loaded['format_version']) != 1:
            raise ValueError('Unsupported OCR model format')
        return {key: loaded[key] for key in loaded.files}


def recognize_candidate(image, box, model):
    plate = warp_candidate(image, box)
    codes = predict(features(plate)[None, :, :], model)[0]
    return plate_text(codes), plate


def evaluate(xs, ys, paths, missed, ious, model, output):
    predicted = predict(xs, model)
    exact = np.all(predicted == ys, axis=1)
    position_accuracy = np.mean(predicted == ys, axis=0)
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'predictions.csv').open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.writer(handle)
        writer.writerow(['path', 'truth', 'prediction', 'exact', 'detector_quad_iou'])
        for path, truth, guess, ok, iou in zip(paths, ys, predicted, exact, ious):
            writer.writerow([path, plate_text(truth), plate_text(guess),
                             int(ok), '{:.4f}'.format(iou)])
    matched = ious > .7
    summary = {'evaluated': len(xs), 'detector_misses': missed,
               'exact_rate': float(np.mean(exact)),
               'exact_rate_all': float(np.sum(exact) / (len(xs) + missed)),
               'iou_gt_070_count': int(np.sum(matched)),
               'exact_rate_iou_gt_070': (float(np.mean(exact[matched]))
                                         if np.any(matched) else None),
               'position_accuracy': [float(v) for v in position_accuracy]}
    (output / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                                         encoding='utf-8')
    return summary


def main():
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('train', 'eval', 'infer'))
    parser.add_argument('--root', type=Path,
                        default=Path('F:/data/CCPD-Dataset/data/CCPD2019'))
    parser.add_argument('--model', type=Path, default=Path('build/ocr_baseline/model.npz'))
    parser.add_argument('--output', type=Path, default=Path('build/ocr_baseline'))
    parser.add_argument('--limit', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=20261030)
    parser.add_argument('--source', choices=('gt', 'gt_live', 'detector'), default='gt')
    parser.add_argument('--save-crops', type=int, default=0)
    parser.add_argument('--train-min-iou', type=float, default=.7)
    parser.add_argument('--split', type=str,
                        help='CCPD split name for evaluation; defaults to val')
    parser.add_argument('--image', type=Path, help='Input image for infer mode')
    args = parser.parse_args()
    if args.mode == 'infer':
        if args.image is None:
            parser.error('infer requires --image')
        from plate_locator_adaptive import find_candidates_adaptive
        image = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError('Cannot read {}'.format(args.image))
        frame, _, _, _ = letterbox(image)
        candidates = find_candidates_adaptive(frame)
        if not candidates:
            print(json.dumps({'plate': None, 'reason': 'no_candidate'},
                             ensure_ascii=False))
            return
        text, plate = recognize_candidate(frame, candidates[0][1],
                                          load_model(args.model))
        args.output.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(args.output / 'infer_plate.png'), plate)
        print(json.dumps({'plate': text, 'candidate_score': candidates[0][0],
                          'crop': str(args.output / 'infer_plate.png')},
                         ensure_ascii=False))
        return
    if args.mode == 'train' and args.split not in (None, 'train'):
        parser.error('train mode can only read train split')
    split = 'train' if args.mode == 'train' else (args.split or 'val')
    paths = read_split(args.root, split, args.limit, args.seed)
    xs, ys, used, missed, ious = load_samples(
        args.root, paths, source=args.source,
        debug_dir=args.output / ('crops_' + args.source),
        save_crops=args.save_crops,
        min_iou=(args.train_min_iou if args.mode == 'train' else 0.0))
    if args.mode == 'train':
        model = train_model(xs, ys)
        args.model.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(str(args.model), **model)
        print('trained images={} excluded={} model={}'.format(
            len(xs), missed, args.model))
    else:
        model = load_model(args.model)
        print(json.dumps(evaluate(xs, ys, used, missed, ious,
                                  model, args.output),
                         ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
