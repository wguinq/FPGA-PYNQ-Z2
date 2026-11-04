#!/usr/bin/env python3
"""Train a small, variable-length plate recognizer on real CCPD plate crops.

The model takes one 128x32 grayscale plate and produces per-column character
probabilities. CTC handles both seven- and eight-character plates without
fixed glyph slots. Labels are read only from train/val TSV files.
"""
import argparse
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from plate_ocr_baseline import ADS, ALPHABETS, PROVINCES


VOCAB = ''.join(dict.fromkeys(PROVINCES[:-1] + ALPHABETS[:-1] + ADS[:-1]))
TO_ID = {char: index + 1 for index, char in enumerate(VOCAB)}
HEIGHT, WIDTH = 32, 128


class Residual1D(nn.Module):
    def __init__(self, channels, dilation):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, 5, padding=2 * dilation,
                      dilation=dilation, groups=channels, bias=False),
            nn.Conv1d(channels, channels, 1, bias=False),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return x + self.net(x)


class TinyPlateCTC(nn.Module):
    def __init__(self, classes=len(VOCAB) + 1):
        super().__init__()
        layers = []
        channels = 1
        for output, stride in ((32, (2, 2)), (64, (2, 2)),
                               (96, (2, 1)), (128, (2, 1))):
            layers += [nn.Conv2d(channels, output, 3, stride=stride,
                                 padding=1, bias=False),
                       nn.BatchNorm2d(output), nn.ReLU(inplace=True)]
            channels = output
        self.features = nn.Sequential(*layers)
        self.sequence = nn.Sequential(Residual1D(128, 1),
                                      Residual1D(128, 2),
                                      Residual1D(128, 4))
        self.classifier = nn.Conv1d(128, classes, 1)

    def forward(self, images):
        # Keep the pooled axis until Squeeze. OpenCV 4.2 cannot import an
        # ONNX ReduceMean node with keepdims=0.
        x = self.features(images).mean(dim=2, keepdim=True).squeeze(2)
        x = self.sequence(x)
        return self.classifier(x).permute(0, 2, 1)  # batch, time, class


class PlateDataset(Dataset):
    def __init__(self, root, groups, augment=False, limit=0, seed=20261030,
                 exclude_labels=None):
        self.items = []
        self.images = []
        self.augment = augment
        self.low_resolution = False
        self.rng = np.random.default_rng(seed)
        for group in groups:
            split = 'train' if augment else 'val'
            path = root / '{}_{}.tsv'.format(group, split)
            if not path.is_file():
                raise FileNotFoundError(path)
            with path.open(encoding='utf-8') as handle:
                rows = [line.rstrip('\n').split('\t', 1) for line in handle]
            if exclude_labels is not None:
                blocked = exclude_labels.get(group, set())
                rows = [row for row in rows if row[1] not in blocked]
            if limit > 0 and len(rows) > limit:
                rows = random.Random(seed + len(group)).sample(rows, limit)
            for relative, label in rows:
                if not label or any(char not in TO_ID for char in label):
                    raise ValueError('Unsupported label {} in {}'.format(label, path))
                image = cv2.imread(str(root / relative), cv2.IMREAD_GRAYSCALE)
                if image is None:
                    raise FileNotFoundError(root / relative)
                image = cv2.resize(image, (WIDTH, HEIGHT),
                                   interpolation=cv2.INTER_AREA)
                self.images.append(image)
                self.items.append((label, group))
        self.images = np.stack(self.images)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        image = self.images[index]
        label, group = self.items[index]
        if self.augment:
            image = augment_plate(image, self.rng)
            if self.low_resolution and self.rng.random() < .5:
                small_w = int(self.rng.integers(48, 113))
                small_h = int(self.rng.integers(12, 29))
                image = cv2.resize(cv2.resize(image, (small_w, small_h),
                                   interpolation=cv2.INTER_AREA), (WIDTH, HEIGHT),
                                   interpolation=cv2.INTER_LINEAR)
        image = torch.from_numpy(image.copy()).float().unsqueeze(0) / 255.0
        return image, torch.tensor([TO_ID[char] for char in label],
                                   dtype=torch.long), label, group


def augment_plate(image, rng):
    transform = np.float32([[rng.uniform(.91, 1.06), 0, rng.uniform(-3, 3)],
                            [0, rng.uniform(.91, 1.05), rng.uniform(-2, 2)]])
    image = cv2.warpAffine(image, transform, (WIDTH, HEIGHT),
                           borderMode=cv2.BORDER_REPLICATE)
    if rng.random() < .25:
        image = cv2.GaussianBlur(image, (3, 3), rng.uniform(.3, 1.0))
    alpha = rng.uniform(.8, 1.2)
    beta = rng.uniform(-18, 18)
    if rng.random() < .2:
        noise = rng.normal(0, rng.uniform(1, 8), image.shape)
    else:
        noise = 0
    return np.clip(image.astype(np.float32) * alpha + beta + noise,
                   0, 255).astype(np.uint8)


def collate(batch):
    images, targets, labels, groups = zip(*batch)
    lengths = torch.tensor([len(target) for target in targets], dtype=torch.long)
    return torch.stack(images), torch.cat(targets), lengths, labels, groups


def greedy_decode(logits):
    ids = logits.argmax(dim=-1).cpu().numpy()
    results = []
    for sequence in ids:
        kept = []
        previous = 0
        for index in sequence:
            index = int(index)
            if index and index != previous:
                kept.append(VOCAB[index - 1])
            previous = index
        results.append(''.join(kept))
    return results


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    counts = defaultdict(int)
    correct = defaultdict(int)
    examples = []
    province_counts, province_first, province_exact = defaultdict(int), defaultdict(int), defaultdict(int)
    for images, _, _, labels, groups in loader:
        guesses = greedy_decode(model(images.to(device)))
        for truth, guess, group in zip(labels, guesses, groups):
            counts[group] += 1
            correct[group] += int(truth == guess)
            province_counts[truth[0]] += 1
            province_first[truth[0]] += int(bool(guess) and guess[0] == truth[0])
            province_exact[truth[0]] += int(guess == truth)
            if len(examples) < 12 and truth != guess:
                examples.append({'truth': truth, 'prediction': guess,
                                 'group': group})
    result = {'overall': sum(correct.values()) / max(1, sum(counts.values()))}
    for group in sorted(counts):
        result[group] = {'exact': correct[group], 'count': counts[group],
                         'exact_rate': correct[group] / counts[group]}
    result['error_examples'] = examples
    result['provinces'] = {p: {'count': province_counts[p], 'first_correct': province_first[p],
                              'exact': province_exact[p]} for p in province_counts}
    return result


def train(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    device = torch.device('cuda' if torch.cuda.is_available() and
                          not args.cpu else 'cpu')
    excluded = None
    if args.plate_disjoint:
        excluded = {}
        for group in ('base7', 'green8'):
            with (args.data / '{}_val.tsv'.format(group)).open(
                    encoding='utf-8') as handle:
                excluded[group] = {line.rstrip('\n').split('\t', 1)[1]
                                   for line in handle}
        excluded['detector7'] = excluded['base7']
    train_groups = ('base7', 'green8', 'detector7') if args.detector_train else (
        'base7', 'green8')
    if args.clpd:
        train_groups += ('clpd',)
        if excluded is not None:
            excluded['clpd'] = set()
    train_set = PlateDataset(args.data, train_groups, True,
                             args.max_train_per_group, args.seed,
                             exclude_labels=excluded)
    val_set = PlateDataset(args.data, ('base7', 'green8') + (('clpd',) if args.clpd else ()), False,
                           args.max_val_per_group, args.seed)
    train_set.low_resolution = args.low_resolution
    sampler = None
    if args.balance_provinces:
        frequencies = defaultdict(int)
        for label, _ in train_set.items:
            frequencies[label[0]] += 1
        weights = [min(5.0, (max(frequencies.values()) / frequencies[label[0]]) ** .5)
                   for label, _ in train_set.items]
        sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
    train_loader = DataLoader(train_set, batch_size=args.batch_size,
                              shuffle=sampler is None, sampler=sampler,
                              num_workers=0, collate_fn=collate)
    val_loader = DataLoader(val_set, batch_size=args.batch_size,
                            shuffle=False, num_workers=0, collate_fn=collate)
    model = TinyPlateCTC().to(device)
    if args.init_checkpoint is not None:
        initial = torch.load(args.init_checkpoint, map_location=device,
                             weights_only=False)
        if initial['vocab'] != VOCAB:
            raise ValueError('Initial checkpoint vocabulary differs from code')
        model.load_state_dict(initial['state_dict'])
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * .05)
    args.output.mkdir(parents=True, exist_ok=True)
    best, history = -1.0, []
    print(json.dumps({'device': str(device), 'cuda_name':
                      torch.cuda.get_device_name(0) if device.type == 'cuda'
                      else None, 'train': len(train_set), 'val': len(val_set),
                      'parameters': sum(p.numel() for p in model.parameters()),
                      'plate_disjoint': args.plate_disjoint,
                      'detector_train': args.detector_train,
                      'clpd': args.clpd,
                      'balance_provinces': args.balance_provinces,
                      'low_resolution': args.low_resolution,
                      'init_checkpoint': str(args.init_checkpoint) if
                      args.init_checkpoint else None,
                      'vocab_size': len(VOCAB), 'time_steps':
                      model(torch.zeros(1, 1, HEIGHT, WIDTH,
                                        device=device)).shape[1]},
                     ensure_ascii=False), flush=True)
    for epoch in range(1, args.epochs + 1):
        start = time.perf_counter()
        model.train()
        total_loss, seen = 0.0, 0
        for images, targets, target_lengths, _, _ in train_loader:
            images = images.to(device)
            targets = targets.to(device)
            logits = model(images)
            input_lengths = torch.full((len(images),), logits.shape[1],
                                       dtype=torch.long, device=device)
            loss = criterion(logits.log_softmax(-1).transpose(0, 1),
                             targets, input_lengths, target_lengths)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += float(loss.detach()) * len(images)
            seen += len(images)
        scheduler.step()
        metrics = validate(model, val_loader, device)
        row = {'epoch': epoch, 'train_loss': total_loss / seen,
               'val': metrics, 'seconds': round(time.perf_counter() - start, 1)}
        history.append(row)
        print(json.dumps({'epoch': epoch, 'loss': round(row['train_loss'], 4),
                          'val_base7': metrics['base7']['exact_rate'],
                          'val_green8': metrics['green8']['exact_rate'],
                          'seconds': row['seconds']}, ensure_ascii=False),
              flush=True)
        (args.output / 'history.json').write_text(
            json.dumps(history, ensure_ascii=False, indent=2), encoding='utf-8')
        checkpoint = {'state_dict': model.state_dict(), 'vocab': VOCAB,
                      'input_size': [WIDTH, HEIGHT], 'epoch': epoch,
                      'metrics': metrics, 'parameters':
                      sum(p.numel() for p in model.parameters()),
                      'plate_disjoint': args.plate_disjoint,
                      'detector_train': args.detector_train,
                      'clpd': args.clpd,
                      'balance_provinces': args.balance_provinces,
                      'low_resolution': args.low_resolution}
        torch.save(checkpoint, args.output / 'last.pt')
        if metrics['overall'] > best:
            best = metrics['overall']
            torch.save(checkpoint, args.output / 'best.pt')
            (args.output / 'best_metrics.json').write_text(
                json.dumps(metrics, ensure_ascii=False, indent=2),
                encoding='utf-8')
    print('best_overall_exact={:.4f} checkpoint={}'.format(
        best, args.output / 'best.pt'))


def evaluate(args):
    torch.set_num_threads(min(4, torch.get_num_threads()))
    device = torch.device('cuda' if torch.cuda.is_available() and
                          not args.cpu else 'cpu')
    checkpoint = torch.load(args.output / 'best.pt', map_location=device,
                            weights_only=False)
    if checkpoint['vocab'] != VOCAB:
        raise ValueError('Checkpoint vocabulary differs from code')
    model = TinyPlateCTC().to(device)
    model.load_state_dict(checkpoint['state_dict'])
    val_set = PlateDataset(args.data, ('base7', 'green8'), False,
                           args.max_val_per_group, args.seed)
    loader = DataLoader(val_set, batch_size=args.batch_size,
                        num_workers=0, collate_fn=collate)
    print(json.dumps(validate(model, loader, device), ensure_ascii=False,
                     indent=2))


def main():
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('train', 'eval'))
    parser.add_argument('--data', type=Path,
                        default=Path('build/ocr_real_crops'))
    parser.add_argument('--output', type=Path,
                        default=Path('build/ocr_tiny_ctc'))
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=.001)
    parser.add_argument('--max-train-per-group', type=int, default=0)
    parser.add_argument('--max-val-per-group', type=int, default=0)
    parser.add_argument('--seed', type=int, default=20261030)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--balance-provinces', action='store_true')
    parser.add_argument('--low-resolution', action='store_true')
    parser.add_argument('--clpd', action='store_true', help='Include prepared CLPD train and validation splits')
    parser.add_argument('--plate-disjoint', action='store_true',
                        help='Exclude training plates whose full text appears in val')
    parser.add_argument('--detector-train', action='store_true',
                        help='Also use detector7_train.tsv crops from actual proposals')
    parser.add_argument('--init-checkpoint', type=Path,
                        help='Initialize weights from a previous checkpoint')
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        parser.error('epochs and batch size must be positive')
    if args.mode == 'train':
        train(args)
    else:
        evaluate(args)


if __name__ == '__main__':
    main()
