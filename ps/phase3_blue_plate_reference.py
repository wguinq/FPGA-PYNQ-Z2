#!/usr/bin/env python3
"""Experimental blue plate candidate detection on one saved frame (CPU)."""
import argparse
from pathlib import Path

import cv2
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('color', help='Original color image')
    parser.add_argument('--sobel', help='Sobel output from the same frame (optional)')
    parser.add_argument('--output', default='plate_reference', help='Output filename prefix')
    parser.add_argument('--mask-mode', choices=('hsv', 'rgb'), default='hsv',
                        help='hsv: validated prototype; rgb: exact Phase-3A RTL comparison rule')
    args = parser.parse_args()
    color = cv2.imread(args.color, cv2.IMREAD_COLOR)
    if color is None:
        raise SystemExit('无法读取彩色图: ' + args.color)
    height, width = color.shape[:2]
    edge = None
    if args.sobel:
        edge = cv2.imread(args.sobel, cv2.IMREAD_GRAYSCALE)
        if edge is None or edge.shape != (height, width):
            raise SystemExit('Sobel 图读取失败或尺寸与彩色图不一致')
    if args.mask_mode == 'hsv':
        hsv = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array([95, 110, 80], np.uint8), np.array([135, 255, 255], np.uint8))
    else:
        b = color[:, :, 0].astype(np.int16)
        g = color[:, :, 1].astype(np.int16)
        r = color[:, :, 2].astype(np.int16)
        mask = ((b >= 80) & (b >= r + 20) & (b >= g + 10)).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (25, 9)))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        (_, _), (rw, rh), angle = cv2.minAreaRect(contour)
        long_side, short_side = max(rw, rh), min(rw, rh)
        if short_side < 1:
            continue
        ratio = long_side / short_side
        if long_side < 45 or short_side < 10 or not 2.0 <= ratio <= 6.0:
            continue
        fill = cv2.countNonZero(mask[y:y+h, x:x+w]) / (w*h)
        if fill < 0.25:
            continue
        density = None if edge is None else float(np.mean(edge[y:y+h, x:x+w] >= 80))
        # A colored rectangle alone is not enough: plate characters should also
        # produce a useful amount of edge energy.
        if density is not None and density < 0.30:
            continue
        score = long_side * short_side * fill * (1.0 if density is None else density)
        box = cv2.boxPoints(cv2.minAreaRect(contour)).astype(np.int32)
        candidates.append((score, x, y, w, h, fill, density, ratio, angle, box))
    candidates.sort(reverse=True)
    drawn = color.copy()
    for i, (_, x, y, w, h, fill, density, ratio, angle, box) in enumerate(candidates):
        cv2.polylines(drawn, [box], True, (0, 255, 0), 2)
        cv2.putText(drawn, str(i+1), (x, max(15, y-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        print(f'候选 {i+1}: x={x}, y={y}, 外接宽={w}, 外接高={h}, 旋转长宽比={ratio:.2f}, 角度={angle:.1f}, 蓝色占比={fill:.3f}' +
              ('' if density is None else f', Sobel亮边比例={density:.3f}'))
    if not candidates:
        print('未找到候选区域；请检查颜色、尺寸和阈值。')
    prefix = str(Path(args.output))
    for name, image in [('_mask.png', mask), ('_boxes.png', drawn)]:
        if not cv2.imwrite(prefix + name, image):
            raise SystemExit('无法写入 ' + prefix + name)
    if candidates:
        _, x, y, w, h, _, _, _, _, _ = candidates[0]
        padx, pady = max(4, w // 25), max(4, h // 8)
        roi = color[max(0, y-pady):min(height, y+h+pady), max(0, x-padx):min(width, x+w+padx)]
        if not cv2.imwrite(prefix + '_roi.png', roi):
            raise SystemExit('无法写入 ROI 图片')
    print('输出前缀:', prefix)


if __name__ == '__main__':
    main()
