#!/usr/bin/env python3
"""UVC camera -> custom Phase-3 overlay HDMI OUT, with optional async detection.

The HDMI scanout is an independent AXI VDMA in the custom bitstream. This
script only supplies its DDR framebuffers; no PYNQ base overlay is loaded.
"""
import argparse
import csv
import io
from contextlib import redirect_stdout
import multiprocessing as mp
import os
from pathlib import Path
from queue import Empty, Full, Queue
from threading import Thread, Condition, Event
import sys
import time

import cv2
import numpy as np


WIDTH, HEIGHT = 640, 480


def ensure_hdmi_reference_clock(clocks=None):
    """Set the custom overlay's 100 MHz reference using the live PS PLL.

    HWH divider values assume the Vivado PS PLL configuration, which loading
    a bitstream does not install. On the board a 4*4 divider produced 62.5 MHz
    and hence 15.75 MHz pixels. PYNQ's frequency setter uses the live PLL.
    Call after Overlay downloads (and applies HWH dividers), before DMA starts.
    """
    if clocks is None:
        from pynq import Clocks
        clocks = Clocks
    before = float(clocks.fclk0_mhz)
    if not np.isfinite(before) or abs(before - 100.0) > .5:
        clocks.fclk0_mhz = 100.0
        # Allow the downstream MMCM to relock and release its pixel reset.
        time.sleep(.1)
    actual = float(clocks.fclk0_mhz)
    if not np.isfinite(actual) or abs(actual - 100.0) > .5:
        raise RuntimeError('HDMI requires FCLK0=100 MHz; actual {:.3f} MHz'.format(actual))
    print('HDMI reference FCLK0: {:.3f} -> {:.3f} MHz (required 100.000)'.format(
        before, actual), flush=True)
    return actual


class PlateBanner:
    """Render cached Unicode glyphs without requiring board-side font packages."""

    def __init__(self):
        with np.load(str(Path(__file__).with_name('plate_glyphs.npz')),
                     allow_pickle=False) as atlas:
            self.glyphs = dict(zip(str(atlas['chars'].item()), atlas['masks']))
        self.cache = {}

    def draw(self, frame, text, origin=(8, 8)):
        if text not in self.cache:
            if len(self.cache) >= 16:
                self.cache.clear()
            banner = np.zeros((44, 16 + 32 * len(text), 3), np.uint8)
            for index, char in enumerate(text):
                mask = self.glyphs.get(char)
                if mask is not None:
                    banner[6:38, 8 + index * 32:40 + index * 32, 1] = mask
            self.cache[text] = banner
        banner = self.cache[text]
        x, y = origin
        h = min(banner.shape[0], frame.shape[0] - y)
        w = min(banner.shape[1], frame.shape[1] - x)
        if h > 0 and w > 0:
            frame[y:y + h, x:x + w] = banner[:h, :w]


FRAME_BYTES = WIDTH * 3
VDMA_CONTROL = 0x00
VDMA_STATUS = 0x04
VDMA_PARK = 0x28
VDMA_VSIZE = 0x50
VDMA_HSIZE = 0x54
VDMA_STRIDE = 0x58
VDMA_FRAME0 = 0x5C


class HdmiScanout:
    """Run the MM2S VDMA in park mode without depending on a UIO interrupt."""

    def __init__(self, overlay, pixel_order='GBR'):
        from pynq import MMIO, allocate

        ensure_hdmi_reference_clock()
        self.pixel_order = pixel_order
        self.channel_map = [value for output, color in enumerate(pixel_order)
                            for value in ('BGR'.index(color), output)]
        info = overlay.ip_dict['axi_vdma_hdmi']
        self.mmio = MMIO(info['phys_addr'], info['addr_range'])
        self.frames = [allocate(shape=(HEIGHT, WIDTH, 3), dtype=np.uint8,
                                cacheable=True) for _ in range(3)]
        self.requested = 0
        self.mmio.write(VDMA_CONTROL, 0x4)
        deadline = time.monotonic() + 2.0
        while self.mmio.read(VDMA_CONTROL) & 0x4:
            if time.monotonic() > deadline:
                raise RuntimeError('HDMI VDMA reset timed out')
            time.sleep(0.002)
        self.mmio.write(VDMA_CONTROL, 0x1)  # RS=1, Circular_Park=0 (park mode)
        for index, frame in enumerate(self.frames):
            frame.fill(0)
            frame.flush()
            address = int(frame.physical_address)
            if address > 0xFFFFFFFF:
                raise RuntimeError('Frame buffer is outside the Zynq 32-bit AXI address range')
            self.mmio.write(VDMA_FRAME0 + index * 4, address)
        self.mmio.write(VDMA_PARK, 0)
        self.mmio.write(VDMA_STRIDE, FRAME_BYTES)
        self.mmio.write(VDMA_HSIZE, FRAME_BYTES)
        self.mmio.write(VDMA_VSIZE, HEIGHT)  # VSIZE must be written last.
        deadline = time.monotonic() + 2.0
        while self.mmio.read(VDMA_STATUS) & 0x1:
            if time.monotonic() > deadline:
                raise RuntimeError('HDMI VDMA did not start: status=0x{:08x}'.format(
                    self.mmio.read(VDMA_STATUS)))
            time.sleep(0.002)

    def show_bgr(self, bgr):
        status = self.mmio.read(VDMA_STATUS)
        if status & 0x71:
            raise RuntimeError('HDMI VDMA stopped or failed: status=0x{:08x}'.format(status))
        park = self.mmio.read(VDMA_PARK)
        active = (park >> 16) & 0x1f
        # The previous request has not reached a frame boundary yet. Never
        # overwrite a buffer that the VDMA may already have selected.
        if active != self.requested:
            return False
        next_index = (self.requested + 1) % 3
        # RGB2DVI consumes VDMA bytes as G, B, R (bits 7:0, 15:8, 23:16).
        # Keep OpenCV frames in BGR and reorder only when copying to scanout.
        if self.pixel_order == 'RGB':
            cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB, dst=self.frames[next_index])
        elif self.pixel_order == 'BGR':
            np.copyto(self.frames[next_index], bgr)
        else:
            cv2.mixChannels([bgr], [self.frames[next_index]], self.channel_map)
        self.frames[next_index].flush()
        self.mmio.write(VDMA_PARK, next_index)
        self.requested = next_index
        return True

    def close(self):
        self.mmio.write(VDMA_CONTROL, 0)
        for frame in self.frames:
            frame.freebuffer()


def locate_with_diagnostics(bgr, mask, raw_mask, debug_dir, frame, plate_color='blue'):
    """Keep source, mask and reject reasons from the same worker iteration."""
    from plate_locator_adaptive import find_candidates_adaptive
    if not debug_dir:
        return find_candidates_adaptive(bgr, proposal_mask=mask, plate_color=plate_color)
    folder = Path(debug_dir) / 'matched'
    folder.mkdir(parents=True, exist_ok=True)
    prefix = folder / 'frame_{:06d}'.format(frame)
    log = io.StringIO()
    with redirect_stdout(log):
        candidates = find_candidates_adaptive(bgr, debug=True, proposal_mask=mask, plate_color=plate_color)
    (Path(str(prefix) + '_reasons.txt')).write_text(
        'frame={} candidates={}\n'.format(frame, len(candidates)) + log.getvalue(),
        encoding='utf-8')
    images = [('camera.png', bgr)]
    if plate_color in ('green', 'auto'):
        green_raw = cv2.inRange(cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV),
                                np.uint8([35, 35, 45]), np.uint8([95, 255, 255]))
        images.append(('green_reference_raw.png', green_raw))
    if plate_color == 'auto':
        blue_raw = cv2.inRange(cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV),
                               np.uint8([95, 70, 45]), np.uint8([135, 255, 255]))
        images.append(('blue_reference_raw.png', blue_raw))
    if raw_mask is not None:
        images.append(('pl_raw.png', raw_mask))
    if mask is not None:
        images.append(('pl_aligned.png', mask))
    boxes = bgr.copy()
    for rank, candidate in enumerate(candidates[:3], 1):
        color = candidate_color(candidate, plate_color)
        ink = (255, 160, 0) if color == 'blue' else (0, 255, 0)
        cv2.polylines(boxes, [candidate[1].astype(np.int32)], True, ink, 2)
        x, y, _, _ = candidate[2]
        cv2.putText(boxes, '{} {}'.format(rank, color), (max(0, x), max(18, y - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, .5, ink, 1)
    images.append(('boxes.png', boxes))
    for suffix, data in images:
        if not cv2.imwrite(str(prefix) + '_' + suffix, data):
            raise RuntimeError('Cannot save diagnostic frame {}'.format(prefix))
    return candidates


def candidate_color(candidate, fallback='blue'):
    """Auto candidates append a color; keep legacy six-field candidates valid."""
    if len(candidate) > 6 and candidate[6] in ('blue', 'green'):
        return candidate[6]
    return fallback if fallback in ('blue', 'green') else 'unknown'


def recognize_candidates(bgr, candidates, reader, warp, margin_x, margin_y,
                         max_plates, fallback_color, debug_dir=None, frame=0):
    """OCR a bounded number of same-frame candidates, retaining their ranks."""
    predictions = []
    for rank, candidate in enumerate(candidates[:max_plates], 1):
        color = candidate_color(candidate, fallback_color)
        crop = warp(bgr, candidate[1], margin_x, margin_y)
        started = time.perf_counter()
        text = reader.recognize_crop(crop)
        seconds = time.perf_counter() - started
        predictions.append({'rank': rank, 'color': color, 'text': text,
                            'ocr_seconds': seconds})
        if debug_dir:
            prefix = Path(debug_dir) / 'frame_{:06d}'.format(frame)
            filename = '{}_rank{:02d}_{}_crop.jpg'.format(prefix, rank, color)
            if not cv2.imwrite(filename, crop):
                raise RuntimeError('Cannot save OCR crop {}'.format(filename))
            if rank == 1:
                # Preserve the old first-candidate filename for existing tools.
                cv2.imwrite(str(prefix) + '_crop.jpg', crop)
                cv2.imwrite(str(prefix) + '_camera.jpg', bgr)
    return predictions


def packet_predictions(item):
    """Read current packets and older single-plate result packets."""
    if len(item) > 7:
        return item[7]
    if item[5] is None:
        return []
    color = candidate_color(item[4][0]) if item[4] else 'unknown'
    return [{'rank': 1, 'color': color, 'text': item[5], 'ocr_seconds': item[6]}]


def result_log_rows(item):
    """Keep the original CSV columns first; one row per OCR candidate."""
    predictions = packet_predictions(item)
    if not predictions:
        predictions = [{'rank': 1 if item[4] else '', 'color': '',
                        'text': '', 'ocr_seconds': 0.0}]
    rows = []
    for prediction in predictions:
        rank = prediction['rank']
        candidate = item[4][rank - 1] if rank else None
        if candidate is None:
            x = y = w = h = score = ''
        else:
            x, y, w, h = candidate[2]
            score = candidate[0]
        rows.append((item[1], prediction['text'], x, y, w, h, score,
                     round(1000 * item[3], 2),
                     round(1000 * prediction['ocr_seconds'], 2),
                     rank, prediction['color']))
    return rows


def draw_detection_result(bgr, item, banner):
    """Show rank/color on boxes and the same rank next to each OCR string."""
    for rank, candidate in enumerate(item[4], 1):
        color = candidate_color(candidate, 'unknown')
        ink = (255, 160, 0) if color == 'blue' else (0, 255, 0)
        box = candidate[1].astype(np.int32).copy()
        box[:, 0] = np.clip(box[:, 0], 0, WIDTH - 1)
        box[:, 1] = np.clip(box[:, 1], 0, HEIGHT - 1)
        cv2.polylines(bgr, [box], True, ink, 2)
        x, y, _, _ = candidate[2]
        cv2.putText(bgr, '{} {}'.format(rank, color) if color != 'unknown' else str(rank),
                    (max(0, x), max(18, y - 4)), cv2.FONT_HERSHEY_SIMPLEX, .6, ink, 2)
    if banner is not None:
        for index, prediction in enumerate(packet_predictions(item)):
            if prediction['text']:
                text = '{} {}'.format(prediction['rank'], prediction['text'][:8])
                banner.draw(bgr, text, origin=(8, 8 + index * 48))


def character_binary(crop, plate_color):
    """Diagnostic display only: white glyphs for both blue and green plates."""
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    polarity = cv2.THRESH_BINARY_INV if plate_color == 'green' else cv2.THRESH_BINARY
    return cv2.threshold(gray, 0, 255, polarity | cv2.THRESH_OTSU)[1]


def render_character_preview(item):
    canvas = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
    preview = item[8] if item is not None and len(item) > 8 else None
    if preview is None:
        cv2.putText(canvas, 'Waiting for a fresh plate crop', (24, 220),
                    cv2.FONT_HERSHEY_SIMPLEX, .7, (255,255,255), 2)
    else:
        mask = preview['binary']
        h, w = mask.shape
        scale = min((WIDTH-32)/w, (HEIGHT-120)/h)
        size = (max(1,int(w*scale)), max(1,int(h*scale)))
        enlarged = cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)
        left, top = (WIDTH-size[0])//2, (HEIGHT-size[1])//2
        canvas[top:top+size[1],left:left+size[0]] = cv2.cvtColor(enlarged,cv2.COLOR_GRAY2BGR)
        cv2.putText(canvas, 'Character binary: rank 1 {} frame {}'.format(
            preview['color'], item[1]), (16,32), cv2.FONT_HERSHEY_SIMPLEX,
            .6, (255,255,255), 1)
    cv2.putText(canvas, 'BTN2: camera view   BTN0: snapshot', (16,HEIGHT-20),
                cv2.FONT_HERSHEY_SIMPLEX, .55, (255,255,255), 1)
    return canvas


def detect_worker(bitfile, mode, ocr_model, ocr_debug_dir, crop_margin_x, crop_margin_y, pixels, condition,
                  sequence, captured_at, results, stopping, max_ocr_plates=1, overlay=None):
    boot = time.monotonic()
    milestones = set()

    def milestone(name, frame=None):
        if name not in milestones:
            milestones.add(name)
            print('Detector startup {} elapsed_ms={:.1f} frame={}'.format(
                name, 1000 * (time.monotonic() - boot), frame), flush=True)

    milestone('worker_started')
    try:
        if overlay is None:
            os.nice(5)
    except OSError:
        pass
    cv2.setNumThreads(1)
    try:
        from plate_locator_adaptive import find_candidates_adaptive
        from plate_ocr_baseline import warp_candidate
        if ocr_model:
            from plate_ctc_infer import PlateCTCReader
            from plate_ocr_baseline import warp_candidate
            ocr_reader = PlateCTCReader(ocr_model)
            milestone("model_loaded")
            if ocr_debug_dir:
                Path(ocr_debug_dir).mkdir(parents=True, exist_ok=True)
        if mode in ('adaptive_pl', 'adaptive_pl_green'):
            from pynq import Overlay, allocate, MMIO
            milestone("pynq_imported")
            from phase3_uvc_locator import align_phase3_mask
            milestone("locator_imported")
            if overlay is None:
                overlay = Overlay(bitfile, download=False)
            milestone("overlay_parsed")
            mode_info = overlay.ip_dict.get('axi_gpio_plate_mode')
            if mode == 'adaptive_pl_green' and mode_info is None:
                raise RuntimeError('Green PL mode requires the green overlay with axi_gpio_plate_mode')
            if mode_info is not None:
                control = MMIO(mode_info['phys_addr'], mode_info['addr_range'])
                control.write(4, 0)
                control.write(0, 1 if mode == 'adaptive_pl_green' else 0)
                print('PL color mask: {}'.format(
                    'green' if mode == 'adaptive_pl_green' else 'blue'), flush=True)
            dma = overlay.axi_dma_0
            in_buf = allocate(shape=(HEIGHT, WIDTH), dtype=np.uint32)
            out_buf = allocate(shape=(HEIGHT, WIDTH), dtype=np.uint32)
            milestone("dma_buffers_ready")

        shared = np.frombuffer(pixels, dtype=np.uint8).reshape(HEIGHT, WIDTH, 3)
        previous = 0
        while not stopping.is_set():
            with condition:
                while sequence.value == previous and not stopping.is_set():
                    condition.wait(timeout=0.5)
                if stopping.is_set():
                    break
                previous = sequence.value
                source_time = captured_at.value
                bgr = shared.copy()
            milestone("first_frame_received", previous)
            started = time.perf_counter()
            mask = raw_mask = None
            if mode in ('adaptive_pl', 'adaptive_pl_green'):
                rgba = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGBA)
                in_buf[:] = rgba.view(np.uint32).reshape(HEIGHT, WIDTH)
                dma.recvchannel.transfer(out_buf)
                dma.sendchannel.transfer(in_buf)
                dma.sendchannel.wait()
                dma.recvchannel.wait()
                milestone("first_dma_complete", previous)
                raw = out_buf.view(np.uint8).reshape(HEIGHT, WIDTH, 4)
                raw_mask = np.where(raw[:, :, 0] != 0, 255, 0).astype(np.uint8)
                mask = align_phase3_mask(raw_mask)
            plate_color = ('auto' if mode == 'adaptive_auto' else
                           'green' if mode in ('adaptive_green', 'adaptive_pl_green') else 'blue')
            candidates = locate_with_diagnostics(bgr, mask, raw_mask,
                                                 ocr_debug_dir, previous, plate_color)
            milestone('first_locate', previous)
            if candidates:
                milestone('first_candidate', previous)
            try:
                results.put_nowait(('located', previous, source_time,
                                    time.perf_counter() - started,
                                    candidates[:3], None, 0.0, [], None))
            except Full:
                pass
            plate_text = None
            ocr_seconds = 0.0
            predictions = []
            if ocr_model and candidates:
                predictions = recognize_candidates(
                    bgr, candidates, ocr_reader, warp_candidate, crop_margin_x,
                    crop_margin_y, max_ocr_plates, plate_color, ocr_debug_dir, previous)
                plate_text = predictions[0]['text']
                if any(p['text'] for p in predictions):
                    milestone('first_ocr', previous)
                ocr_seconds = sum(prediction['ocr_seconds'] for prediction in predictions)
            preview = None
            if candidates:
                crop = warp_candidate(bgr, candidates[0][1], crop_margin_x, crop_margin_y)
                color = candidate_color(candidates[0], plate_color)
                preview = {'binary': character_binary(crop, color), 'color': color}
                if ocr_debug_dir:
                    cv2.imwrite(str(Path(ocr_debug_dir) / 'frame_{:06d}_characters.png'.format(previous)),
                                preview['binary'])
            item = ('ok', previous, source_time, time.perf_counter() - started,
                    candidates[:3], plate_text, ocr_seconds, predictions, preview)
            try:
                results.put_nowait(item)
            except Full:
                try:
                    results.get_nowait()
                    results.put_nowait(item)
                except Empty:
                    pass
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        try:
            results.put_nowait(('error', repr(exc)))
        except Full:
            pass
    finally:
        if 'in_buf' in locals():
            in_buf.freebuffer()
        if 'out_buf' in locals():
            out_buf.freebuffer()


def detector_thread(*args):
    import asyncio

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        detect_worker(*args)
    finally:
        loop.close()


def main():
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('bitfile', help='Custom phase3_hdmi.bit; matching HWH must be beside it')
    parser.add_argument('--camera', type=int, default=0)
    parser.add_argument('--test-pattern', action='store_true',
                        help='Show RGB bars without a camera or detector')
    parser.add_argument('--camera-fps', type=int, default=30)
    parser.add_argument('--camera-buffers', type=int, default=1,
                        help='V4L2 queued capture buffers; compare 4 against 1')
    parser.add_argument('--camera-format', choices=('MJPG', 'YUYV'), default='MJPG',
                        help='UVC format; compare YUYV to avoid JPEG decoding')
    parser.add_argument('--pixel-order', choices=('RGB', 'BGR', 'RBG', 'GRB', 'GBR', 'BRG'),
                        default='GBR', help='Byte order in VDMA memory (default: GBR)')
    parser.add_argument('--detector', choices=('off', 'adaptive', 'adaptive_pl', 'adaptive_green', 'adaptive_pl_green', 'adaptive_auto'),
                        default='off')
    parser.add_argument('--ocr-model', type=Path,
                        help='ONNX plate OCR model; model_info.json must be beside it')
    parser.add_argument('--max-ocr-plates', type=int, default=None,
                        help='OCR up to 1..3 candidates per detection; default auto=3, other modes=1. More plates cost more detection time.')
    parser.add_argument('--ocr-log', type=Path,
                        help='Write every detector result and OCR prediction to CSV')
    parser.add_argument('--ocr-debug-dir', type=Path,
                        help='Save each OCR candidate crop and source frame for diagnosis')
    parser.add_argument('--debug-raw-every', type=int, default=0,
                        help='Save raw camera frame every N frames, including misses; 0 disables')
    parser.add_argument('--ocr-margin-x', type=float, default=0.0,
                        help='Extra fraction of plate width on each side (0..0.1)')
    parser.add_argument('--ocr-margin-y', type=float, default=0.0,
                        help='Extra fraction of plate height above/below (0..0.1)')
    parser.add_argument('--frames', type=int, default=0,
                        help='0 means run until Ctrl-C')
    parser.add_argument('--max-result-age', type=float, default=0.7)
    parser.add_argument('--buttons', action='store_true', help='BTN0 snapshot, BTN1 detection, BTN2 character view (3-button overlay)')
    parser.add_argument('--binary-view', action='store_true', help='Start with rank-1 plate character binary preview; BTN2 switches view')
    parser.add_argument('--snapshot-dir', type=Path, default=Path('/home/xilinx/captures'))
    args = parser.parse_args()
    if args.max_ocr_plates is None:
        args.max_ocr_plates = 3 if args.detector == 'adaptive_auto' else 1
    if not 1 <= args.max_ocr_plates <= 3:
        parser.error('--max-ocr-plates must be between 1 and 3')
    if args.camera_buffers < 1:
        parser.error('--camera-buffers must be positive')
    if args.debug_raw_every < 0 or (args.debug_raw_every and not args.ocr_debug_dir):
        parser.error('--debug-raw-every must be nonnegative and requires --ocr-debug-dir')
    if args.debug_raw_every:
        (args.ocr_debug_dir / 'raw').mkdir(parents=True, exist_ok=True)
    if not (0 <= args.ocr_margin_x <= .1 and 0 <= args.ocr_margin_y <= .1):
        parser.error('OCR margins must be between 0 and 0.1')
    if args.camera_fps <= 0 or args.frames < 0 or args.max_result_age <= 0:
        parser.error('Invalid frame rate, frame count, or result age')
    if args.test_pattern and args.detector != 'off':
        parser.error('--test-pattern requires --detector off')
    if args.ocr_model and args.detector == 'off':
        parser.error('--ocr-model requires a detector other than off')
    if args.ocr_log and not args.ocr_model:
        parser.error('--ocr-log requires --ocr-model')
    if args.ocr_debug_dir and not args.ocr_model:
        parser.error('--ocr-debug-dir requires --ocr-model')
    if args.ocr_model and not (args.ocr_model.is_file() and
                               (args.ocr_model.parent / 'model_info.json').is_file()):
        parser.error('--ocr-model and adjacent model_info.json must exist')
    banner = PlateBanner() if args.ocr_model else None

    try:
        from pynq import Overlay
    except ImportError as exc:
        raise SystemExit('Run on the PYNQ board: {}'.format(exc))
    overlay = Overlay(args.bitfile)
    scanout = HdmiScanout(overlay, args.pixel_order)
    camera = None
    worker = None
    controls = None
    detecting = args.detector != 'off'
    binary_view = args.binary_view
    minimum_detection_frame = 0
    notice = ''
    notice_until = 0.0
    try:
        if args.buttons:
            from board_controls import BoardControls
            controls = BoardControls(overlay, args.snapshot_dir)
        if args.test_pattern:
            pattern = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
            colors = ((0, 0, 255), (0, 255, 0), (255, 0, 0))
            for i, (label, color) in enumerate(zip(('RED', 'GREEN', 'BLUE'), colors)):
                left = i * WIDTH // 3
                right = (i + 1) * WIDTH // 3
                pattern[:, left:right] = color
                cv2.putText(pattern, label, (left + 12, HEIGHT // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        else:
            camera = cv2.VideoCapture(args.camera, cv2.CAP_V4L2)
            camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*args.camera_format))
            camera.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
            camera.set(cv2.CAP_PROP_FPS, args.camera_fps)
            camera.set(cv2.CAP_PROP_BUFFERSIZE, args.camera_buffers)
            if not camera.isOpened():
                raise RuntimeError('Cannot open UVC camera {}'.format(args.camera))
            actual_fourcc = int(camera.get(cv2.CAP_PROP_FOURCC))
            actual_format = ''.join(chr((actual_fourcc >> (8*i)) & 255) for i in range(4))
            print('Camera actual: {} {}x{} requested_fps={} reported_fps={}'.format(
                actual_format, int(camera.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT)), args.camera_fps,
                camera.get(cv2.CAP_PROP_FPS)), flush=True)

        if args.detector != 'off':
            context = mp.get_context('spawn')
            pixels = context.RawArray('B', HEIGHT * WIDTH * 3)
            shared = np.frombuffer(pixels, dtype=np.uint8).reshape(HEIGHT, WIDTH, 3)
            condition = Condition()
            sequence = context.Value('I', 0, lock=False)
            captured_at = context.Value('d', 0.0, lock=False)
            results = Queue(maxsize=2)
            stopping = Event()
            worker = Thread(
                target=detector_thread,
                args=(args.bitfile, args.detector,
                      str(args.ocr_model) if args.ocr_model else None,
                      str(args.ocr_debug_dir) if args.ocr_debug_dir else None,
                      args.ocr_margin_x, args.ocr_margin_y,
                      pixels, condition, sequence,
                      captured_at, results, stopping, args.max_ocr_plates, overlay), daemon=True)
            worker.start()

        ocr_log_handle = None
        ocr_log_writer = None
        if args.ocr_log:
            args.ocr_log.parent.mkdir(parents=True, exist_ok=True)
            ocr_log_handle = args.ocr_log.open('w', encoding='utf-8-sig', newline='')
            ocr_log_writer = csv.writer(ocr_log_handle)
            ocr_log_writer.writerow(('frame', 'plate', 'x', 'y', 'w', 'h',
                                     'score', 'detector_ms', 'ocr_ms', 'rank', 'color'))
        count = 0
        pushed = 0
        result_count = 0
        ocr_count = 0
        ocr_seconds_total = 0.0
        latest = None
        binary_cache = None
        binary_until = 0.0
        reported_plate = None
        reported_error = False
        started = time.perf_counter()
        read_seconds = 0.0
        scanout_seconds = 0.0
        print('Custom Phase-3 HDMI: 640x480 detector={}; Ctrl-C to stop'.format(
            args.detector), flush=True)
        if args.ocr_model:
            print('OCR limit: {} plate(s) per detection; each additional plate adds inference time'.format(
                args.max_ocr_plates), flush=True)
        while args.frames == 0 or count < args.frames:
            tick = time.perf_counter()
            if args.test_pattern:
                time.sleep(1 / 30)
                ok, bgr = True, pattern
            else:
                ok, bgr = camera.read()
            read_seconds += time.perf_counter() - tick
            if not ok or bgr is None:
                raise RuntimeError('UVC camera stopped returning frames')
            if bgr.shape != (HEIGHT, WIDTH, 3):
                raise RuntimeError('Expected 640x480 BGR; got {}'.format(bgr.shape))
            count += 1
            if args.debug_raw_every and count % args.debug_raw_every == 0:
                cv2.imwrite(str(args.ocr_debug_dir / 'raw' /
                                'frame_{:06d}_camera.jpg'.format(count)), bgr)
            source_time = time.perf_counter()
            capture_requested = False
            if controls is not None:
                if controls.pending is not None and controls.pending.done():
                    try:
                        saved = controls.pending.result()
                        print('Snapshot saved: {}'.format(saved), flush=True)
                        notice = 'Snapshot saved'
                    except Exception as exc:
                        print('Snapshot error: {}'.format(exc), flush=True)
                        notice = 'Snapshot FAILED'
                    controls.pending = None
                    notice_until = source_time + 2
                while True:
                    try:
                        action = controls.events.get_nowait()
                    except Empty:
                        break
                    if action == 'capture':
                        capture_requested = True
                    elif action == 'toggle':
                        detecting = not detecting if worker is not None else False
                        latest = None
                        binary_cache = None
                        minimum_detection_frame = count + 1
                        notice = 'Detection ON' if detecting else 'Detection PAUSED'
                        notice_until = source_time + 2
                    elif action == 'binary':
                        binary_view = not binary_view
                        binary_cache = None
                        notice = 'Character binary view' if binary_view else 'Camera view'
                        notice_until = source_time + 2
                snapshot_raw = bgr.copy() if capture_requested else None
            if worker is not None and worker.is_alive() and detecting:
                with condition:
                    np.copyto(shared, bgr)
                    sequence.value = count
                    captured_at.value = source_time
                    condition.notify()
            elif worker is not None and not worker.is_alive() and not reported_error:
                print('Detector exited; HDMI preview continues', flush=True)
                reported_error = True

            if worker is not None:
                while True:
                    try:
                        item = results.get_nowait()
                    except Empty:
                        break
                    if item[0] == 'error':
                        print('Detector error: {}; HDMI preview continues'.format(
                            item[1]), flush=True)
                        reported_error = True
                    else:
                        if (detecting and item[1] >= minimum_detection_frame
                                and (latest is None or item[1] >= latest[1])):
                            if (item[0] == 'located' and latest is not None
                                    and item[4] and len(item[4]) == len(latest[4])
                                    and source_time - latest[2] <= args.max_result_age):
                                same_boxes = True
                                for current, previous_box in zip(item[4], latest[4]):
                                    x, y, w, h = current[2]
                                    px, py, pw, ph = previous_box[2]
                                    overlap = (max(0, min(x+w, px+pw) - max(x, px))
                                               * max(0, min(y+h, py+ph) - max(y, py)))
                                    if overlap < 0.6 * (w*h + pw*ph - overlap):
                                        same_boxes = False
                                        break
                                if same_boxes and latest[0] == 'ok':
                                    item = item[:5] + (latest[5], 0.0, latest[7], None)
                            latest = item
                            if binary_view and item[0] == 'ok' and len(item) > 8 and item[8]:
                                binary_cache = item
                                binary_until = time.monotonic() + 1.0
                        if item[0] == 'located':
                            continue
                        result_count += 1
                        predictions = packet_predictions(item)
                        ocr_count += len(predictions)
                        ocr_seconds_total += sum(p['ocr_seconds'] for p in predictions)
                        signature = tuple((p['rank'], p['color'], p['text']) for p in predictions)
                        if signature != reported_plate:
                            for prediction in predictions:
                                print('OCR frame={} rank={} color={} plate={} ocr_ms={:.1f}'.format(
                                    item[1], prediction['rank'], prediction['color'],
                                    prediction['text'], 1000 * prediction['ocr_seconds']), flush=True)
                            reported_plate = signature
                        if ocr_log_writer is not None:
                            ocr_log_writer.writerows(result_log_rows(item))

            fresh = detecting and latest is not None and source_time - latest[2] <= args.max_result_age
            display = bgr
            if binary_view:
                binary_item = (binary_cache if detecting and time.monotonic() < binary_until
                               else None)
                display = render_character_preview(binary_item)
                if binary_item is not None and binary_item is not latest:
                    cv2.putText(display, 'Last valid result', (16, 62),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            elif fresh:
                draw_detection_result(display, latest, banner)
            tick = time.perf_counter()
            if not detecting:
                cv2.putText(display, 'Detection PAUSED', (8, HEIGHT - 12), cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 255, 255), 2)
            if source_time < notice_until:
                cv2.putText(display, notice, (8, HEIGHT - 40), cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 255, 255), 2)
            if capture_requested and controls is not None:
                metadata = {'camera_frame': count, 'detection_enabled': detecting,
                            'display_mode': 'character_binary' if binary_view else 'camera',
                            'binary_source_frame': binary_item[1] if binary_view and binary_item is not None else None,
                            'result_frame': latest[1] if latest else None,
                            'result_age_ms': round(1000 * (source_time - latest[2]), 1) if latest else None,
                            'plate': latest[5] if latest else None,
                            'plates': packet_predictions(latest) if latest else [],
                            'note': 'Camera and async detection result may be different frames; no matched mask is claimed.'}
                accepted = controls.capture(snapshot_raw, display.copy(), metadata)
                notice = 'Saving snapshot' if accepted else 'Snapshot writer busy'
                notice_until = source_time + 2
            pushed += bool(scanout.show_bgr(display))
            scanout_seconds += time.perf_counter() - tick
            if count % 30 == 0:
                elapsed = max(time.perf_counter() - started, 1e-6)
                print('camera_fps={:.2f} hdmi_fps={:.2f} detector_fps={:.2f} camera_ms={:.1f} '
                      'hdmi_copy_ms={:.1f} vdma_status=0x{:08x} vdma_park=0x{:08x}'.format(
                          30 / elapsed, pushed / elapsed, result_count / elapsed,
                          1000 * read_seconds / 30,
                          1000 * scanout_seconds / 30,
                          scanout.mmio.read(VDMA_STATUS),
                          scanout.mmio.read(VDMA_PARK)), flush=True)
                if args.ocr_model:
                    print('ocr_fps={:.2f} ocr_ms={:.1f}'.format(
                        ocr_count / elapsed,
                        1000 * ocr_seconds_total / max(ocr_count, 1)), flush=True)
                if ocr_log_handle is not None:
                    ocr_log_handle.flush()
                started = time.perf_counter()
                read_seconds = 0.0
                scanout_seconds = 0.0
                result_count = 0
                ocr_count = 0
                ocr_seconds_total = 0.0
                pushed = 0
    except KeyboardInterrupt:
        print('Stopped by user', flush=True)
        try:
            black = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
            deadline = time.monotonic() + 0.5
            while not scanout.show_bgr(black):
                if time.monotonic() >= deadline:
                    raise RuntimeError('HDMI did not accept the black frame')
                time.sleep(0.002)
            while (scanout.mmio.read(VDMA_PARK) >> 16) & 0x1f != scanout.requested:
                if time.monotonic() >= deadline:
                    raise RuntimeError('HDMI did not select the black frame')
                time.sleep(0.002)
            time.sleep(0.1)
        except RuntimeError as exc:
            print('Exit clear: {}'.format(exc), flush=True)
    finally:
        if controls is not None:
            controls.close()
        if worker is not None:
            stopping.set()
            with condition:
                condition.notify_all()
            worker.join(timeout=2.0)
            if worker.is_alive():
                print('Detector did not stop within 2 seconds', flush=True)
        if camera is not None:
            camera.release()
        if 'ocr_log_handle' in locals() and ocr_log_handle is not None:
            ocr_log_handle.close()
        scanout.close()


if __name__ == '__main__':
    main()
