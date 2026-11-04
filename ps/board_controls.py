"""Debounced AXI GPIO polling and asynchronous camera snapshots."""
import json
import threading
import time
from pathlib import Path
from queue import Queue
from concurrent.futures import ThreadPoolExecutor
import cv2


def button_actions(rising):
    return tuple(action for bit, action in ((1,'capture'),(2,'toggle'),(4,'binary'))
                 if rising & bit)


class BoardControls:
    def __init__(self, overlay, directory):
        if 'axi_gpio_buttons' not in overlay.ip_dict:
            raise RuntimeError('Overlay lacks axi_gpio_buttons; rebuild with vivado/add_buttons.tcl')
        from pynq import MMIO
        info=overlay.ip_dict['axi_gpio_buttons']
        self.gpio=MMIO(info['phys_addr'],info['addr_range'])
        width = int(info.get('parameters', {}).get('C_GPIO_WIDTH', 2))
        self.button_mask = 7 if width >= 3 else 3
        self.gpio.write(4,self.button_mask)
        print('Buttons: BTN0 snapshot, BTN1 detection{}'.format(
            ', BTN2 character binary view' if width >= 3 else
            '; BTN2 unavailable in this overlay'), flush=True)
        self.directory=Path(directory);self.directory.mkdir(parents=True,exist_ok=True)
        self.events=Queue()
        self.stop=threading.Event()
        self.pool=ThreadPoolExecutor(max_workers=1)
        self.pending=None
        self.thread=threading.Thread(target=self.poll,daemon=True);self.thread.start()

    def poll(self):
        raw=stable=self.gpio.read(0)&self.button_mask
        changed=time.monotonic()
        while not self.stop.wait(.01):
            value=self.gpio.read(0)&self.button_mask
            now=time.monotonic()
            if value!=raw:raw=value;changed=now
            if raw!=stable and now-changed>=.04:
                rising=raw&~stable;stable=raw
                for action in button_actions(rising):self.events.put(action)

    def capture(self, raw, display, metadata):
        if self.pending is not None and not self.pending.done():return False
        name=time.strftime('%Y%m%d_%H%M%S')+'_'+str(time.time_ns())
        folder=self.directory/name
        def write():
            folder.mkdir()
            for key,image in [('camera',raw),('display',display)]:
                if not cv2.imwrite(str(folder/(key+'.png')),image):raise RuntimeError('Snapshot write failed')
            (folder/'result.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf-8')
            return str(folder)
        self.pending=self.pool.submit(write)
        return True

    def close(self):
        self.stop.set();self.thread.join(timeout=1)
        self.pool.shutdown(wait=True)
