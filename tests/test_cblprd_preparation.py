import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import cv2
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'ps'))
from prepare_cblprd_mixed import main


class PreparationTests(unittest.TestCase):
    def test_reserved_labels_duplicates_and_unsupported_characters(self):
        root=Path(__file__).resolve().parents[1]/'build'
        with tempfile.TemporaryDirectory(dir=root) as tmp:
            p=Path(tmp);source=p/'source';real=p/'real';source.mkdir();real.mkdir()
            random=np.random.default_rng(0)
            for i in range(4):
                cv2.imwrite(str(source/('im%d.png'%i)),random.integers(0,255,(32,128),dtype=np.uint8))
            cv2.imwrite(str(real/'val.png'),random.integers(0,255,(32,128),dtype=np.uint8))
            (real/'base7_val.tsv').write_text('val.png\t京A12345\n',encoding='utf-8')
            (source/'train.txt').write_text(
                'im0.png 沪A12345 普通蓝牌\n'
                'im1.png 京A12345 普通蓝牌\n'
                'im2.png 粤AI1234 普通蓝牌\n'
                'im3.png 黑M1315V 普通蓝牌\n'
                'im3.png 粤A12345 双层黄牌\n'
                'im2.png 冀A54321 普通蓝牌\n',encoding='utf-8')
            # Same pixels as training but a different label: quarantine both.
            (source/'val.txt').write_text('im2.png 苏A54321 普通蓝牌\n',encoding='utf-8')
            out=p/'out'
            with patch.object(sys,'argv',['prepare','--source',str(source),'--real',str(real),'--output',str(out),'--workers','1']):
                main()
            rows=json.loads((out/'manifest.json').read_text(encoding='utf-8'))
            self.assertEqual({r['plate'] for r in rows},{'沪A12345','京A12345'})
            self.assertEqual([r['plate'] for r in rows if r['split']=='train'],['沪A12345'])
            audit=json.loads((out/'audit.json').read_text(encoding='utf-8'))
            self.assertEqual(audit['excluded']['unsupported_characters'],1)
            self.assertEqual(audit['excluded']['conflicting_duplicate_pixels'],2)


if __name__=='__main__':unittest.main()
