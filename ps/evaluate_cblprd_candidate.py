"""Freeze a validation-selected candidate, then compare against deployed weights."""
import argparse
import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from train_cblprd_mix import CachedPlates,evaluate
from train_tiny_plate_ctc import TinyPlateCTC,VOCAB,collate


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--data',type=Path,default=Path('build/ocr_cblprd_prepared'))
    p.add_argument('--original',type=Path,default=Path('build/ocr_tiny_ctc_detector_ft/best.pt'))
    p.add_argument('--output',type=Path,default=Path('build/ocr_cblprd_selected'))
    a=p.parse_args();a.output.mkdir(exist_ok=True,parents=True)
    torch.set_num_threads(2);device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    rows=json.loads((a.data/'manifest.json').read_text(encoding='utf-8'))
    real=[r for r in rows if r['split']=='val' and not r['group'].startswith('cbl_')]
    test=[r for r in rows if r['split']=='test'];report={}
    model=TinyPlateCTC().to(device)
    for name,path in [('original',a.original),('candidate',a.checkpoint)]:
        saved=torch.load(path,map_location=device,weights_only=False)
        if saved['vocab']!=VOCAB:raise RuntimeError('Vocabulary mismatch')
        model.load_state_dict(saved['state_dict'])
        real_loader=DataLoader(CachedPlates(a.data,real),batch_size=128,collate_fn=collate,num_workers=0)
        test_loader=DataLoader(CachedPlates(a.data,test),batch_size=128,collate_fn=collate,num_workers=0)
        report[name]={'real_validation':evaluate(model,real_loader,device,collect=True),
                      'synthetic_test':evaluate(model,test_loader,device)}
    changes=[]
    before={r['index']:r for r in report['original']['real_validation'].pop('predictions')}
    after=report['candidate']['real_validation'].pop('predictions')
    for r in after:
        old=before[r['index']]['prediction']
        if old!=r['prediction']:
            changes.append(dict(r,old_prediction=old,regression=old==r['truth'] and r['prediction']!=r['truth'],
                                improvement=old!=r['truth'] and r['prediction']==r['truth']))
    report.update(checkpoint=str(a.checkpoint),regressions=sum(r['regression'] for r in changes),
                  improvements=sum(r['improvement'] for r in changes),
                  note='Real validation influenced model selection. Synthetic test is label/pixel disjoint. Board diagnostics are separate development samples.')
    (a.output/'comparison.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    (a.output/'paired_real_changes.json').write_text(json.dumps(changes,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=True),flush=True)


if __name__=='__main__':main()
