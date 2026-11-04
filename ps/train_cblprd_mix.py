"""Fine-tune the original TinyPlateCTC with source-balanced real/GAN crops."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import sys
import time
import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset,DataLoader
from train_tiny_plate_ctc import TinyPlateCTC,VOCAB,TO_ID,augment_plate,collate,greedy_decode

MASS={'base7':.25,'green8':.15,'detector7':.15,'clpd':.05,'cbl_blue':.20,'cbl_green':.20}


class CachedPlates(Dataset):
    def __init__(self,root,rows,augment=False,seed=20261103):
        self.images=np.load(root/'images.npy',mmap_mode='r')
        self.rows=rows;self.augment=augment;self.rng=np.random.default_rng(seed)
    def __len__(self):return len(self.rows)
    def __getitem__(self,i):
        r=self.rows[i];im=self.images[r['index']].copy()
        if self.augment:
            im=augment_plate(im,self.rng)
            # Mild screen/capture-like resolution loss; never change labels.
            if self.rng.random()<.25:
                size=(int(self.rng.integers(72,121)),int(self.rng.integers(18,31)))
                im=cv2.resize(cv2.resize(im,size,interpolation=cv2.INTER_AREA),(128,32),interpolation=cv2.INTER_LINEAR)
        return torch.from_numpy(im).float().unsqueeze(0)/255.,torch.tensor([TO_ID[c] for c in r['plate']],dtype=torch.long),r['plate'],r['group']


def edit_distance(a,b):
    row=list(range(len(b)+1))
    for i,x in enumerate(a,1):
        next_row=[i]
        for j,y in enumerate(b,1):next_row.append(min(next_row[-1]+1,row[j]+1,row[j-1]+(x!=y)))
        row=next_row
    return row[-1]


@torch.no_grad()
def evaluate(model,loader,device,collect=False):
    model.eval();counts={};provinces={};predictions=[];offset=0
    for images,_,_,labels,groups in loader:
        guesses=greedy_decode(model(images.to(device)))
        for truth,guess,group in zip(labels,guesses,groups):
            if collect:
                r=loader.dataset.rows[offset]
                predictions.append({'index':r['index'],'path':r['path'],'truth':truth,'prediction':guess,'group':group})
            offset+=1
            for target,key in ((counts,group),(provinces,truth[0])):
                r=target.setdefault(key,{'count':0,'exact':0,'first_correct':0,'char_errors':0,'chars':0})
                r['count']+=1;r['exact']+=int(truth==guess);r['first_correct']+=int(bool(guess) and truth[0]==guess[0]);r['char_errors']+=edit_distance(truth,guess);r['chars']+=len(truth)
    for r in list(counts.values())+list(provinces.values()):
        r.update(exact_rate=r['exact']/r['count'],province_rate=r['first_correct']/r['count'],cer=r['char_errors']/r['chars'])
    result={'groups':counts,'provinces':provinces,'macro_source_exact':sum(r['exact_rate'] for r in counts.values())/len(counts)}
    if collect:result['predictions']=predictions
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,default=Path('build/ocr_cblprd_prepared'))
    p.add_argument('--init-checkpoint',type=Path,default=Path('build/ocr_tiny_ctc_detector_ft/best.pt'))
    p.add_argument('--output',type=Path,default=Path('build/ocr_cblprd_mix'))
    p.add_argument('--epochs',type=int,default=6);p.add_argument('--samples-per-epoch',type=int,default=100000)
    p.add_argument('--batch-size',type=int,default=128);p.add_argument('--lr',type=float,default=.00005)
    p.add_argument('--seed',type=int,default=20261103)
    p.add_argument('--source-mass',type=Path,help='Optional JSON source sampling proportions')
    p.add_argument('--reference-baseline',type=Path,help='Keep retention gate relative to original deployed model')
    p.add_argument('--skip-test',action='store_true',help='Validation-only continuation; leave frozen test unopened')
    a=p.parse_args()
    if min(a.epochs,a.samples_per_epoch,a.batch_size)<1 or a.lr<=0:p.error('Positive training settings required')
    if (a.output/'baseline.json').exists():p.error('Output already contains a run; choose another directory')
    a.output.mkdir(parents=True,exist_ok=True)
    random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed);torch.set_num_threads(2);cv2.setNumThreads(1)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    rows=json.loads((a.data/'manifest.json').read_text(encoding='utf-8'))
    train_rows=[r for r in rows if r['split']=='train'];val_rows=[r for r in rows if r['split']=='val'];test_rows=[r for r in rows if r['split']=='test']
    train=CachedPlates(a.data,train_rows,True,a.seed);val=CachedPlates(a.data,val_rows)
    loader=DataLoader(val,batch_size=a.batch_size,collate_fn=collate,num_workers=0)
    initial=torch.load(a.init_checkpoint,map_location='cpu',weights_only=False)
    if initial['vocab']!=VOCAB:raise RuntimeError('Vocabulary differs from checkpoint')
    model=TinyPlateCTC().to(device);model.load_state_dict(initial['state_dict'])
    print(json.dumps({'stage':'baseline','device':str(device),'gpu':torch.cuda.get_device_name(0) if device.type=='cuda' else None,'train':len(train_rows),'val':len(val_rows),'heldout_test':len(test_rows)},ensure_ascii=False),flush=True)
    baseline=evaluate(model,loader,device,collect=True)
    reference=json.loads(a.reference_baseline.read_text(encoding='utf-8')) if a.reference_baseline else baseline
    mass=json.loads(a.source_mass.read_text(encoding='utf-8')) if a.source_mass else MASS
    if set(mass)!=set(MASS) or any(v<=0 for v in mass.values()) or abs(sum(mass.values())-1)>1e-6:
        raise ValueError('Source masses must be positive, sum to one, and cover all sources')
    (a.output/'baseline.json').write_text(json.dumps(baseline,ensure_ascii=False,indent=2),encoding='utf-8')
    config={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()}
    config.update(source_mass=mass,initial_sha256=hashlib.sha256(a.init_checkpoint.read_bytes()).hexdigest(),manifest_sha256=hashlib.sha256((a.data/'manifest.json').read_bytes()).hexdigest(),selection='macro source exact; safe gate base7 drop<=0.005, green8 drop<=0.01 vs original')
    (a.output/'config.json').write_text(json.dumps(config,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'baseline_rates':{g:r['exact_rate'] for g,r in baseline['groups'].items()}},ensure_ascii=False),flush=True)
    counts=Counter(r['group'] for r in train_rows);weights=torch.tensor([mass[r['group']]/counts[r['group']] for r in train_rows],dtype=torch.double)
    generator=torch.Generator().manual_seed(a.seed)
    optimizer=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=1e-4)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,a.epochs,eta_min=a.lr*.05)
    criterion=nn.CTCLoss(blank=0,zero_infinity=True);history=[];best=-1.;safe_best=-1.;seen=set()
    for epoch in range(1,a.epochs+1):
        started=time.perf_counter();model.train();loss_sum=0.;count=0
        indices=torch.multinomial(weights,a.samples_per_epoch,replacement=True,generator=generator).tolist();seen.update(indices)
        train_loader=DataLoader(train,batch_size=a.batch_size,sampler=indices,collate_fn=collate,num_workers=0)
        for step,(images,targets,lengths,_,_) in enumerate(train_loader,1):
            images=images.to(device);targets=targets.to(device);logits=model(images)
            in_lengths=torch.full((len(images),),logits.shape[1],dtype=torch.long,device=device)
            loss=criterion(logits.log_softmax(-1).transpose(0,1),targets,in_lengths,lengths)
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite CTC loss')
            optimizer.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step()
            loss_sum+=float(loss.detach())*len(images);count+=len(images)
            if step%100==0:print(json.dumps({'epoch':epoch,'step':step,'samples':count,'loss':round(loss_sum/count,4)}),flush=True)
        scheduler.step();metrics=evaluate(model,loader,device)
        safe=all(metrics['groups'][g]['exact_rate']>=reference['groups'][g]['exact_rate']-tol for g,tol in [('base7',.005),('green8',.01)])
        row={'epoch':epoch,'loss':loss_sum/count,'val':metrics,'safe_gate':safe,'seconds':round(time.perf_counter()-started,1),'sampled_groups':dict(Counter(train_rows[i]['group'] for i in indices)),'unique_rows_seen':len(seen)}
        history.append(row)
        checkpoint={'state_dict':model.state_dict(),'vocab':VOCAB,'input_size':[128,32],'epoch':epoch,'metrics':metrics,'parameters':sum(x.numel() for x in model.parameters()),'source':'cblprd_real_mix','config':config}
        torch.save(checkpoint,a.output/'last.pt')
        score=metrics['macro_source_exact']
        if score>best:best=score;torch.save(checkpoint,a.output/'best.pt')
        if safe and score>safe_best:safe_best=score;torch.save(checkpoint,a.output/'best_safe.pt')
        (a.output/'history.json').write_text(json.dumps(history,ensure_ascii=False,indent=2),encoding='utf-8')
        print(json.dumps({'epoch':epoch,'safe_gate':safe,'rates':{g:r['exact_rate'] for g,r in metrics['groups'].items()},'seconds':row['seconds']},ensure_ascii=False),flush=True)
    chosen=a.output/('best_safe.pt' if (a.output/'best_safe.pt').exists() else 'best.pt')
    saved=torch.load(chosen,map_location=device,weights_only=False);model.load_state_dict(saved['state_dict'])
    real_rows=[r for r in val_rows if not r['group'].startswith('cbl_')]
    real_loader=DataLoader(CachedPlates(a.data,real_rows),batch_size=a.batch_size,collate_fn=collate,num_workers=0)
    candidate_real=evaluate(model,real_loader,device,collect=True)
    previous={r['index']:r for r in baseline['predictions']}
    changes=[]
    for r in candidate_real['predictions']:
        old=previous[r['index']]
        if old['prediction']!=r['prediction']:
            changes.append(dict(r,old_prediction=old['prediction'],
                                regression=old['prediction']==r['truth'] and r['prediction']!=r['truth'],
                                improvement=old['prediction']!=r['truth'] and r['prediction']==r['truth']))
    (a.output/'paired_real_validation.json').write_text(json.dumps({'changes':changes,
        'regressions':sum(r['regression'] for r in changes),'improvements':sum(r['improvement'] for r in changes)},ensure_ascii=False,indent=2),encoding='utf-8')
    if a.skip_test:
        print(json.dumps({'complete':True,'chosen':str(chosen),'safe':safe_best>=0,'test':'not evaluated'}),flush=True)
        return
    test_loader=DataLoader(CachedPlates(a.data,test_rows),batch_size=a.batch_size,collate_fn=collate,num_workers=0)
    candidate_test=evaluate(model,test_loader,device)
    model.load_state_dict(initial['state_dict']);original_test=evaluate(model,test_loader,device)
    report={'chosen':str(chosen),'safe_candidate_available':safe_best>=0,'candidate':candidate_test,'original':original_test,'note':'Frozen synthetic test; real validation was used in selection. Board frames remain development diagnostics, not independent test.'}
    (a.output/'test_comparison.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'complete':True,'chosen':str(chosen),'test_original':original_test['macro_source_exact'],'test_candidate':candidate_test['macro_source_exact'],'safe':safe_best>=0}),flush=True)


if __name__=='__main__':
    if hasattr(sys.stdout,'reconfigure'):sys.stdout.reconfigure(encoding='utf-8')
    main()
