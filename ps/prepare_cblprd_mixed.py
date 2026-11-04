"""Audit blue/green CBLPRD and real crops; cache grayscale inputs without editing sources."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import cv2
import numpy as np

VOCAB = '皖沪津渝冀晋蒙辽吉黑苏浙京闽赣鲁豫鄂湘粤桂琼川贵云藏陕甘青宁新警学ABCDEFGHJKLMNPQRSTUVWXYZ0123456789'
TYPES = {'普通蓝牌': 'cbl_blue', '新能源小型车': 'cbl_green', '新能源大型车': 'cbl_green'}
BOARD_LABELS = {'黑M1315V','皖ADB8617','黑AF52193','黑AFU775','渝AN7968','粤AMT229','粤B779ST','桂AP8C55'}


def audit_image(row):
    try:
        raw = Path(row['path']).read_bytes()
        image = cv2.imdecode(np.frombuffer(raw,np.uint8),cv2.IMREAD_GRAYSCALE)
        if image is None or min(image.shape) < 8:
            return None, 'unreadable_or_tiny'
        digest = hashlib.sha256(str(image.shape).encode()+image.tobytes()).hexdigest()
        return (digest, list(image.shape), cv2.resize(image,(128,32),interpolation=cv2.INTER_AREA)), None
    except (OSError,cv2.error) as exc:
        return None, type(exc).__name__


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=Path('F:/data/CBLPRD-330k_v1'))
    p.add_argument('--real',type=Path,default=Path('build/ocr_real_crops'))
    p.add_argument('--output',type=Path,default=Path('build/ocr_cblprd_prepared'))
    p.add_argument('--workers',type=int,default=4)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    if (a.output/'manifest.json').exists():
        raise SystemExit('Prepared manifest exists; choose a new output to preserve the audited split')
    cv2.setNumThreads(1)
    rows=[];excluded=Counter();initial_seen=set();input_counts=Counter()
    for group in ('base7','green8','detector7','clpd'):
        for split in ('train','val'):
            path=a.real/('{}_{}.tsv'.format(group,split))
            if not path.exists():continue
            for line in path.read_text(encoding='utf-8-sig').splitlines():
                relative,label=line.split('\t',1)
                if split=='train':initial_seen.add(label)
                rows.append({'path':str((a.real/relative).resolve()),'plate':label,'group':group,'split':split})
    for split in ('train','val'):
        for line in (a.source/(split+'.txt')).read_text(encoding='utf-8-sig').splitlines():
            parts=line.split();input_counts[split]+=1
            if len(parts)!=3:
                excluded['malformed']+=1;continue
            relative,label,kind=parts
            if kind not in TYPES:
                excluded['other_plate_type']+=1;continue
            if any(c not in VOCAB for c in label):
                excluded['unsupported_characters']+=1;continue
            if len(label)!=(7 if kind=='普通蓝牌' else 8):
                excluded['unexpected_length']+=1;continue
            if split=='val' and label in initial_seen:
                excluded['synthetic_validation_label_seen_by_initial_model']+=1;continue
            partition=split
            if split=='val' and int(hashlib.sha256(label.encode()).hexdigest(),16)%5==0:
                partition='test'
            rows.append({'path':str((a.source/relative).resolve()),'plate':label,
                         'group':TYPES[kind],'split':partition})
    protected={r['plate'] for r in rows if r['split']!='train'}|BOARD_LABELS
    filtered=[]
    for row in rows:
        if row['split']=='train' and row['plate'] in protected:
            excluded['training_label_reserved_for_validation_or_board']+=1
        else:filtered.append(row)
    rows=filtered
    # Validate every retained image and keep exact decoded-pixel hashes. One
    # fixed-size array avoids retaining hundreds of thousands of Python arrays.
    cache=np.lib.format.open_memmap(a.output/'images.npy',mode='w+',dtype=np.uint8,shape=(len(rows),32,128))
    valid=[];errors=[]
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        # Bound the pending work and results on Windows to limit peak memory.
        for start in range(0,len(rows),512):
            batch=rows[start:start+512]
            for offset,(result,error) in enumerate(pool.map(audit_image,batch)):
                row=batch[offset];index=start+offset
                if error:
                    excluded['image_'+error]+=1;errors.append({'path':row['path'],'reason':error});continue
                digest,shape,image=result;cache[index]=image
                valid.append(dict(row,index=index,pixel_sha256=digest,original_shape=shape))
            if start%10240==0:print(json.dumps({'audited':min(start+512,len(rows)),'total':len(rows)}),flush=True)
    cache.flush();del cache
    real_train_hashes={r['pixel_sha256'] for r in valid if r['split']=='train' and not r['group'].startswith('cbl_')}
    keep=[]
    for r in valid:
        if r['split']!='train' and r['group'].startswith('cbl_') and r['pixel_sha256'] in real_train_hashes:
            excluded['synthetic_validation_pixels_seen_by_initial_model']+=1
        else:keep.append(r)
    protected_hashes={r['pixel_sha256'] for r in keep if r['split']!='train'}
    final=[];seen={}
    # Validation/test take priority over training. Conflicting-label exact
    # duplicate images are quarantined from every partition.
    labels_by_hash={}
    for r in keep:labels_by_hash.setdefault(r['pixel_sha256'],set()).add(r['plate'])
    for r in sorted(keep,key=lambda r:0 if r['split']=='test' else 1 if r['split']=='val' else 2):
        h=r['pixel_sha256']
        if len(labels_by_hash[h])>1:
            excluded['conflicting_duplicate_pixels']+=1;continue
        if r['split']=='train' and h in protected_hashes:
            excluded['training_pixels_reserved_for_validation']+=1;continue
        if h in seen:
            excluded['duplicate_pixels']+=1;continue
        seen[h]=r['split'];final.append(r)
    sets={s:{r['plate'] for r in final if r['split']==s} for s in ('train','val','test')}
    assert not(sets['train']&sets['val'] or sets['train']&sets['test'] or sets['val']&sets['test'])
    counts=Counter(r['split']+'/'+r['group'] for r in final)
    summary={'input_records':dict(input_counts),'cache_records':len(rows),'retained':len(final),
             'counts':dict(counts),'excluded':dict(excluded),'source':'GAN-generated CBLPRD mixed with real CCPD/CLPD',
             'test_rule':'official CBL validation labels with sha256(label) mod5=0; never used for checkpoint selection',
             'train_provinces':dict(Counter(r['plate'][0] for r in final if r['split']=='train')),
             'board_labels_excluded_from_training':sorted(BOARD_LABELS)}
    (a.output/'manifest.json').write_text(json.dumps(final,ensure_ascii=False),encoding='utf-8')
    (a.output/'audit.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    (a.output/'errors.json').write_text(json.dumps(errors,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
