"""Prepare independently grouped CLPD crops; exclude all existing CCPD labels."""
import csv
import json
import random
from collections import Counter,defaultdict
from pathlib import Path
import cv2
import numpy as np
from train_tiny_plate_ctc import VOCAB


def main():
    root=Path('F:/data/CLPD')
    data=Path('build/ocr_real_crops')
    seen=set()
    for name in ('base7_train','base7_val','green8_train','green8_val','detector7_train'):
        seen.update(line.split('\t',1)[1] for line in (data/(name+'.tsv')).read_text(encoding='utf-8').splitlines())
    rows=list(csv.DictReader((root/'CLPD.csv').open(encoding='gbk',newline='')))
    groups=defaultdict(list)
    rejects=Counter()
    for row in rows:
        text=row['label'].strip()
        if len(text) not in (7,8) or any(c not in VOCAB for c in text):
            rejects['unsupported_label']+=1;continue
        if text in seen:
            rejects['existing_ccpd_number']+=1;continue
        groups[text].append(row)
    provinces=defaultdict(list)
    for text in groups:provinces[text[0]].append(text)
    split={}
    rng=random.Random(20261101)
    for province,labels in sorted(provinces.items()):
        labels=sorted(labels);rng.shuffle(labels)
        n=max(1,round(len(labels)*.2)) if len(labels)>1 else 0
        for i,label in enumerate(labels):split[label]='val' if i<n else 'train'
    manifests={'train':[],'val':[]};counts={'train':Counter(),'val':Counter()}
    for label,items in groups.items():
        part=split[label]
        folder=data/'clpd'/part;folder.mkdir(parents=True,exist_ok=True)
        for row in items:
            path=root/row['path'].replace('\\','/')
            image=cv2.imdecode(np.fromfile(str(path),dtype=np.uint8),cv2.IMREAD_COLOR)
            if image is None:raise ValueError(str(path))
            points=np.float32([[float(row['x'+str(i)]),float(row['y'+str(i)])] for i in range(1,5)])
            if cv2.contourArea(points)<4:rejects['invalid_quad']+=1;continue
            target=np.float32([[0,0],[223,0],[223,63],[0,63]])
            crop=cv2.warpPerspective(image,cv2.getPerspectiveTransform(points,target),(224,64),borderMode=cv2.BORDER_REPLICATE)
            output=folder/(path.stem+'.jpg')
            cv2.imencode('.jpg',crop)[1].tofile(str(output))
            manifests[part].append(output.relative_to(data).as_posix()+'\t'+label)
            counts[part][label[0]]+=1
    for part,lines in manifests.items():
        (data/('clpd_'+part+'.tsv')).write_text('\n'.join(lines)+'\n',encoding='utf-8')
    report={'source_rows':len(rows),'accepted_train':len(manifests['train']),'accepted_val':len(manifests['val']),
            'rejected':dict(rejects),'province_counts':{k:dict(v) for k,v in counts.items()},
            'notes':'Full plate grouped split; all existing CCPD numbers excluded. Singleton provinces have no validation label.'}
    Path('build/clpd_preparation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=True))


if __name__=='__main__':main()
