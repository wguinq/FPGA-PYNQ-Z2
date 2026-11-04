"""Compare original and mixed CLPD checkpoints on identical validation crops."""
import argparse
import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from train_tiny_plate_ctc import TinyPlateCTC, PlateDataset, collate, validate


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    torch.set_num_threads(4)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset=PlateDataset(Path('build/ocr_real_crops'),('base7','green8','clpd'),False)
    loader=DataLoader(dataset,batch_size=128,collate_fn=collate)
    saved=torch.load(args.checkpoint,map_location=device,weights_only=False)
    model=TinyPlateCTC().to(device).eval()
    model.load_state_dict(saved['state_dict'])
    metrics=validate(model,loader,device)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(metrics,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in metrics.items() if k in ('overall','base7','green8','clpd')},ensure_ascii=True))


if __name__=='__main__':main()
