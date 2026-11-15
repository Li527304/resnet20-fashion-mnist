"""Shared inference path for command line and local demo."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F
from models import VARIANTS, ResNet20

ROOT=Path(__file__).resolve().parent
CLASSES=['T恤','裤子','套头衫','连衣裙','外套','凉鞋','衬衫','运动鞋','包','短靴']

def preprocess(image):
    image=image.convert('L').resize((28,28),Image.Resampling.BILINEAR)
    a=torch.from_numpy(np.array(image).copy()).float()[None,None]/255
    return (F.pad(a,(2,2,2,2))-.2860)/.3530

def load(variant='base', seed=17, device='cpu', results_dir=None):
    if results_dir is None:
        results_dir = 'results' if variant in ('base', 'se') else 'results/structure'
    ck=torch.load(ROOT/Path(results_dir)/f'{variant}_{seed}'/'best.pt',map_location=device,weights_only=True)
    model=ResNet20(variant).to(device); model.load_state_dict(ck['state_dict']); model.eval()
    return model

@torch.no_grad()
def predict(model,image):
    probabilities=model(preprocess(image).to(next(model.parameters()).device)).softmax(1)[0]
    return [{'class_id':int(i),'label':CLASSES[int(i)],'probability':float(probabilities[i])} for i in probabilities.argsort(descending=True)]

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('image');p.add_argument('--variant',choices=VARIANTS,default='base');p.add_argument('--seed',type=int,default=17);p.add_argument('--output',default='results/inference.json');p.add_argument('--results-dir',type=Path);a=p.parse_args()
    result=predict(load(a.variant,a.seed,results_dir=a.results_dir),Image.open(a.image));path=ROOT/a.output;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(result,ensure_ascii=False,indent=2))
