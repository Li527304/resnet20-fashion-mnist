"""Recompute all stored predictions from checkpoints and check reported metrics."""
import argparse, json, hashlib
from pathlib import Path
import numpy as np
import torch
from train import ROOT,data,evaluate
from infer import load,preprocess
from torchvision.datasets import FashionMNIST
from models import profile

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir', type=Path, default=ROOT/'results')
    results_dir = ROOT / parser.parse_args().results_dir
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    device='cuda' if torch.cuda.is_available() else 'cpu'
    _,_,_,_,x,y=data(device, results_dir)
    cfg=json.loads((results_dir/'config.json').read_text(encoding='utf-8'))
    checks=[]
    ds=FashionMNIST(ROOT/'data',train=False)
    assert torch.equal(preprocess(ds[0][0])[0],x[0].cpu())
    split=json.loads((results_dir/'split.json').read_text(encoding='utf-8'))
    assert not set(split['train_ids'])&set(split['validation_ids'])
    assert len(set(split['train_ids'])|set(split['validation_ids']))==60000
    for v in cfg['variants']:
        for seed in cfg['seeds']:
            out=results_dir/f'{v}_{seed}';m=json.loads((out/'metrics.json').read_text(encoding='utf-8'));net=load(v,seed,device,results_dir)
            loss,acc,logits=evaluate(net,x,y);saved=np.load(out/'predictions.npz');prediction=logits.numpy().argmax(1)
            assert np.array_equal(prediction,saved['logits'].argmax(1))
            assert np.array_equal(y.cpu().numpy(), saved['labels'])
            assert int((prediction==saved['labels']).sum())==m['test_correct']
            assert abs(acc-m['test_accuracy'])<1e-7
            assert profile(net,device)==(m['params'],m['macs'])
            cm=np.zeros((10,10),dtype=np.int64);np.add.at(cm,(saved['labels'],prediction),1)
            assert np.array_equal(cm,saved['confusion'])
            assert hashlib.sha256((out/'best.pt').read_bytes()).hexdigest()==m['checkpoint_sha256']
            checks.append({'variant':v,'seed':seed,'test_correct':m['test_correct'],'prediction_match':True,'max_logit_absolute_difference':float(np.abs(logits.numpy()-saved['logits']).max())})
    (results_dir/'verification.json').write_text(json.dumps({'status':'passed','runs':checks,'split_disjoint':True,'inference_preprocess_matches_training':True},ensure_ascii=False,indent=2),encoding='utf-8');print('PASS:',len(checks),'checkpoints reproduced; predictions, metrics, complexity and hashes match.')

if __name__=='__main__':main()
