"""Train all preregistered runs. Test set is evaluated after checkpoint selection."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse, csv, hashlib, json, platform, random, time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from torchvision.datasets import FashionMNIST
from models import VARIANTS, ResNet20, profile

ROOT=Path(__file__).resolve().parent
DEFAULT_RESULTS = ROOT / 'results'
FashionMNIST.mirrors=['https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/']

def write_json(path,obj): path.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')

def data(device, results_dir=DEFAULT_RESULTS):
    tr=FashionMNIST(ROOT/'data',train=True,download=True)
    te=FashionMNIST(ROOT/'data',train=False,download=True)
    gen=torch.Generator().manual_seed(2026)
    train_ids=[]; val_ids=[]
    for c in range(10):
        ids=torch.where(tr.targets==c)[0]; ids=ids[torch.randperm(len(ids),generator=gen)]
        val_ids.extend(ids[:600].tolist()); train_ids.extend(ids[600:].tolist())
    def tensor(x): return (F.pad(x[:,None].float()/255,(2,2,2,2))-.2860)/.3530
    x=tensor(tr.data).to(device); y=tr.targets.to(device)
    xt=tensor(te.data).to(device); yt=te.targets.to(device)
    split={'train_ids':train_ids,'validation_ids':val_ids,'seed':2026}
    write_json(results_dir/'split.json',split)
    return x[train_ids],y[train_ids],x[val_ids],y[val_ids],xt,yt

@torch.no_grad()
def evaluate(model,x,y,batch=512):
    model.eval(); logits=torch.cat([model(z) for z in x.split(batch)])
    return F.cross_entropy(logits,y).item(),(logits.argmax(1)==y).float().mean().item(),logits.cpu()

def main():
    p=argparse.ArgumentParser(); p.add_argument('--epochs',type=int,default=15)
    p.add_argument('--seeds',nargs='+',type=int,default=[17,29,43]); p.add_argument('--variants',nargs='+',choices=VARIANTS,default=['base','se'])
    p.add_argument('--batch-size',type=int,default=256)
    p.add_argument('--results-dir', type=Path, default=DEFAULT_RESULTS)
    args=p.parse_args()
    results_dir = ROOT / args.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4); torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    device='cuda' if torch.cuda.is_available() else 'cpu'
    write_json(results_dir/'environment.json',{'python':platform.python_version(),'torch':torch.__version__,'device':device,'gpu':torch.cuda.get_device_name(0) if device=='cuda' else platform.processor(),'platform':platform.platform()})
    config={key: value for key, value in vars(args).items() if key != 'results_dir'}|{'lr':.002,'weight_decay':.0001,'optimizer':'AdamW','scheduler':'CosineAnnealingLR','split_seed':2026,'augmentation':'none','input':[1,32,32],'goal_accuracy':.90}
    cfg=results_dir/'config.json'
    if cfg.exists() and json.loads(cfg.read_text(encoding='utf-8'))!=config: raise RuntimeError('Existing config differs; use another output directory rather than mixing experiments.')
    write_json(cfg,config)
    x,y,xv,yv,xt,yt=data(device, results_dir)
    for variant in args.variants:
        for seed in args.seeds:
            out=results_dir/f'{variant}_{seed}'; out.mkdir(exist_ok=True)
            if (out/'metrics.json').exists(): continue
            random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
            if device=='cuda': torch.cuda.manual_seed_all(seed)
            model=ResNet20(variant).to(device); params,macs=profile(model,device)
            opt=torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=.0001)
            scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=args.epochs)
            # Independent permutation stream gives every architecture identical minibatch orders.
            order_gen=torch.Generator().manual_seed(seed)
            history=[]; best=-1; best_epoch=0; started=time.perf_counter()
            for epoch in range(1,args.epochs+1):
                model.train(); perm=torch.randperm(len(y),generator=order_gen).to(device)
                loss_sum=torch.zeros((),device=device); correct=torch.zeros((),device=device)
                for ids in perm.split(args.batch_size):
                    opt.zero_grad(set_to_none=True); logits=model(x[ids]); loss=F.cross_entropy(logits,y[ids]); loss.backward(); opt.step()
                    loss_sum+=loss.detach()*len(ids); correct+=(logits.argmax(1)==y[ids]).sum()
                vl,va,_=evaluate(model,xv,yv)
                row={'epoch':epoch,'train_loss':loss_sum.item()/len(y),'train_accuracy':correct.item()/len(y),'val_loss':vl,'val_accuracy':va,'lr':opt.param_groups[0]['lr']}
                history.append(row); scheduler.step()
                if va>best:
                    best=va; best_epoch=epoch
                    torch.save({'state_dict':model.state_dict(),'variant':variant,'seed':seed,'epoch':epoch},out/'best.pt')
                with (out/'history.csv').open('w',newline='',encoding='utf-8') as f:
                    w=csv.DictWriter(f,fieldnames=list(row)); w.writeheader(); w.writerows(history)
                print(f'{variant} seed={seed} epoch={epoch}/{args.epochs} train={row["train_accuracy"]:.4f} val={va:.4f} elapsed={time.perf_counter()-started:.1f}s',flush=True)
            train_seconds=time.perf_counter()-started
            ck=torch.load(out/'best.pt',map_location=device,weights_only=True); model.load_state_dict(ck['state_dict'])
            tl,ta,logits=evaluate(model,xt,yt)
            pred=logits.argmax(1).numpy(); labels=yt.cpu().numpy()
            cm=np.zeros((10,10),dtype=np.int64); np.add.at(cm,(labels,pred),1)
            precision=np.diag(cm)/np.maximum(cm.sum(0),1); recall=np.diag(cm)/cm.sum(1)
            f1=2*precision*recall/np.maximum(precision+recall,1e-12)
            np.savez_compressed(out/'predictions.npz',logits=logits.numpy(),labels=labels,confusion=cm)
            times=[]; model.eval(); sample=xt[:1]
            with torch.no_grad():
                for _ in range(50): model(sample)
                for _ in range(100):
                    if device=='cuda': torch.cuda.synchronize()
                    t=time.perf_counter(); model(sample)
                    if device=='cuda': torch.cuda.synchronize()
                    times.append((time.perf_counter()-t)*1000)
            metrics={'variant':variant,'seed':seed,'best_epoch':best_epoch,'val_accuracy':best,'test_accuracy':ta,'test_correct':int((labels==pred).sum()),'test_n':len(labels),'test_loss':tl,'macro_f1':float(f1.mean()),'params':params,'macs':macs,'flops_conv_linear':2*macs,'train_seconds':train_seconds,'latency_ms_median':float(np.median(times)),'latency_ms_p95':float(np.percentile(times,95)),'checkpoint_sha256':hashlib.sha256((out/'best.pt').read_bytes()).hexdigest()}
            write_json(out/'metrics.json',metrics); print(json.dumps(metrics),flush=True)

if __name__=='__main__': main()
