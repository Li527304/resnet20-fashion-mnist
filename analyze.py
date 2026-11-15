"""Generate every comparison figure/table directly from measured results."""
import os
from pathlib import Path
ROOT=Path(__file__).resolve().parent
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.mplconfig'))
import csv,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.datasets import FashionMNIST
from infer import CLASSES

NAMES={'base':'ResNet20','se':'SE-ResNet20'}
COLORS=['#18775d','#527bab']
plt.rcParams.update({'font.family':'Microsoft YaHei','axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white','axes.facecolor':'white','axes.labelcolor':'#30463e','text.color':'#20392f','axes.edgecolor':'#d8e1da','grid.color':'#e5eae5','font.size':11,'savefig.dpi':200})

def finish(fig,name):
    fig.savefig(ROOT/'figures'/name,bbox_inches='tight',facecolor='white');plt.close(fig)

def main():
    (ROOT/'figures').mkdir(exist_ok=True)
    cfg=json.loads((ROOT/'results/config.json').read_text(encoding='utf-8'))
    summary=[];all_metrics=[]
    for v in cfg['variants']:
        rows=[json.loads((ROOT/'results'/f'{v}_{seed}'/'metrics.json').read_text(encoding='utf-8')) for seed in cfg['seeds']]
        all_metrics.extend(rows)
        acc=np.array([r['test_accuracy'] for r in rows]);f1=np.array([r['macro_f1'] for r in rows])
        summary.append({'variant':v,'name':NAMES[v],'accuracy_mean':float(acc.mean()),'accuracy_std':float(acc.std(ddof=1)),'macro_f1_mean':float(f1.mean()),'params':rows[0]['params'],'macs':rows[0]['macs'],'latency_ms_mean':float(np.mean([r['latency_ms_median'] for r in rows])),'train_seconds_mean':float(np.mean([r['train_seconds'] for r in rows])),'runs':rows})
    (ROOT/'results/summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    reference=json.loads((ROOT/'results/reference_baselines.json').read_text(encoding='utf-8'))
    reference['base_mean_accuracy']=summary[0]['accuracy_mean']
    reference['difference_pp']=100*(summary[0]['accuracy_mean']-reference['test_accuracy'])
    (ROOT/'results/reference_comparison.json').write_text(json.dumps(reference,ensure_ascii=False,indent=2),encoding='utf-8')
    with (ROOT/'results/all_metrics.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(all_metrics[0]));w.writeheader();w.writerows(all_metrics)
    fig,axes=plt.subplots(1,3,figsize=(13.6,4.1),layout='constrained')
    for ax,key,title,scale in zip(axes,['accuracy_mean','params','macs'],['测试准确率 / %','参数量 / K','计算量 / M MACs'],[100,.001,.000001]):
        vals=[s[key]*scale for s in summary];bars=ax.bar(range(len(summary)),vals,color=COLORS,width=.57)
        if key=='accuracy_mean':
            ax.errorbar(range(len(summary)),vals,yerr=[s['accuracy_std']*100 for s in summary],fmt='none',color='#254536',capsize=5)
            ax.set_ylim(85,max(vals)+1.4)
        else:ax.set_ylim(0,max(vals)*1.18)
        for bar,val in zip(bars,vals):ax.text(bar.get_x()+bar.get_width()/2,val+.12 if key=='accuracy_mean' else val+max(vals)*.025,f'{val:.2f}',ha='center',va='bottom',weight='bold')
        ax.set_xticks(range(len(summary)),[s['name'] for s in summary],rotation=15);ax.set_title(title,pad=18,loc='left',weight='bold');ax.grid(axis='y',alpha=.65);ax.set_axisbelow(True)
    finish(fig,'comparison.png')
    fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
    for v,col in zip(cfg['variants'],COLORS):
        hs=[]
        for seed in cfg['seeds']:
            with (ROOT/'results'/f'{v}_{seed}'/'history.csv').open(encoding='utf-8') as f:hs.append(list(csv.DictReader(f)))
        for ax,key,title in zip(axes,['train_loss','val_accuracy'],['训练损失','验证准确率 / %']):
            a=np.array([[float(r[key]) for r in h] for h in hs])*(100 if key=='val_accuracy' else 1)
            xs=np.arange(1,a.shape[1]+1);ax.plot(xs,a.mean(0),label=NAMES[v],color=col,lw=2)
            ax.fill_between(xs,a.min(0),a.max(0),color=col,alpha=.11);ax.set_title(title,loc='left',weight='bold');ax.set_xlabel('训练轮次');ax.grid(alpha=.6);ax.legend(frameon=False)
    finish(fig,'curves.png')
    fig,axes=plt.subplots(1,len(summary),figsize=(11,4.5),layout='constrained')
    for v,ax in zip(cfg['variants'],axes):
        cm=np.load(ROOT/'results'/f'{v}_17'/'predictions.npz')['confusion']
        im=ax.imshow(cm/cm.sum(1,keepdims=True),vmin=0,vmax=1,cmap='Greens')
        for i in range(10):
            for j in range(10):
                if cm[i,j]>0:ax.text(j,i,str(cm[i,j]),ha='center',va='center',fontsize=6.5,color='white' if cm[i,j]>500 else '#244334')
        ax.set_xticks(range(10),CLASSES,rotation=65,fontsize=8);ax.set_yticks(range(10),CLASSES,fontsize=8);ax.set_title(NAMES[v]+' · seed17',loc='left',weight='bold');ax.set_xlabel('预测类别');ax.set_ylabel('真实类别')
    finish(fig,'confusion.png')
    cm=np.load(ROOT/'results/base_17/predictions.npz')['confusion']
    fig,ax=plt.subplots(figsize=(6.4,4.8),layout='constrained')
    ax.imshow(cm/cm.sum(1,keepdims=True),vmin=0,vmax=1,cmap='Greens')
    for i in range(10):
        for j in range(10):
            if cm[i,j]>0:ax.text(j,i,str(cm[i,j]),ha='center',va='center',fontsize=9,color='white' if cm[i,j]>500 else '#244334')
    ax.set_xticks(range(10),CLASSES,rotation=55,fontsize=10);ax.set_yticks(range(10),CLASSES,fontsize=10)
    ax.set_xlabel('预测类别');ax.set_ylabel('真实类别')
    finish(fig,'confusion_base.png')
    ds=FashionMNIST(ROOT/'data',train=False);p=np.load(ROOT/'results/base_17/predictions.npz');labels=p['labels'];logits=p['logits'];pred=logits.argmax(1);probs=np.exp(logits-logits.max(1,keepdims=True));probs/=probs.sum(1,keepdims=True)
    # Fixed rule: first six correct and first six incorrect test samples.
    ids=np.r_[np.where(labels==pred)[0][:6],np.where(labels!=pred)[0][:6]]
    fig,axes=plt.subplots(2,6,figsize=(13,4.8),layout='constrained')
    for ax,i in zip(axes.flat,ids):
        ax.imshow(np.array(ds[int(i)][0]),cmap='gray');ax.set_xticks([]);ax.set_yticks([])
        ax.set_title(f'#{i} 真实：{CLASSES[labels[i]]}\n预测：{CLASSES[pred[i]]} {probs[i,pred[i]]*100:.1f}%',fontsize=10,color='#18775d' if labels[i]==pred[i] else '#ae5736')
    finish(fig,'examples.png')
    ds[0][0].save(ROOT/'figures/sample.png')
    cm=p['confusion'];pairs=sorted([(int(cm[i,j]),CLASSES[i],CLASSES[j]) for i in range(10) for j in range(10) if i!=j],reverse=True)
    facts={'se_accuracy_delta_pp':100*(summary[1]['accuracy_mean']-summary[0]['accuracy_mean']),'base_seed17_top_confusions':pairs[:5],'example_ids':ids.tolist()}
    (ROOT/'results/facts.json').write_text(json.dumps(facts,ensure_ascii=False,indent=2),encoding='utf-8')
    recalls={}
    for v in cfg['variants']:
        recalls[v]=np.mean([np.diag(cm)/cm.sum(1) for seed in cfg['seeds'] for cm in [np.load(ROOT/'results'/f'{v}_{seed}'/'predictions.npz')['confusion']]],axis=0)
    per_class=[{'label':name,'base_recall':float(a),'se_recall':float(b),'delta_pp':float((b-a)*100)} for name,a,b in zip(CLASSES,recalls['base'],recalls['se'])]
    (ROOT/'results/per_class.json').write_text(json.dumps(per_class,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2));print(json.dumps(facts,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
