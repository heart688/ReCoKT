"""Causal continuous KeenKT inference with the existing C3 revisit feature rule."""
import math
import torch

@torch.no_grad()
def predict(model,head,skills,responses):
    if len(skills)!=len(responses) or len(skills)<2:raise ValueError('invalid trajectory')
    model.eval()
    if head is not None:head.eval()
    batch=dict(skills=torch.tensor([skills],dtype=torch.long),responses=torch.tensor([responses],dtype=torch.float32),mask=torch.ones((1,len(skills)),dtype=torch.bool))
    captured=[];hook=model.head.register_forward_pre_hook(lambda _module,args:captured.append(args[0]))
    try:logits,_,valid=model.next_response(batch)
    finally:hook.remove()
    if len(captured)!=1 or logits.shape!=valid.shape or captured[0].shape[:2]!=logits.shape or not bool(valid.all()):raise ValueError('feature contract mismatch')
    features=captured[0][0];base=logits[0]
    if head is None:return base.sigmoid().double().tolist()
    last={};counts={};anchors=[];gaps=[];priors=[]
    for i in range(1,len(skills)):
        k=skills[i];anchor=last.get(k,-1);anchors.append(anchor);gaps.append(i-anchor-1 if anchor>=0 else 0);priors.append(counts.get(k,0));last[k]=i;counts[k]=counts.get(k,0)+1
    n=len(anchors);idx=torch.tensor([max(0,a-1) for a in anchors],dtype=torch.long);has=torch.tensor([a>=1 for a in anchors],dtype=torch.bool)
    anchor_f=features[idx].masked_fill(~has[:,None],0.);anchor_z=base[idx].masked_fill(~has,0.)
    anchor_y=torch.tensor([responses[a] if a>=1 else 0. for a in anchors],dtype=features.dtype)
    gap=torch.tensor(gaps,dtype=features.dtype);prior=torch.tensor(priors,dtype=features.dtype)
    composed=torch.cat((features,anchor_f,base[:,None],anchor_z[:,None],anchor_y[:,None],gap.log1p()[:,None],prior.log1p()[:,None]),dim=1)
    if len(composed)!=n:raise ValueError('feature cardinality differs')
    corrected=base+torch.where(has & (gap>0),head(composed),torch.zeros_like(base))
    result=corrected.sigmoid().double().tolist()
    if any(not math.isfinite(p) or not 0<=p<=1 for p in result):raise ValueError('invalid prediction')
    return result
