"""Causalized skill-only KeenKT backbone, CPU, fixed-budget BCE pilot.

Based on the author's AAAI 2026 implementation (Apache-2.0).  Its temporal
SE gate is changed to a prefix mean and the fully masked first attention row is
zeroed to avoid future-answer leakage. Auxiliary diffusion and contrastive
losses are disabled for the shared BCE recipe; this is therefore an explicitly
modified backbone, not a reproduction of KeenKT.
"""
from dataclasses import dataclass,asdict
import torch
from torch import nn
from torch.nn import functional as F
from .impl.KeenKT import KEENKT

@dataclass(frozen=True)
class KeenKTConfig:
    n_skills:int
    hidden_dim:int=32
    window:int=50

class KeenKT(nn.Module):
    def __init__(self,config):
        super().__init__();self.config=config
        if any(type(x) is not int or x<=0 for x in asdict(config).values()):raise ValueError('positive dimensions required')
        self.backbone=KEENKT(n_question=config.n_skills,n_pid=0,emb_type='stoc_qid',
            use_uncertainty_aug=False,n_blocks=1,dropout=.1,d_model=32,d_ff=64,
            final_fc_dim=64,final_fc_dim2=32,seq_len=50,use_diffusion=False,use_CL=False,n_heads=2)
        layers=list(self.backbone.out.children());self.head=layers[-1];self.backbone.out=nn.Sequential(*layers[:-1])

    def _features(self,skills,responses):
        qm,qv,am,av=self.backbone.base_emb(skills,responses)
        mean,cov=self.backbone.model(qm,qv,am,av,self.backbone.atten_type)
        mean=self.backbone.se_gate(mean);cov=self.backbone.se_gate(cov)
        return self.backbone.out(torch.cat((mean,cov,qm,qv),dim=-1))[:,1:]

    def forward(self,skills,responses,mask):
        if skills.ndim!=2 or skills.shape!=responses.shape or skills.shape!=mask.shape or mask.dtype!=torch.bool:raise ValueError('invalid batch')
        if bool((mask[:,1:]&~mask[:,:-1]).any()):raise ValueError('padding must be a suffix')
        if bool(((skills[mask]<0)|(skills[mask]>=self.config.n_skills)).any()):raise ValueError('skill outside catalog')
        if bool(((responses[mask]!=0)&(responses[mask]!=1)).any()):raise ValueError('responses must be binary')
        if skills.shape[1]<2:return self.head.weight.new_zeros((skills.shape[0],0))
        s=torch.where(mask,skills,0).long();r=torch.where(mask,responses,0).long();window=self.config.window
        if s.shape[1]<=window:
            features=self._features(s,r)
        else:
            chunks=[self._features(s[:,:window],r[:,:window])]
            for start in range(window,s.shape[1],32):
                stop=min(start+32,s.shape[1]);count=stop-start;batch=s.shape[0]
                ss=torch.stack([s[:,i-window+1:i+1] for i in range(start,stop)],dim=1).reshape(batch*count,window)
                rr=torch.stack([r[:,i-window+1:i+1] for i in range(start,stop)],dim=1).reshape(batch*count,window)
                chunks.append(self._features(ss,rr)[:,-1,:].reshape(batch,count,-1))
            features=torch.cat(chunks,dim=1)
        return self.head(features).squeeze(-1).masked_fill(~(mask[:,:-1]&mask[:,1:]),0.)
    def next_response(self,batch):
        z=self(batch['skills'],batch['responses'],batch['mask'])
        return z,batch['responses'][:,1:].float(),batch['mask'][:,:-1]&batch['mask'][:,1:]
    def losses(self,batch):
        z,y,v=self.next_response(batch)
        return F.binary_cross_entropy_with_logits(z,torch.where(v,y,0.),reduction='none'),v
    def metadata(self):return {'architecture':'KeenKT_causal_bce_adaptation','history':'rolling50',**asdict(self.config)}

def make_model(n_skills,seed):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed);return KeenKT(KeenKTConfig(n_skills=n_skills))
