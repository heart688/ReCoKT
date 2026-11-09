"""Prototype KT plugin using source-specific deterministic base recipes.

This migrates independently trained adapters without claiming a single shared
backbone. The recipe/hash must match at load time; no raw training records are
used. Runtime revocation is inherited and does not delete durable files.
"""
from collections import OrderedDict
from pathlib import Path
import hashlib,json
import torch
from shared_dkt_plugin_v3 import RecoDKTPlugin,sha

def state_sha(state):
    h=hashlib.sha256()
    for name in sorted(state):
        t=state[name].detach().cpu().contiguous()
        h.update(name.encode());h.update(str(t.dtype).encode());h.update(str(tuple(t.shape)).encode());h.update(t.numpy().tobytes())
    return h.hexdigest()

def factorize(base,dense,rank=16,tolerance=2e-6):
    if set(base)!=set(dense):raise ValueError('state keys')
    result={};worst=0.
    for name,origin in base.items():
        target=dense[name]
        if target.shape!=origin.shape or target.dtype!=origin.dtype:raise ValueError('state shape')
        if torch.equal(target,origin):row={'kind':'same'};estimate=origin
        elif not torch.is_floating_point(target):raise ValueError('nonfloating changed')
        elif target.ndim==2 and min(target.shape)>rank and rank*sum(target.shape)<target.numel():
            d=target.double()-origin.double();u,s,v=torch.linalg.svd(d,full_matrices=False)
            left=(u[:,:rank]*s[:rank]).float();right=v[:rank].float();row={'kind':'factor','left':left,'right':right};estimate=origin+left@right
        else:
            delta=(target-origin).float();row={'kind':'delta','delta':delta};estimate=origin+delta
        err=float((estimate-target).abs().max());worst=max(worst,err)
        if err>tolerance:raise ValueError('rank reconstruction '+name+': '+str(err))
        result[name]=row
    return result,worst

def restore(base,delta):
    if set(base)!=set(delta):raise ValueError('state keys')
    out={}
    for name,origin in base.items():
        row=delta[name]
        if row['kind']=='same':out[name]=origin
        elif row['kind']=='factor':out[name]=origin+row['left']@row['right']
        elif row['kind']=='delta':out[name]=origin+row['delta']
        else:raise ValueError('unknown factor')
    return out

class RecipeKTPlugin(RecoDKTPlugin):
    def __init__(self,folder,model_factory,head_factory,predictor=None,max_resident=3):
        self.folder=Path(folder);self.manifest=json.loads((self.folder/'manifest.json').read_text())
        if self.manifest['status']!='complete' or self.manifest['torch_version']!=str(torch.__version__):raise ValueError('runtime or manifest mismatch')
        self.model_factory=model_factory;self.head_factory=head_factory;self.predictor=predictor;self.max_resident=max_resident
        if max_resident<1:raise ValueError('max_resident')
        self.active=dict(self.manifest['components']);self.resident=OrderedDict();self.withdrawn=set(self.manifest['initially_withdrawn']);self.sessions={}
        if any(self.withdrawn.intersection(v['members']) for v in self.active.values()):raise ValueError('initial route provenance')

    def component(self,name):
        if name not in self.active:raise ValueError('component is not active')
        if name in self.resident:self.resident.move_to_end(name);return self.resident[name]
        info=self.active[name];path=self.folder/info['file']
        if sha(path)!=info['sha256']:raise ValueError('adapter hash')
        data=torch.load(path,map_location='cpu',weights_only=True)
        if data['spec']['members']!=info['members'] or data['source_sha256']!=info['source_sha256']:raise ValueError('adapter identity')
        model=self.model_factory(self.manifest['n_skills'],data['spec']);base=model.state_dict()
        if state_sha(base)!=data['base_state_sha256']:raise ValueError('base recipe changed')
        model.load_state_dict(restore(base,data['delta']),strict=True);model.eval().requires_grad_(False)
        head=self.head_factory(data['head_seed'],data['feature_dim']);head.load_state_dict(data['head_state'],strict=True);head.eval().requires_grad_(False)
        self.resident[name]=(model,head);self.resident.move_to_end(name)
        while len(self.resident)>self.max_resident:self.resident.popitem(last=False)
        return model,head
