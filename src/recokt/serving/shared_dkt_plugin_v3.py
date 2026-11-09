"""Prototype source-scoped DKT plugin; no raw records are stored here.

Archived dense checkpoints are factored relative to a provided common base.
Withdrawal below revokes active routes and cached models in this process; the
caller must separately handle durable copies, concurrent readers and state.
"""
from collections import OrderedDict
from pathlib import Path
import hashlib,json
import torch

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for x in iter(lambda:f.read(1024*1024),b''):h.update(x)
    return h.hexdigest()

def factorize(base,dense,rank=16,tolerance=2e-6):
    if set(base)!=set(dense):raise ValueError('parameter keys differ')
    result={};worst=0.
    for name,origin in base.items():
        target=dense[name]
        if target.shape!=origin.shape or target.dtype!=origin.dtype or not torch.is_floating_point(target):raise ValueError('incompatible state')
        difference=target.double()-origin.double()
        if target.ndim==2:
            u,s,v=torch.linalg.svd(difference,full_matrices=False);r=min(rank,len(s))
            left=(u[:,:r]*s[:r]).float();right=v[:r].float();result[name]={'kind':'factor','left':left,'right':right}
            estimate=origin+left@right
        elif target.ndim==1:
            delta=(target-origin).float();result[name]={'kind':'vector','delta':delta};estimate=origin+delta
        else:raise ValueError('unsupported parameter rank')
        err=float((estimate-target).abs().max());worst=max(worst,err)
        if err>tolerance:raise ValueError('rank approximation tolerance '+name+': '+str(err))
    return result,worst

def restore(base,delta):
    if set(base)!=set(delta):raise ValueError('factor keys differ')
    out={}
    for name,origin in base.items():
        row=delta[name]
        if row['kind']=='factor':out[name]=origin+row['left']@row['right']
        elif row['kind']=='vector':out[name]=origin+row['delta']
        else:raise ValueError('unknown adapter format')
    return out

class RecoDKTPlugin:
    """One base on disk; component models materialized on demand, LRU cached."""
    def __init__(self,folder,model_factory,head_factory,predictor=None,max_resident=3):
        self.folder=Path(folder);self.manifest=json.loads((self.folder/'manifest.json').read_text())
        if self.manifest['status']!='complete':raise ValueError('incomplete manifest')
        base_file=self.folder/self.manifest['base_file']
        if sha(base_file)!=self.manifest['base_sha256']:raise ValueError('base hash')
        self.base=torch.load(base_file,map_location='cpu',weights_only=True)['model_state']
        self.model_factory=model_factory;self.head_factory=head_factory;self.predictor=predictor;self.max_resident=max_resident
        if max_resident<1:raise ValueError('max_resident')
        self.active=dict(self.manifest['components']);self.resident=OrderedDict();self.withdrawn=set(self.manifest['initially_withdrawn']);self.sessions={}
        if any(self.withdrawn.intersection(v['members']) for v in self.active.values()):raise ValueError('initial route provenance')

    def component(self,name):
        if name not in self.active:raise ValueError('component is not active')
        if name in self.resident:
            self.resident.move_to_end(name);return self.resident[name]
        info=self.active[name];path=self.folder/info['file']
        if sha(path)!=info['sha256']:raise ValueError('adapter hash')
        data=torch.load(path,map_location='cpu',weights_only=True)
        if data['spec']['members']!=info['members'] or data['source_sha256']!=info['source_sha256']:raise ValueError('adapter identity')
        model=self.model_factory(self.manifest['n_skills'],data['spec']);model.load_state_dict(restore(self.base,data['delta']),strict=True);model.eval().requires_grad_(False)
        head=self.head_factory(data['head_seed'],data['feature_dim']);head.load_state_dict(data['head_state'],strict=True);head.eval().requires_grad_(False)
        self.resident[name]=(model,head);self.resident.move_to_end(name)
        while len(self.resident)>self.max_resident:self.resident.popitem(last=False)
        return model,head

    def route(self,school):
        school=str(school)
        if school in self.withdrawn:raise ValueError('withdrawn school cannot be served')
        eligible=[(name,v) for name,v in self.active.items() if school in v['members']]
        local=sorted(name for name,v in eligible if len(v['members'])==1)
        shared=sorted(name for name,v in eligible if len(v['members'])==2)
        if len(local)!=1 or len(shared)>2:raise ValueError('invalid school route')
        return [(local[0],1-.3*len(shared))]+[(name,.3) for name in shared]

    def predict_history(self,school,skills,responses):
        """Return causal predictions for positions 1..n-1 via supplied adapter."""
        if self.predictor is None:raise ValueError('a backbone prediction adapter is required')
        if len(skills)<2 or len(skills)!=len(responses):raise ValueError('invalid history')
        route=self.route(school);outputs=[]
        for name,weight in route:
            model,head=self.component(name)
            result=self.predictor(model,head,skills,responses)
            if len(result)!=len(skills)-1:raise ValueError('backbone prediction alignment')
            outputs.append((weight,result))
        return [sum(w*values[i] for w,values in outputs) for i in range(len(skills)-1)]

    def open_session(self,session_id,school,stream_factory):
        """Create plugin-owned per-component student states; caller retains history."""
        if session_id in self.sessions:raise ValueError('session already exists')
        route=self.route(school);states=[]
        try:
            for name,weight in route:
                model,head=self.component(name)
                states.append((name,weight,stream_factory(model,head)))
        except BaseException:
            for _,_,state in states:state.clear()
            raise
        self.sessions[session_id]={'school':str(school),'states':states}

    def observe_session(self,session_id,skill,response):
        if response not in (0,1):raise ValueError('response must be binary')
        for _,_,state in self.sessions[session_id]['states']:state.observe(skill,response)

    def predict_session(self,session_id,skill):
        return sum(weight*state.predict(skill) for _,weight,state in self.sessions[session_id]['states'])

    def close_session(self,session_id):
        entry=self.sessions.pop(session_id)
        for _,_,state in entry['states']:state.clear()

    def withdraw(self,school):
        """Drop all active references depending on school; no disk deletion."""
        school=str(school)
        if school in self.withdrawn:raise ValueError('already withdrawn')
        removed=sorted(name for name,v in self.active.items() if school in v['members'])
        if not removed:raise ValueError('school has no active components')
        invalidated=[];closed=[]
        for session_id,entry in list(self.sessions.items()):
            if entry['school']==school or any(name in removed for name,_,_ in entry['states']):
                (closed if entry['school']==school else invalidated).append(session_id)
                self.close_session(session_id)
        for name in removed:self.active.pop(name);self.resident.pop(name,None)
        self.withdrawn.add(school)
        if any(school in v['members'] for v in self.active.values()):raise RuntimeError('stale route')
        return {'school':school,'deactivated':removed,'sessions_replay_required':invalidated,'owner_sessions_closed':closed,
                'plugin_owned_states_cleared':True,'storage_deleted':False,'concurrency_managed':False}
