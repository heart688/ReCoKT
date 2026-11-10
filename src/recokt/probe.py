#!/usr/bin/env python3
"""Per-school component training in a multi-process federated simulation."""
import argparse
from copy import deepcopy
import hashlib
import io
import json
import multiprocessing as mp
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''


import torch
from .backbones import candidate as core
from .backbones import keenkt
from recokt_vendor import baseline_training as bt, data, federated as fed
from recokt_vendor.rollback_adapters import StagewiseDKT
from recokt_vendor.revisit_readout import RevisitHead, corrected_logits, extract_revisit_features

torch.set_num_threads(1)
ROUNDS = 6
BATCH = 32
LR = 0.008


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024), b''): h.update(chunk)
    return h.hexdigest()

def pack_state(state):
    # Keep tensors out of multiprocessing's MUSA-sensitive storage reducer.
    buf=io.BytesIO();torch.save(state,buf);return buf.getvalue()

def unpack_state(raw):
    if not isinstance(raw,bytes):raise TypeError('state payload must be bytes')
    return torch.load(io.BytesIO(raw),map_location='cpu',weights_only=True)

def guard():
    """Portable runner: resource scheduling is the caller's responsibility."""
    return None

def base_model(arch,n,spec):
    if arch=='dkt': return bt.make_model('dkt',n,1)
    seed=spec['backbone_seed']+1009*min(int(s) for s in spec['members'])
    return keenkt.make_model(n,seed) if arch=='keenkt' else bt.make_model('sakt',n,seed)

def adapter_model(arch,n,spec):
    base=base_model(arch,n,spec)
    if arch=='dkt': return StagewiseDKT(base,rank=16,stages=1,seed=811).set_active((0,)).train_stage(0)
    return core.GenericRank16Adapter(base,16,811)

def adapter_state(model,arch): return model.adapter_state(0) if arch=='dkt' else model.adapter_state()
def load_adapter(model,arch,state):
    if arch=='dkt': model.adapters[0].load_state_dict(state,strict=True)
    else: model.load_adapter_state(state)
def adapter_params(model,arch):
    return [p for p in model.parameters() if p.requires_grad] if arch=='dkt' else model.adapter_parameters()
def materialize(model,arch): return model.materialize((0,)) if arch=='dkt' else model.materialize()
def init_head(arch,head_seed,dim):
    return RevisitHead('full',seed=head_seed) if arch=='dkt' else core.GenericFullRevisitHead(dim,head_seed)

def school_worker(conn, arch, n, spec, school, entry):
    try:
        torch.set_num_threads(1)
        path=Path(entry['path'])
        if sha(path)!=entry['sha256']: raise ValueError('train SHA mismatch')
        rows=[]
        with path.open() as f:
            for line in f:
                if line.strip():
                    row=json.loads(line); data._validate_record(row,school,n); rows.append(row)
        if len(rows)!=entry['n_windows']: raise ValueError('window count mismatch')
        n_targets=sum(len(r['skills'])-1 for r in rows)
        if n_targets!=entry['n_targets'] or n_targets<1: raise ValueError('target count mismatch')
        model=adapter_model(arch,n,spec)
        conn.send({'event':'ready','school':school,'targets':n_targets,'windows':len(rows)})
        caches=None
        while True:
            msg=conn.recv(); op=msg['op']
            if op=='stop': return
            if op=='backbone':
                guard(); load_adapter(model,arch,unpack_state(msg['state'])); model.train()
                torch.manual_seed(spec['backbone_seed']+761923+msg['round']*101+int(school))
                optimizer=torch.optim.Adam(adapter_params(model,arch),lr=LR)
                seed=(fed.client_seed(spec['backbone_seed'],msg['round'],school) if arch=='dkt' else spec['backbone_seed']+msg['round'])
                steps=0
                for batch in data.make_loader(rows,BATCH,True,seed):
                    losses,valid=model.losses(batch)
                    if not bool(valid.any()): continue
                    loss=losses[valid].mean(); optimizer.zero_grad(set_to_none=True); loss.backward()
                    torch.nn.utils.clip_grad_norm_(adapter_params(model,arch),5.0); optimizer.step(); steps+=1
                conn.send({'event':'backbone','school':school,'count':n_targets,'steps':steps,'state':pack_state(adapter_state(model,arch))})
            elif op=='prepare_head':
                guard(); load_adapter(model,arch,unpack_state(msg['state'])); dense=materialize(model,arch).eval().requires_grad_(False)
                caches=[]; count=0; dim=None
                for batch in data.make_loader(rows,BATCH,False,0):
                    with torch.no_grad(): pack=(extract_revisit_features(dense,batch) if arch=='dkt' else core._revisit_pack(dense,batch))
                    caches.append(pack)
                    if arch=='dkt': count+=int((pack['valid_mask']&pack['revisit_mask']&(pack['gap']>0)).sum());dim=64
                    else: count+=int(pack['valid_mask'].sum());dim=pack['feature_dim']
                if count<1: raise ValueError('head has no legal targets')
                conn.send({'event':'head_ready','school':school,'count':count,'dim':dim})
            elif op=='head':
                guard(); h=init_head(arch,spec['head_seed'],msg['dim']).train();h.load_state_dict(unpack_state(msg['state']),strict=True)
                optimizer=torch.optim.Adam(h.parameters(),lr=LR);steps=0
                for pack in caches:
                    if arch=='dkt':
                        valid=pack['valid_mask']&pack['revisit_mask']&(pack['gap']>0)
                        if not bool(valid.any()):continue
                        logits=corrected_logits(h,pack)
                        loss=torch.nn.functional.binary_cross_entropy_with_logits(logits[valid],pack['targets'][valid])
                    else:
                        valid=pack['valid_mask']
                        if not bool(valid.any()):continue
                        logits=torch.where(pack['revisit_mask'],pack['base_logits']+h(pack['features']),pack['base_logits'])
                        loss=torch.nn.functional.binary_cross_entropy_with_logits(logits[valid],pack['targets'][valid])
                    optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(h.parameters(),5.0);optimizer.step();steps+=1
                conn.send({'event':'head','school':school,'steps':steps,'state':pack_state({k:v.detach().cpu().clone() for k,v in h.state_dict().items()})})
            else: raise ValueError('unsupported op')
    except BaseException as e:
        try: conn.send({'event':'failure','school':school,'error':repr(e),'traceback':traceback.format_exc()})
        except BaseException: pass
    finally: conn.close()

def receive(conn,expected):
    if not conn.poll(3600): raise TimeoutError('client response timeout')
    out=conn.recv()
    if out.get('event')=='failure':raise RuntimeError(out.get('error')+'\n'+out.get('traceback',''))
    if out.get('event')!=expected:raise ValueError('wrong client response')
    return out

def avg(states, weights):
    total=sum(weights)
    if total<=0:raise ValueError('bad weighting')
    keys=list(states[0])
    if any(list(s)!=keys for s in states):raise ValueError('state mismatch')
    return {k:sum(st[k]*(w/total) for st,w in zip(states,weights)) for k in keys}
