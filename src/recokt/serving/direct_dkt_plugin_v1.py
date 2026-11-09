"""Load source-scoped DKT adapters saved directly by federated training.

The caller supplies backbone construction and adapter materialization hooks;
this module stores no school records. Revocation is in-process only.
"""
from collections import OrderedDict
from pathlib import Path
import json
import torch
from .shared_dkt_plugin_v3 import RecoDKTPlugin,sha
from .recipe_kt_plugin_v1 import state_sha

class DirectDKTPlugin(RecoDKTPlugin):
    def __init__(self,folder,adapter_factory,load_adapter,materialize,head_factory,predictor=None,max_resident=3):
        self.folder=Path(folder);self.manifest=json.loads((self.folder/'manifest.json').read_text())
        if self.manifest['status']!='complete' or self.manifest['torch_version']!=str(torch.__version__):raise ValueError('runtime or manifest mismatch')
        self.adapter_factory=adapter_factory;self.load_adapter=load_adapter;self.materialize=materialize;self.head_factory=head_factory;self.predictor=predictor;self.max_resident=max_resident
        if max_resident<1:raise ValueError('max resident')
        self.active=dict(self.manifest['components']);self.resident=OrderedDict();self.withdrawn=set(self.manifest['initially_withdrawn']);self.sessions={}
        if any(self.withdrawn.intersection(v['members']) for v in self.active.values()):raise ValueError('initial route provenance')

    def component(self,name):
        if name not in self.active:raise ValueError('component is not active')
        if name in self.resident:self.resident.move_to_end(name);return self.resident[name]
        info=self.active[name];file=self.folder/info['file']
        if sha(file)!=info['sha256']:raise ValueError('adapter hash')
        data=torch.load(file,map_location='cpu',weights_only=True)
        if data['architecture']!='dkt' or data['spec']['members']!=info['members'] or data['base_state_sha256']!=self.manifest['base_state_sha256']:raise ValueError('adapter identity')
        adapter=self.adapter_factory(data['n_skills'],data['spec'])
        if state_sha(adapter.base.state_dict())!=data['base_state_sha256']:raise ValueError('backbone fingerprint mismatch')
        self.load_adapter(adapter,data['adapter_state'])
        model=self.materialize(adapter).eval().requires_grad_(False)
        head=self.head_factory(data['head_seed'],data['feature_dim']);head.load_state_dict(data['head_state'],strict=True);head.eval().requires_grad_(False)
        self.resident[name]=(model,head);self.resident.move_to_end(name)
        while len(self.resident)>self.max_resident:self.resident.popitem(last=False)
        return model,head
