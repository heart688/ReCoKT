"""Small synthetic end-to-end checks; no experimental data or checkpoints."""
import hashlib
import json
from pathlib import Path
import tempfile

import torch
from recokt.train import train_component, file_sha
from recokt.predict import load_component, predict_component, route
from recokt.serving.export import export_bank
from recokt.serving.managed_quant_serving_factory_v1 import make_session_factory
from recokt.serving.selective_state_policy_v2 import (SelectiveManagedServingBundle,
    exact_stream_type_guard, make_quantized_dkt_factory)
from recokt.factor_stream import FactorStream


def main():
    torch.set_num_threads(1)
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp); entries = {}
        for school in ['101', '102']:
            rows = []
            for student in range(2):
                skills = [0,1,2,0,1,2,0,1,2,0]
                responses = [(i+student+int(school))%2 for i in range(len(skills))]
                seen = {}; history = []
                for skill, answer in zip(skills, responses):
                    old = seen.get(skill, [])
                    history.append(''.join(map(str,old[-2:])) if len(old)>=2 else 'short')
                    seen.setdefault(skill,[]).append(answer)
                rows.append({'school_id':school,
                             'student_id':hashlib.sha256(f'{school}:{student}'.encode()).hexdigest(),
                             'skills':skills,'responses':responses,'history':history,
                             'sample_ids':['']+[f'{school}-{student}-{i}' for i in range(1,len(skills))]})
            path = root/f'{school}.jsonl'
            path.write_text(''.join(json.dumps(x)+'\n' for x in rows))
            entries[school] = {'path':str(path),'sha256':file_sha(path),
                               'n_windows':len(rows),'n_targets':sum(len(x['skills'])-1 for x in rows)}
        spec={'kind':'pair16','members':['101','102'],'original_index':0,'backbone_seed':1,'head_seed':23001}
        for arch in ('dkt','sakt','keenkt'):
            output = root/arch
            receipt = train_component(arch, 4, spec, entries, output)
            assert receipt['rounds_backbone'] == receipt['rounds_head'] == 6
            meta, loaded = load_component(output/'component.pt')
            sk = [0,1,2,0,1,2,0]; rs = [1,0,1,0,1,0,1]
            p = predict_component(arch, loaded, sk, rs)
            altered = rs.copy(); altered[4:] = [1-x for x in altered[4:]]
            q = predict_component(arch, loaded, sk, altered)
            assert len(p) == len(sk)-1 and all(0 <= x <= 1 for x in p)
            assert max(abs(x-y) for x,y in zip(p[:4],q[:4])) < 1e-6, arch
            print(f'{arch}: 6+6 synthetic pair training, load and causal prediction OK', flush=True)
        local_spec={'kind':'local_rank16','members':['101'],'original_index':0,
                    'backbone_seed':1,'head_seed':23001}
        train_component('dkt',4,local_spec,{'101':entries['101']},root/'local')
        component_files={'101':('local/component.pt',['101']),
                         'e0':('dkt/component.pt',['101','102'])}
        bank={'architecture':'dkt','components':{name:{'file':path,'members':members,
                    'sha256':file_sha(root/path)} for name,(path,members) in component_files.items()}}
        bank_file=root/'bank.json';bank_file.write_text(json.dumps(bank))
        bundle=export_bank(bank_file,root/'serving')
        factory=make_quantized_dkt_factory(state_reuse_guard=exact_stream_type_guard(FactorStream))
        with SelectiveManagedServingBundle(bundle,factory) as service:
            service.open_session('s','101',make_session_factory('dkt'))
            for k,r in [(0,1),(1,0),(0,1)]: service.observe_session('s',k,r)
            result=service.withdraw('102')
            assert result['sessions_state_preserved']==['s']
            prediction=service.predict_session('s',2)
            _, local=load_component(root/'local/component.pt')
            expected=predict_component('dkt',local,[0,1,0,2],[1,0,1,0])[-1]
            assert abs(prediction-expected)<1e-7
            assert (root/'dkt/component.pt').exists(), 'source component must remain'
        print('DKT managed export and selective withdrawal OK', flush=True)
        components = {'a':{'members':['101']}, 'e0':{'members':['101','102']},
                      'e2':{'members':['103','101']}}
        assert route(components,'101') == [('a',.4),('e0',.3),('e2',.3)]
        assert route(components,'101',['102']) == [('a',.7),('e2',.3)]
        try: route(components,'102',['102'])
        except ValueError: pass
        else: raise AssertionError('withdrawn school accepted')
        print('withdrawal route and weight return OK', flush=True)


if __name__ == '__main__': main()
