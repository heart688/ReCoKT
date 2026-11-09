"""Train a frozen component bank and evaluate a withdrawal registry."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

from recokt_vendor.metrics import binary_metrics
from .train import file_sha, train_component
from .predict import load_component, predict_component, route


def read(path): return json.loads(Path(path).read_text())


def train_bank(args):
    manifest = read(args.manifest); index = read(args.index)['files']; plan = read(args.plan)
    if plan.get('status') != 'frozen' or plan.get('test_used') is not False:
        raise ValueError('expected a frozen, test-independent plan')
    frozen = plan['manifests'][args.dataset]
    if file_sha(args.manifest) != frozen['sha256'] or file_sha(args.index) != frozen['training_index_sha256']:
        raise ValueError('manifest or training index differs from the frozen plan')
    if len(manifest['schools']) != 20 or len(set(manifest['schools'])) != 20:
        raise ValueError('expected 20 distinct source schools')
    jobs = [j for j in plan['jobs'] if (j['dataset'],j['arch'],j['seed']) ==
            (args.dataset,args.arch,args.seed)]
    if len(jobs) != 40 or len({j['component'] for j in jobs}) != 40:
        raise ValueError('expected one complete 20-local, 20-pair bank')
    if sum(len(j['spec']['members']) == 1 for j in jobs) != 20:
        raise ValueError('expected 20 local components')
    schools = list(manifest['schools'])
    locals_ = [tuple(j['spec']['members']) for j in jobs if len(j['spec']['members']) == 1]
    pairs = [tuple(j['spec']['members']) for j in jobs if len(j['spec']['members']) == 2]
    if (set(locals_) != {(s,) for s in schools} or len(pairs) != 20 or
            set(pairs) != {(s, schools[(i+1) % 20]) for i,s in enumerate(schools)}):
        raise ValueError('component membership does not match the frozen ring')
    base = args.manifest.parent.resolve(); args.output.mkdir(parents=True, exist_ok=True)
    bank = {'architecture':args.arch,'dataset':args.dataset,'seed':args.seed,
            'plan_sha256':file_sha(args.plan),'manifest_sha256':file_sha(args.manifest),
            'index_sha256':file_sha(args.index),'components':{}}
    for job in jobs:
        name, spec = job['component'], job['spec']; folder = args.output/name
        if Path(name).name != name or name in ('', '.', '..'): raise ValueError('unsafe component name')
        entries = {}
        for school in spec['members']:
            rel = manifest['client_files'][school]['train']; path = (base/rel).resolve()
            if not path.is_relative_to(base): raise ValueError('training path escapes dataset')
            counts = manifest['encoding_counts'][school]['train']
            entries[school] = {'path':str(path),'sha256':index[rel]['sha256'],
                               'n_targets':counts['emitted_targets'],'n_windows':counts['n_windows']}
        if folder.exists():
            receipt = read(folder/'receipt.json')
            if (receipt['status'] != 'complete' or receipt['spec'] != spec or
                receipt['architecture'] != args.arch or
                receipt['source_train_sha256'] != {s:entries[s]['sha256'] for s in spec['members']} or
                receipt['component_sha256'] != file_sha(folder/'component.pt') or
                receipt['original_sha256'] != file_sha(folder/'adapter_original.pt')):
                raise ValueError(f'existing component does not match frozen input: {name}')
        else:
            receipt = train_component(args.arch, manifest['n_skills'], spec, entries, folder)
        bank['components'][name] = {'members':spec['members'],'file':f'{name}/component.pt',
                                    'sha256':receipt['component_sha256']}
        print(f'complete {name} ({len(bank["components"])}/40)', flush=True)
    dest = args.output/'bank.json'
    if dest.exists():
        if read(dest) != bank: raise ValueError('existing bank manifest differs')
    else:
        with dest.open('x') as f: json.dump(bank,f,indent=2); f.write('\n')


def evaluate(args):
    bank=read(args.bank); registry=read(args.registry); root=args.bank.parent.resolve()
    if str(registry['withdrawn']) != args.withdrawn or registry.get('split') not in ('val', 'test'):
        raise ValueError('registry request or split mismatch')
    components={}
    for name,x in bank['components'].items():
        p=(root/x['file']).resolve()
        if not p.is_relative_to(root) or file_sha(p)!=x['sha256']:
            raise ValueError('component path or hash mismatch')
        components[name]={'members':x['members'],'path':p}
    if str(args.withdrawn) not in {s for c in components.values() for s in c['members']}:
        raise ValueError('withdrawn school not present')
    all_schools={s for c in components.values() for s in c['members']}
    if set(map(str, registry['schools'])) != all_schools-{args.withdrawn}:
        raise ValueError('registry retained-school set mismatch')
    targets={t['sample_id']:t for t in registry['targets']}
    if len(targets)!=len(registry['targets']): raise ValueError('duplicate registry target')
    cache={}; observed={}; byschool=defaultdict(list)
    for tr in registry['trajectories']:
        school=str(tr['school_id']); sk=tr['skills']; answers=tr['responses']; ids=tr['sample_ids']
        if school==args.withdrawn or len(sk)!=len(answers) or len(sk)!=len(ids) or ids[0]!='':
            raise ValueError('invalid or withdrawn trajectory')
        probs=[]
        for name,weight in route(components,school,[args.withdrawn]):
            if name not in cache: cache[name]=load_component(components[name]['path'])
            z, loaded=cache[name]
            if z['architecture']!=bank['architecture'] or z['spec']['members']!=components[name]['members']:
                raise ValueError('component identity mismatch')
            probs.append((weight,predict_component(bank['architecture'],loaded,sk,answers)))
        for i,sid in enumerate(ids[1:],1):
            t=targets[sid]
            if sid in observed or (str(t['school_id']),t['student_id'],t['segment_id'],t['event_index'],t['skill_id'],t['target']) != (school,tr['student_id'],tr['segment_id'],i,sk[i],answers[i]):
                raise ValueError('target identity mismatch')
            p=sum(w*ps[i-1] for w,ps in probs);observed[sid]=True
            byschool[school].append((t['student_id'],int(answers[i]),p))
    if set(observed)!=set(targets) or set(byschool)!=set(map(str,registry['schools'])):
        raise ValueError('target or school coverage mismatch')
    per={}
    for school,rows in byschool.items():
        m=binary_metrics([y for _,y,_ in rows],[p for _,_,p in rows])
        groups=defaultdict(list)
        for student,y,p in rows: groups[student].append((p-y)**2)
        m['student_balanced_brier']=sum(sum(v)/len(v) for v in groups.values())/len(groups)
        per[school]=m
    aucs=[v['auc'] for v in per.values()]
    result={'dataset':bank['dataset'],'architecture':bank['architecture'],'seed':bank['seed'],
            'withdrawn':args.withdrawn,'schools':len(per),'targets':len(observed),
            'bank_sha256':file_sha(args.bank),'registry_sha256':file_sha(args.registry),
            'auc':None if None in aucs else sum(aucs)/len(aucs),
            'student_balanced_brier':sum(v['student_balanced_brier'] for v in per.values())/len(per),
            'per_school':per}
    with args.output.open('x') as f: json.dump(result,f,indent=2,allow_nan=False);f.write('\n')


def main():
    p=argparse.ArgumentParser(); sub=p.add_subparsers(dest='command',required=True)
    t=sub.add_parser('train');t.add_argument('--plan',type=Path,required=True);t.add_argument('--manifest',type=Path,required=True);t.add_argument('--index',type=Path,required=True);t.add_argument('--dataset',required=True);t.add_argument('--arch',choices=['dkt','sakt','keenkt'],required=True);t.add_argument('--seed',type=int,required=True);t.add_argument('--output',type=Path,required=True)
    e=sub.add_parser('evaluate');e.add_argument('--bank',type=Path,required=True);e.add_argument('--registry',type=Path,required=True);e.add_argument('--withdrawn',required=True);e.add_argument('--output',type=Path,required=True)
    a=p.parse_args(); train_bank(a) if a.command=='train' else evaluate(a)


if __name__=='__main__':main()
