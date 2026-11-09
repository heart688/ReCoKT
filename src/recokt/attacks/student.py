"""Portable scorer adapted from attack_expanded_score_v3.py; see attacks/README.md."""
import argparse,hashlib,json,math,random,statistics,sys,time
from pathlib import Path
from .design import balanced_masks
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def auc(pairs):
 a=sorted(pairs,key=lambda z:z[1]);np=sum(y for y,_ in a);nn=len(a)-np
 if not np or not nn:return None
 ranks=0.;i=0
 while i<len(a):
  j=i+1
  while j<len(a) and a[j][1]==a[i][1]:j+=1
  ranks+=sum(y for y,_ in a[i:j])*(i+1+j)/2;i=j
 return (ranks-np*(np+1)/2)/(np*nn)
def normal(x,mu,var):return -.5*(math.log(2*math.pi*var)+(x-mu)**2/var)
def kde(x,ys):
 h=max(1e-3,1.06*statistics.stdev(ys)*len(ys)**(-.2));z=[-.5*((x-y)/h)**2 for y in ys];m=max(z);return m+math.log(sum(math.exp(t-m) for t in z)/len(z))-math.log(h)-.5*math.log(2*math.pi)
def values(pairs,threshold):
 pos=[s for y,s in pairs if y];neg=[s for y,s in pairs if not y]
 if not pos or not neg:return None
 return {'auc':auc(pairs),'tpr':sum(s>threshold for s in pos)/len(pos),'fpr':sum(s>threshold for s in neg)/len(neg),'positive':len(pos),'negative':len(neg)}
def score(ds, root):
 O=Path(root);pp=O/'plan_private.json';plan=json.loads(pp.read_text())
 if plan.get('dataset') != ds:raise ValueError('dataset mismatch')
 if len(plan['jobs']) != 88 or len(plan['candidates']) != 128 or len(plan['calibration']) != 64 or len(plan['evaluation']) != 64:raise ValueError('expected frozen 64/8/16 systems and 64/64 students')
 if set(plan['calibration']) & set(plan['evaluation']) or set(plan['candidates']) != set(plan['calibration']) | set(plan['evaluation']):raise ValueError('student split mismatch')
 masks=[j['masks'] for j in plan['jobs']]
 if masks != balanced_masks(ds,plan['calibration'],plan['evaluation']):raise ValueError('membership masks differ from frozen balanced design')
 scores=[]
 for k in range(88):
  d=O/'systems'/('shadow%02d'%k);c=json.loads((d/'query_commit.json').read_text())
  if c['status']!='complete' or c['plan_sha256']!=sha(pp) or c['queries_sha256']!=sha(d/'query_private.json'):raise ValueError('query proof')
  scores.append(json.loads((d/'query_private.json').read_text()))
 refs=range(64);residual=[];dist={}
 for s in plan['candidates']:
  ins=[i for i in refs if masks[i][s]];outs=[i for i in refs if not masks[i][s]]
  if len(ins)!=32 or len(outs)!=32:raise ValueError('ref balance')
  one=[scores[i]['Original'][s]['mean_true_label_logodds'] for i in ins];zero=[scores[i]['Original'][s]['mean_true_label_logodds'] for i in outs]
  dist[s]={'in':statistics.mean(one),'out':statistics.mean(zero),'u':[scores[i]['Post'][s]['mean_loglik'] for i in ins],'h':[scores[i]['Post'][s]['mean_loglik'] for i in outs],'r':[scores[i]['Clean'][s]['mean_loglik'] for i in refs]}
  if s in plan['calibration']:
   for z in (one,zero):
    m=statistics.mean(z);residual.extend((x-m)**2 for x in z)
 variance=max(1e-6,statistics.mean(residual))
 def get(s,i):
  d=dist[s];y=masks[i][s];o=scores[i]['Original'][s]['mean_true_label_logodds'];pl=scores[i]['Post'][s]['mean_true_label_logodds'];cl=scores[i]['Clean'][s]['mean_true_label_logodds'];post=scores[i]['Post'][s]['mean_loglik'];clean=scores[i]['Clean'][s]['mean_loglik']
  lira=lambda x:normal(x,d['in'],variance)-normal(x,d['out'],variance);privacy=lambda x:kde(x,d['u'])-kde(x,d['h']);eff=lambda x:kde(x,d['u'])-kde(x,d['r'])
  return {'Original_LiRA':(y,lira(o)),'Post_LiRA':(y,lira(pl)),'Clean_LiRA':(y,lira(cl)),'Post_RULI_privacy':(y,privacy(post)),'Clean_RULI_privacy_negative_control':(y,privacy(clean)),'RULI_efficacy_single_output':(y,eff(post if y else clean)),'Clean_RULI_efficacy_negative_control':(y,eff(clean))}
 cal={};test={}
 for subset,indices,target in ((plan['calibration'],range(64,72),cal),(plan['evaluation'],range(72,88),test)):
  for s in subset:
   for i in indices:
    for name,v in get(s,i).items():target.setdefault(name,{})[(s,i)]=v
 result={};rng=random.Random(20261107);eval_students=plan['evaluation'];eval_systems=list(range(72,88));pair_rng=random.Random(int(hashlib.sha256(('balanced-v1:'+ds+':evaluation:eval').encode()).hexdigest(),16));pair_order=list(eval_students);pair_rng.shuffle(pair_order);pairs=[pair_order[i:i+2] for i in range(0,len(pair_order),2)];draws=[([s for _ in pairs for s in rng.choice(pairs)],[rng.choice(eval_systems) for _ in eval_systems]) for _ in range(500)]
 for name,observations in test.items():
  negative=sorted((v for y,v in cal[name].values() if not y),reverse=True);allowed=int(.1*len(negative));threshold=negative[min(allowed,len(negative)-1)];overall=values(list(observations.values()),threshold);per_system={str(i):values([observations[(s,i)] for s in eval_students],threshold) for i in eval_systems};boot={'auc':[],'tpr':[],'fpr':[]}
  for students,systems in draws:
   result_boot=values([observations[(s,i)] for s in students for i in systems],threshold)
   if result_boot is not None:
    for field in boot:boot[field].append(result_boot[field])
  ci={}
  for field,x in boot.items():
   x.sort();ci[field]={'low':x[int(.025*(len(x)-1))],'high':x[int(.975*(len(x)-1))],'replicates':len(x)}
  result[name]={'overall':overall,'per_system':per_system,'requested_calibration_fpr':.1,'calibration_realized_fpr':sum(v>threshold for v in negative)/len(negative),'threshold':threshold,'pair_block_and_system_cluster_bootstrap_95pct':ci}
 delta=[abs(scores[i]['Post'][s]['mean_loglik']-scores[i]['Clean'][s]['mean_loglik']) for s in eval_students for i in eval_systems]
 out={'status':'complete','scope':'two datasets each DKT/F2770 one retained route; each result 64 distinct students x 16 systems; repeated observations correlated','dataset':ds,'attacks':result,'paired_post_clean_ll_absmax':max(delta),'student_count':64,'evaluation_system_count':16,'reference_system_count':64,'bootstrap_scope':'conditional on frozen references and calibration threshold; 32 student-pair blocks and 16 systems resampled independently','calibration_system_count':8,'plan_sha256':sha(pp),'scorer_sha256':sha(__file__),'completed_ns':time.time_ns()}
 return out
def main():
 a=argparse.ArgumentParser(description=__doc__);a.add_argument('--dataset',choices=['as09','as12'],required=True);a.add_argument('--input-dir',required=True);a.add_argument('--output',required=True);z=a.parse_args()
 out=score(z.dataset,z.input_dir)
 with Path(z.output).open('x') as f:json.dump(out,f,indent=2,allow_nan=False);f.write('\n')
if __name__=='__main__':main()
