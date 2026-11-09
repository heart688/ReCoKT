"""Shared-base rank adapter DKT session; source-scoped components, no dense materialization."""
import torch
import torch.nn.functional as F

class FactorStream:
 def __init__(self,base,adapter,crc):
  self.base=base.eval();self.a=adapter;self.crc=crc.eval() if crc is not None else None;self.params=dict(base.named_parameters());self.gate_bias=self.params["lstm.bias_ih_l0"]+self.params["lstm.bias_hh_l0"]+adapter["vectors.lstm__bias_ih_l0"]+adapter["vectors.lstm__bias_hh_l0"];self.token_cache={};self.clear()
 def clear(self):
  self.h=None;self.c=None;self.current=None;self.latest={};self.counts={};self.pos=0
 def mat(self,key,x):
  name=key.replace('.','__');w=self.params[key]
  A=self.a['matrices.'+name+'.A'];B=self.a['matrices.'+name+'.B']
  return F.linear(x,w)+F.linear(F.linear(x,A),B)
 def bias(self,key):return self.params[key]+self.a['vectors.'+key.replace('.','__')]
 @torch.inference_mode()
 def observe(self,skill,response):
  token=1+int(skill)+int(response)*self.base.config.n_skills
  A=self.a['matrices.embedding__weight.A'];B=self.a['matrices.embedding__weight.B']
  x=self.token_cache.get(token)
  if x is None:x=self.base.embedding.weight[token]+B[token]@A;self.token_cache[token]=x
  if self.h is None:self.h=x.new_zeros(self.base.config.hidden_dim);self.c=x.new_zeros(self.base.config.hidden_dim)
  gates=self.mat('lstm.weight_ih_l0',x)+self.mat('lstm.weight_hh_l0',self.h)+self.gate_bias
  i,f,g,o=gates.chunk(4);self.c=torch.sigmoid(f)*self.c+torch.sigmoid(i)*torch.tanh(g);self.h=torch.sigmoid(o)*torch.tanh(self.c)
  self.current=self.h;self.latest[int(skill)]=(self.h.clone(),int(response),self.pos);self.counts[int(skill)]=self.counts.get(int(skill),0)+1;self.pos+=1
 def output(self,h,skill):
  skill=int(skill);a=self.a['matrices.head__weight.A'];b=self.a['matrices.head__weight.B']
  return torch.dot(self.base.head.weight[skill],h)+torch.dot(b[skill],a@h)+self.base.head.bias[skill]+self.a['vectors.head__bias'][skill]
 @torch.inference_mode()
 def predict(self,skill):
  if self.current is None:raise ValueError('empty history')
  skill=int(skill);z=self.output(self.current,skill);last=self.latest.get(skill)
  if last is not None and self.crc is not None:
   anchor,response,pos=last;gap=self.pos-pos-1
   if gap>0:
    anchor_z=self.output(anchor,skill);scalars=torch.stack((z,anchor_z,z.new_tensor(response),torch.log1p(z.new_tensor(gap)),torch.log1p(z.new_tensor(self.counts[skill]))))
    z=z+self.crc(torch.cat((self.current,anchor,scalars)).unsqueeze(0)).reshape(())
  return float(torch.sigmoid(z))
