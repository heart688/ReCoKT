"""Component loading, causal predictions, and school-withdrawal routing."""
from pathlib import Path
import math
import torch
from . import probe
from .train import state_sha
from .factor_stream import FactorStream
from .backbones import candidate, keen_predict


def load_component(path):
    z = torch.load(Path(path), map_location='cpu', weights_only=True)
    arch, spec = z['architecture'], z['spec']
    base = probe.base_model(arch, z['n_skills'], spec).eval()
    if state_sha(base.state_dict()) != z['base_state_sha256']:
        raise ValueError('component base fingerprint mismatch')
    factors = {k: v['q'].float()*v['scale'] if 'q' in v else v['f']
               for k, v in z['adapter'].items()}
    head = probe.init_head(arch, z['head_seed'], z['feature_dim'])
    head.load_state_dict(z['head_state']); head.eval()
    if arch == 'dkt': return z, (base, factors, head)
    model = probe.adapter_model(arch, z['n_skills'], spec)
    probe.load_adapter(model, arch, factors)
    return z, (probe.materialize(model, arch).eval().requires_grad_(False), head)


@torch.inference_mode()
def predict_component(arch, loaded, skills, responses):
    """Output p(response[t] | history before t) for t=1..len(skills)-1."""
    if len(skills) != len(responses) or len(skills) < 2:
        raise ValueError('trajectory must contain history and a target')
    if arch == 'dkt':
        stream = FactorStream(*loaded); out = []
        for i in range(1, len(skills)):
            stream.observe(skills[i-1], responses[i-1]); out.append(stream.predict(skills[i]))
        return out
    model, head = loaded
    if arch == 'keenkt':
        return keen_predict.predict(model, head, skills, responses)
    if arch != 'sakt': raise ValueError('unknown backbone')
    # The reference SAKT query uses the preceding 50 observations per target.
    features, logits = [], []
    for i in range(1, len(skills)):
        start = max(0, i-50)
        sk = torch.tensor([skills[start:i+1]], dtype=torch.long)
        # The current answer is masked from the next-response query.
        rs = torch.tensor([responses[start:i]+[0]], dtype=torch.float32)
        batch = {'skills': sk, 'responses': rs, 'mask': torch.ones_like(sk, dtype=torch.bool)}
        f, z, _, valid = candidate._capture_features(model, batch)
        if not bool(valid[0, -1]): raise ValueError('invalid SAKT target')
        features.append(f[0, -1]); logits.append(z[0, -1])
    features = torch.stack(features); logits = torch.stack(logits)
    latest, counts, out = {}, {}, []
    for i in range(1, len(skills)):
        skill = skills[i]; anchor = latest.get(skill); count = counts.get(skill, 0)
        z = logits[i-1]
        if anchor is not None and i-anchor-1 > 0:
            a = anchor-1
            scalar = torch.stack((z, logits[a], z.new_tensor(responses[anchor]),
                                  z.new_tensor(math.log1p(i-anchor-1)), z.new_tensor(math.log1p(count))))
            packed = torch.cat((features[i-1], features[a], scalar)).unsqueeze(0)
            z = z + head(packed).reshape(())
        out.append(float(z.sigmoid()))
        latest[skill] = i; counts[skill] = count + 1
    return out


def route(components, school, withdrawn=()):
    school = str(school); removed = set(map(str, withdrawn))
    if school in removed: raise ValueError('withdrawn school cannot receive predictions')
    active = [(name, item) for name, item in components.items()
              if school in item['members'] and not removed.intersection(item['members'])]
    local = [(n, x) for n, x in active if len(x['members']) == 1]
    pairs = [(n, x) for n, x in active if len(x['members']) == 2]
    if len(local) != 1 or len(pairs) > 2: raise ValueError('invalid local/pair route')
    return [(local[0][0], 1. - .3*len(pairs))] + [(n, .3) for n, _ in pairs]


def predict_route(components, school, skills, responses, withdrawn=()):
    selected = route(components, school, withdrawn)
    probabilities = []
    for name, weight in selected:
        z, loaded = load_component(components[name]['path'])
        if z['spec']['members'] != components[name]['members']:
            raise ValueError('component membership mismatch')
        probabilities.append((weight, predict_component(z['architecture'], loaded, skills, responses)))
    return [sum(w*p[i] for w,p in probabilities) for i in range(len(skills)-1)]
