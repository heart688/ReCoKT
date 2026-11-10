"""Coordinate component training across separate school processes."""
import argparse
import hashlib
import json
import multiprocessing as mp
from pathlib import Path

import torch
from . import probe


def file_sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def state_sha(state):
    h = hashlib.sha256()
    for name in sorted(state):
        t = state[name].detach().cpu().contiguous()
        h.update(name.encode()); h.update(str(t.dtype).encode())
        h.update(str(tuple(t.shape)).encode()); h.update(t.numpy().tobytes())
    return h.hexdigest()


def train_component(arch, n_skills, spec, entries, output):
    """Train a fresh local or pair component with the frozen 6+6 recipe."""
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    if arch not in ('dkt', 'sakt', 'keenkt'):
        raise ValueError('unknown backbone')
    members = spec['members']
    if len(members) not in (1, 2) or len(set(members)) != len(members):
        raise ValueError('one or two distinct source schools required')
    if set(entries) != set(members):
        raise ValueError('client data must match component membership')
    output.mkdir(parents=True)
    torch.set_num_threads(1)
    ctx = mp.get_context('spawn'); workers = []; conns = []
    try:
        for school in members:
            parent, child = ctx.Pipe()
            worker = ctx.Process(target=probe.school_worker,
                                 args=(child, arch, n_skills, spec, school, entries[school]))
            worker.start(); child.close(); workers.append(worker); conns.append(parent)
        ready = [probe.receive(c, 'ready') for c in conns]
        model = probe.adapter_model(arch, n_skills, spec)
        base_hash = state_sha(probe.base_model(arch, n_skills, spec).state_dict())
        state = probe.adapter_state(model, arch)
        for r in range(probe.ROUNDS):
            wire = probe.pack_state(state)
            for c in conns: c.send({'op': 'backbone', 'round': r, 'state': wire})
            updates = [probe.receive(c, 'backbone') for c in conns]
            state = probe.avg([probe.unpack_state(x['state']) for x in updates],
                              [x['targets'] for x in ready])
        probe.load_adapter(model, arch, state)
        wire = probe.pack_state(state)
        for c in conns: c.send({'op': 'prepare_head', 'state': wire})
        ready_head = [probe.receive(c, 'head_ready') for c in conns]
        dims = {x['dim'] for x in ready_head}
        if len(dims) != 1: raise ValueError('inconsistent head dimensions')
        dim = dims.pop(); head = probe.init_head(arch, spec['head_seed'], dim)
        hs = head.state_dict()
        for r in range(probe.ROUNDS):
            wire = probe.pack_state(hs)
            for c in conns: c.send({'op': 'head', 'round': r, 'dim': dim, 'state': wire})
            updates = [probe.receive(c, 'head') for c in conns]
            hs = probe.avg([probe.unpack_state(x['state']) for x in updates],
                           [x['count'] for x in ready_head])
        packed = {}
        for key, value in state.items():
            is_factor = ((arch == 'dkt' and key.startswith('matrices.')) or
                         (arch != 'dkt' and (key.endswith('.A') or key.endswith('.B'))))
            if is_factor:
                scale = float(value.abs().max()) / 127 if float(value.abs().max()) else 1.
                packed[key] = {'q': torch.round(value / scale).clamp(-127, 127).to(torch.int8), 'scale': scale}
            else:
                packed[key] = {'f': value}
        common = {'architecture': arch, 'n_skills': n_skills, 'spec': spec,
                  'head_state': hs, 'head_seed': spec['head_seed'], 'feature_dim': dim,
                  'base_state_sha256': base_hash}
        torch.save({**common, 'adapter_state': state}, output / 'adapter_original.pt')
        torch.save({**common, 'adapter': packed}, output / 'component.pt')
        for c in conns: c.send({'op': 'stop'})
        for worker in workers:
            worker.join(timeout=30)
            if worker.is_alive() or worker.exitcode != 0:
                raise RuntimeError('school worker did not finish cleanly')
        receipt = {'status': 'complete', 'architecture': arch, 'spec': spec,
                   'source_train_sha256': {s: entries[s]['sha256'] for s in members},
                   'component_sha256': file_sha(output/'component.pt'),
                   'original_sha256': file_sha(output/'adapter_original.pt'),
                   'rounds_backbone': probe.ROUNDS, 'rounds_head': probe.ROUNDS}
        (output/'receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')
        return receipt
    finally:
        for c in conns:
            try: c.send({'op': 'stop'})
            except (BrokenPipeError, EOFError, OSError): pass
            c.close()
        for worker in workers:
            worker.join(timeout=5)
            if worker.is_alive(): worker.terminate(); worker.join(timeout=5)


def main():
    cli = argparse.ArgumentParser(description='Train one frozen ReCoKT component')
    cli.add_argument('--arch', required=True, choices=['dkt','sakt','keenkt'])
    cli.add_argument('--manifest', type=Path, required=True, help='Prepared dataset manifest.json')
    cli.add_argument('--index', type=Path, required=True, help='Prepared dataset sha256.json')
    cli.add_argument('--spec', type=Path, required=True, help='Frozen component spec JSON')
    cli.add_argument('--output', type=Path, required=True, help='New output directory')
    args = cli.parse_args()
    manifest = json.loads(args.manifest.read_text()); spec = json.loads(args.spec.read_text())
    index = json.loads(args.index.read_text())['files']; root = args.manifest.parent.resolve()
    entries = {}
    for school in spec['members']:
        rel = manifest['client_files'][school]['train']; path = (root/rel).resolve()
        if not path.is_relative_to(root): raise ValueError('training path escapes manifest root')
        counts = manifest['encoding_counts'][school]['train']
        entries[school] = {'path': str(path), 'sha256': index[rel]['sha256'],
                           'n_targets': counts['emitted_targets'], 'n_windows': counts['n_windows']}
    print(json.dumps(train_component(args.arch, manifest['n_skills'], spec, entries, args.output)))


if __name__ == '__main__': main()
