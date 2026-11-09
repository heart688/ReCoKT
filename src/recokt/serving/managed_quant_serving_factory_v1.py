"""Portable adapter for the paper's directly trained quantized components."""
from collections import OrderedDict
import json
from pathlib import Path
from ..predict import load_component, predict_component
from ..factor_stream import FactorStream
from ..train import file_sha, state_sha
from .shared_dkt_plugin_v3 import RecoDKTPlugin


class QuantizedProductionAdapter(RecoDKTPlugin):
    def __init__(self, folder, arch):
        self.folder = Path(folder); self.arch = arch
        self.manifest = json.loads((self.folder/'manifest.json').read_text())
        if self.manifest['status'] != 'complete' or self.manifest['architecture'] != arch:
            raise ValueError('managed manifest identity')
        self.active = dict(self.manifest['components'])
        self.withdrawn = set(self.manifest['initially_withdrawn'])
        self.resident, self.sessions = OrderedDict(), {}
        self.max_resident = 3; self.base = None
        self.predictor = lambda loaded, head, skills, answers: predict_component(arch, loaded, skills, answers)

    def component(self, name):
        if name not in self.active: raise ValueError('component inactive')
        if name in self.resident:
            self.resident.move_to_end(name); return self.resident[name]
        info = self.active[name]; path = self.folder/info['file']
        if file_sha(path) != info['sha256']: raise ValueError('component SHA mismatch')
        z, loaded = load_component(path)
        if z['spec']['members'] != info['members'] or z['architecture'] != self.arch:
            raise ValueError('component provenance mismatch')
        if self.arch == 'dkt':
            if self.base is None: self.base = loaded[0]
            if state_sha(self.base.state_dict()) != z['base_state_sha256']:
                raise ValueError('DKT shared base fingerprint mismatch')
            loaded = (self.base, loaded[1], loaded[2])
        self.resident[name] = (loaded, loaded[-1]); self.resident.move_to_end(name)
        while len(self.resident) > self.max_resident: self.resident.popitem(last=False)
        return self.resident[name]


class PrefixSession:
    def __init__(self, loaded, head, arch):
        self.loaded, self.arch = loaded, arch; self.skills, self.answers = [], []
    def observe(self, skill, answer): self.skills.append(skill); self.answers.append(answer)
    def predict(self, skill):
        return predict_component(self.arch, self.loaded, self.skills+[skill], self.answers+[0])[-1]
    def clear(self): self.skills.clear(); self.answers.clear()


def make_factory(arch): return lambda folder: QuantizedProductionAdapter(folder, arch)
def make_session_factory(arch):
    if arch == 'dkt': return lambda loaded, head: FactorStream(*loaded)
    return lambda loaded, head: PrefixSession(loaded, head, arch)
