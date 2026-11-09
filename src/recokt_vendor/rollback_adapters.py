"""Additive, separately frozen DKT stages; no data loading or training loop.

For every matrix, W_eff = W_base + sum(B_j @ A_j).  Scale is exactly 1:
there is no division by rank and no alpha hyperparameter.  Every vector,
including both LSTM biases, receives a full-sized additive stage delta.
Dropping a stage changes this sum; it does not undo its effect on later stages.
"""

from copy import deepcopy
from dataclasses import asdict
import math

import torch
from torch import nn
from torch.func import functional_call

from .model import DKT, ModelConfig


def _key(name):
    return name.replace(".", "__")


class _LowRankWeight(nn.Module):
    def __init__(self, weight, rank, generator, padding=False):
        super().__init__()
        rows, columns = weight.shape
        # CPU-local RNG also makes construction independent of the base device.
        a = torch.randn(rank, columns, generator=generator, dtype=torch.float64)
        self.A = nn.Parameter((a / math.sqrt(columns)).to(weight))
        self.B = nn.Parameter(weight.new_zeros(rows, rank))
        self.padding = padding

    def forward(self):
        delta = self.B @ self.A
        if self.padding:
            # Structural masking, not an optimizer hook: row zero stays zero
            # even after loading or directly changing a factor's padding row.
            delta = torch.cat((torch.zeros_like(delta[:1]), delta[1:]), dim=0)
        return delta


class _StageAdapter(nn.Module):
    def __init__(self, parameters, rank, seed):
        super().__init__()
        generator = torch.Generator(device="cpu").manual_seed(seed)
        self.matrices = nn.ModuleDict()
        self.vectors = nn.ParameterDict()
        for name, parameter in parameters.items():
            if parameter.ndim == 2:
                self.matrices[_key(name)] = _LowRankWeight(
                    parameter, rank, generator, padding=name == "embedding.weight")
            else:
                self.vectors[_key(name)] = nn.Parameter(torch.zeros_like(parameter))

    def delta(self, name):
        key = _key(name)
        return self.matrices[key]() if key in self.matrices else self.vectors[key]


class StagewiseDKT(nn.Module):
    """Frozen private base plus independent additive adapters.

    Public storage interfaces are ``base`` (ordinary DKT), ``adapters``
    (ModuleList), and ``adapter_state(i)`` (cloned tensor-only state dict).
    Construction selects active prefix (0,) and trainable stage 0.

    Source stage j: set_active(range(j + 1)), then train_stage(j).
    train_stage only changes gradient permissions, never the active set.
    For a direct drop use set_active((0, 2)); later adapters retain their values
    and may still carry dependence on the omitted stage's training history.
    """

    def __init__(self, base_model, rank=8, stages=3, seed=811):
        super().__init__()
        if not isinstance(base_model, DKT):
            raise TypeError("base_model must be the existing DKT")
        for value, name in ((rank, "rank"), (stages, "stages")):
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if type(seed) is not int or not 0 <= seed < 2**63:
            raise ValueError("seed must be an integer in [0, 2**63)")
        if list(base_model.named_buffers()):
            raise ValueError("This DKT adapter implementation requires no model buffers")
        parameters = dict(base_model.named_parameters())
        if (not parameters or any(p.ndim not in (1, 2) or not p.is_floating_point()
                                  for p in parameters.values())):
            raise ValueError("All DKT parameters must be floating-point matrices or vectors")
        first = next(iter(parameters.values()))
        if any(p.device != first.device or p.dtype != first.dtype for p in parameters.values()):
            raise ValueError("DKT parameters must share one device and dtype")
        if bool((base_model.embedding.weight[0].detach() != 0).any()):
            raise ValueError("The external base must already have a zero padding row")
        # Do not freeze or mutate the caller's model, its gradients or its mode.
        self.base = deepcopy(base_model)
        self.config = base_model.config
        self.rank, self.num_stages, self.seed = rank, stages, seed
        self._parameter_names = tuple(parameters)
        self.adapters = nn.ModuleList([
            _StageAdapter(dict(self.base.named_parameters()), rank, (seed + j) % 2**63)
            for j in range(stages)
        ])
        self._active_indices = (0,)
        self._trainable_stage = None
        self.train(base_model.training)
        self.train_stage(0)

    @property
    def active_indices(self):
        return self._active_indices

    @property
    def trainable_stage(self):
        return self._trainable_stage

    def _index(self, index):
        if type(index) is not int or not 0 <= index < self.num_stages:
            raise ValueError("Stage index is outside the configured stage count")
        return index

    def _indices(self, indices):
        indices = tuple(indices)
        for index in indices:
            self._index(index)
        if len(set(indices)) != len(indices):
            raise ValueError("Duplicate active stage")
        # Canonical summation order, including for direct-drop subsets.
        return tuple(sorted(indices))

    def _freeze_except(self, index):
        self.base.requires_grad_(False)
        for parameter in self.base.parameters():
            parameter.grad = None
        for j, adapter in enumerate(self.adapters):
            adapter.requires_grad_(j == index)
            for parameter in adapter.parameters():
                parameter.grad = None
        self._trainable_stage = index

    def set_active(self, indices):
        """Select an ordered subset; an excluded trainable stage is frozen."""
        active = self._indices(indices)
        self._active_indices = active
        if self._trainable_stage not in active:
            self._freeze_except(None)
        return self

    def train_stage(self, index):
        """Only stage index is trainable; set its active prefix separately."""
        index = self._index(index)
        if index not in self._active_indices:
            raise ValueError("Select an active prefix/subset before train_stage(index)")
        self._freeze_except(index)
        return self

    def _effective_parameters(self, indices):
        effective = dict(self.base.named_parameters())
        for name in self._parameter_names:
            for index in indices:
                effective[name] = effective[name] + self.adapters[index].delta(name)
        return effective

    def forward(self, skills, responses, mask):
        return functional_call(self.base, self._effective_parameters(self._active_indices),
                               (skills, responses, mask), strict=True)

    def next_response(self, batch):
        # Reuse DKT's target/mask rules; self.forward routes through functional_call.
        return DKT.next_response(self, batch)

    def losses(self, batch):
        return DKT.losses(self, batch)

    @torch.no_grad()
    def materialize(self, indices=None):
        """Return an independent ordinary, trainable DKT with the selected sum."""
        active = self._active_indices if indices is None else self._indices(indices)
        reference = next(self.base.parameters())
        # Ordinary DKT initialization uses CPU global RNG; preserve that state.
        # Reinstantiation avoids copying LSTM's cached non-leaf functional weights.
        with torch.random.fork_rng(devices=[]):
            result = DKT(self.config).to(device=reference.device, dtype=reference.dtype)
        result.load_state_dict(self._effective_parameters(active), strict=True)
        result.train(self.training)
        return result

    def adapter_state(self, index):
        """Tensor-only snapshot, loadable by adapters[index].load_state_dict."""
        return {name: tensor.detach().clone()
                for name, tensor in self.adapters[self._index(index)].state_dict().items()}

    def metadata(self):
        """Only constructor/selection metadata; no data paths or data access."""
        return dict(version=1, model_config=asdict(self.config), rank=self.rank,
                    stages=self.num_stages, seed=self.seed, scale=1.0,
                    active_indices=list(self._active_indices),
                    trainable_stage=self._trainable_stage, training=self.training)

    def get_extra_state(self):
        return self.metadata()

    def set_extra_state(self, state):
        expected = self.metadata()
        if not isinstance(state, dict) or set(state) != set(expected):
            raise ValueError("Invalid StagewiseDKT checkpoint metadata")
        for key in ("version", "model_config", "rank", "stages", "seed", "scale"):
            if state[key] != expected[key]:
                raise ValueError(f"Checkpoint constructor mismatch: {key}")
        active = self._indices(state["active_indices"])
        index = state["trainable_stage"]
        if index is not None and self._index(index) not in active:
            raise ValueError("Checkpoint trainable stage must be active")
        if type(state["training"]) is not bool:
            raise ValueError("Checkpoint training flag must be boolean")
        self._active_indices = active
        self._freeze_except(index)
        self.train(state["training"])

    @classmethod
    def from_state_dict(cls, state):
        """Rebuild from a full state_dict, including its plain extra metadata."""
        metadata = state["_extra_state"]
        reference = state["base.embedding.weight"]
        with torch.random.fork_rng(devices=[]):
            base = DKT(ModelConfig(**metadata["model_config"]))
        base = base.to(device=reference.device, dtype=reference.dtype)
        result = cls(base, rank=metadata["rank"], stages=metadata["stages"],
                     seed=metadata["seed"])
        result.load_state_dict(state, strict=True)
        return result
