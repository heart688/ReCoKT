"""Adapter and revisit-head primitives from the main experiment source.

Source file SHA-256: 322f05f4044a92f75edc39a1698fa56a1cfd82f0cedbfb4d02a5ceff4259753a.
"""
from copy import deepcopy
import math
RANK = 16

def _libs():
    import torch
    from torch import nn
    from torch.func import functional_call
    from recokt_vendor import baseline_training as bt, data, federated as fed
    return torch, nn, functional_call, bt, data, fed

def _key(name: str) -> str:
    return name.replace(".", "__")

class GenericRank16Adapter:  # nn.Module is imported lazily to keep metadata mode light
    """Additive rank-16 adapter for any supported SAKT/AKT parameter shapes."""

    def __new__(cls, base, rank=RANK, seed=811):
        torch, nn, functional_call, _, _, _ = _libs()

        class Adapter(nn.Module):
            def __init__(self, base_model, rank_value, adapter_seed):
                super().__init__()
                if rank_value != RANK:
                    raise ValueError("Candidate architecture smoke is fixed to rank16")
                self.base = deepcopy(base_model)
                self.rank = rank_value
                self.adapter_seed = adapter_seed
                parameters = dict(self.base.named_parameters())
                if not parameters or any(not p.is_floating_point() for p in parameters.values()):
                    raise ValueError("SAKT/AKT contains a non-floating parameter")
                generator = torch.Generator(device="cpu").manual_seed(adapter_seed)
                self.matrices = nn.ModuleDict()
                self.vectors = nn.ParameterDict()
                for name, parameter in parameters.items():
                    key = _key(name)
                    if parameter.ndim == 2:
                        rows, columns = parameter.shape
                        a = torch.randn((rank_value, columns), generator=generator,
                                        dtype=torch.float64)
                        self.matrices[key] = nn.ParameterDict({
                            "A": nn.Parameter((a / math.sqrt(columns)).to(parameter)),
                            "B": nn.Parameter(torch.zeros((rows, rank_value),
                                                           dtype=parameter.dtype,
                                                           device=parameter.device)),
                        })
                    else:
                        # Biases and AKT's small higher-rank gamma tensors are
                        # additive full tensors, matching the existing DKT
                        # adapter's treatment of non-matrix parameters.  No
                        # DKT hidden size is introduced here.
                        self.vectors[key] = nn.Parameter(torch.zeros_like(parameter))
                self._parameter_names = tuple(parameters)
                self.base.requires_grad_(False)

            def _delta(self, name):
                key = _key(name)
                if key in self.matrices:
                    matrix = self.matrices[key]
                    return matrix["B"] @ matrix["A"]
                return self.vectors[key]

            def effective_state(self):
                state = {}
                for name, value in self.base.named_parameters():
                    state[name] = value + self._delta(name)
                for name, value in self.base.named_buffers():
                    state[name] = value
                return state

            def forward(self, skills, responses, mask):
                return functional_call(self.base, self.effective_state(),
                                       (skills, responses, mask), strict=True)

            def next_response(self, batch):
                logits = self(batch["skills"], batch["responses"], batch["mask"])
                valid = batch["mask"][:, :-1] & batch["mask"][:, 1:]
                return logits, batch["responses"][:, 1:].float(), valid

            def losses(self, batch):
                import torch.nn.functional as F
                logits, targets, valid = self.next_response(batch)
                return F.binary_cross_entropy_with_logits(
                    logits, torch.where(valid, targets, 0.), reduction="none"), valid

            def adapter_parameters(self):
                return list(self.matrices.parameters()) + list(self.vectors.parameters())

            def adapter_state(self):
                return {key: value.detach().cpu().clone()
                        for key, value in self.matrices.state_dict().items()} | {
                            "vectors." + key: value.detach().cpu().clone()
                            for key, value in self.vectors.state_dict().items()}

            def load_adapter_state(self, state):
                matrix_state = {key: value for key, value in state.items()
                                if not key.startswith("vectors.")}
                vector_state = {key[len("vectors."):]: value for key, value in state.items()
                                if key.startswith("vectors.")}
                self.matrices.load_state_dict(matrix_state, strict=True)
                self.vectors.load_state_dict(vector_state, strict=True)

            def materialize(self):
                result = deepcopy(self.base)
                result.load_state_dict(self.effective_state(), strict=True)
                return result

            def metadata(self):
                return {"adapter": "generic_additive_low_rank", "rank": self.rank,
                        "adapter_seed": self.adapter_seed,
                        "base": self.base.metadata()}

        return Adapter(base, rank, seed)

class GenericFullRevisitHead:  # nn.Module is imported lazily
    """Full revisit head with architecture-specific input width."""

    def __new__(cls, feature_dim: int, seed: int):
        torch, nn, _, _, _, _ = _libs()

        class Head(nn.Module):
            def __init__(self):
                super().__init__()
                self.feature_dim = feature_dim
                self.input_dim = 2 * feature_dim + 5
                generator = torch.Generator(device="cpu").manual_seed(seed)
                with torch.random.fork_rng(devices=[]):
                    torch.random.set_rng_state(generator.get_state())
                    self.net = nn.Sequential(nn.Linear(self.input_dim, 8), nn.Tanh(),
                                             nn.Linear(8, 1))
                nn.init.zeros_(self.net[-1].weight)
                nn.init.zeros_(self.net[-1].bias)

            def forward(self, features):
                return torch.tanh(self.net(features).squeeze(-1))

            def metadata(self):
                return {"head": "architecture_specific_full_revisit", "feature_dim": feature_dim,
                        "input_dim": self.input_dim, "hidden_dim": 8, "seed": seed}

        return Head()

def _capture_features(model, batch):
    """Capture the actual final-head input through the public model head."""
    torch, _, _, _, _, _ = _libs()
    captured = []
    hook = model.head.register_forward_pre_hook(lambda _module, args: captured.append(args[0]))
    try:
        logits, targets, valid = model.next_response(batch)
    finally:
        hook.remove()
    if len(captured) != 1:
        raise ValueError("final head did not run exactly once")
    features = captured[0]
    if features.shape[1] == logits.shape[1] + 1:
        features = features[:, 1:]
    if tuple(features.shape[:2]) != tuple(logits.shape):
        raise ValueError("final-head feature and next-target shapes differ")
    return features, logits, targets, valid

def _revisit_pack(model, batch):
    torch, _, _, _, _, _ = _libs()
    features, logits, targets, valid = _capture_features(model, batch)
    skills = batch["skills"]
    responses = batch["responses"]
    target_skills = skills[:, 1:]
    target_count = target_skills.shape[1]
    anchor_features = torch.zeros_like(features)
    anchor_logits = torch.zeros_like(logits)
    anchor_response = torch.zeros_like(targets)
    gap = torch.zeros_like(targets)
    prior_count = torch.zeros_like(targets)
    revisit = torch.zeros_like(valid)
    for row in range(skills.shape[0]):
        for column in range(target_count):
            target_position = column + 1
            prior = [position for position in range(1, target_position)
                     if bool(batch["mask"][row, position])
                     and int(skills[row, position]) == int(target_skills[row, column])]
            if not prior:
                continue
            anchor_position = prior[-1]
            anchor_column = anchor_position - 1
            anchor_features[row, column] = features[row, anchor_column]
            anchor_logits[row, column] = logits[row, anchor_column]
            anchor_response[row, column] = responses[row, anchor_position]
            gap[row, column] = target_position - anchor_position - 1
            prior_count[row, column] = len(prior)
            revisit[row, column] = bool(valid[row, column]) and gap[row, column] > 0
    composed = torch.cat((features, anchor_features,
                          logits.unsqueeze(-1), anchor_logits.unsqueeze(-1),
                          anchor_response.unsqueeze(-1),
                          gap.to(features.dtype).log1p().unsqueeze(-1),
                          prior_count.to(features.dtype).log1p().unsqueeze(-1)), dim=-1)
    return {"features": composed.detach(), "base_logits": logits.detach(),
            "targets": targets.detach(), "valid_mask": valid,
            "revisit_mask": revisit, "feature_dim": int(features.shape[-1]),
            "anchor_policy": "previous target positions only; context position is not exposed by SAKT/AKT final-head API"}
