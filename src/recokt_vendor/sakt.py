"""Minimal causal SAKT adaptation, not a full FAKT backbone or new method.

One attention block reads observed skill/response interactions. Each query is
only the next skill: its response never enters that prediction. Learned absolute
positions are added to history embeddings before key/value projection, never to
the next-skill queries. Windows reset their history just as in the DKT loader.
"""

from dataclasses import asdict, dataclass
import math
from numbers import Real

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class SAKTConfig:
    n_skills: int
    embedding_dim: int = 32
    n_heads: int = 2
    max_steps: int = 50
    ff_dim: int = 64
    dropout: float = 0.0

    def __post_init__(self):
        for name in ("n_skills", "embedding_dim", "n_heads", "max_steps", "ff_dim"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(name + " must be a positive integer")
        if self.embedding_dim % self.n_heads:
            raise ValueError("embedding_dim must be divisible by n_heads")
        if (isinstance(self.dropout, bool) or not isinstance(self.dropout, Real) or
                not math.isfinite(self.dropout) or not 0 <= self.dropout < 1):
            raise ValueError("dropout must be finite and in [0,1)")


class SAKT(nn.Module):
    """At output t, query skill[t+1] attends only to inputs at positions <=t.

    Real interaction tokens are 1+skill+response*K; real query tokens are
    1+skill. Both embedding row0s are padding. Only contiguous prefix windows
    are accepted. Completely padded rows and invalid outputs are finite zeros;
    loss consumers must still use the returned next-target mask.
    """

    def __init__(self, config: SAKTConfig):
        super().__init__()
        self.config = config
        d = config.embedding_dim
        self.interaction_embedding = nn.Embedding(2*config.n_skills+1, d, padding_idx=0)
        self.query_embedding = nn.Embedding(config.n_skills+1, d, padding_idx=0)
        self.position_embedding = nn.Embedding(config.max_steps, d)
        self.attention = nn.MultiheadAttention(d, config.n_heads, dropout=config.dropout, batch_first=True)
        self.attention_norm = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, config.ff_dim), nn.ReLU(),
                                nn.Dropout(config.dropout), nn.Linear(config.ff_dim, d))
        self.ff_norm = nn.LayerNorm(d)
        self.dropout = nn.Dropout(config.dropout)
        self.head = nn.Linear(d, 1)

    def _validate(self, skills, responses, mask):
        if not all(isinstance(x, torch.Tensor) for x in (skills, responses, mask)):
            raise ValueError("skills, responses and mask must be tensors")
        if (skills.ndim != 2 or skills.shape != responses.shape or skills.shape != mask.shape or
                skills.shape[0] < 1 or skills.shape[1] < 1):
            raise ValueError("Expected equally shaped nonempty [B,L] tensors")
        if skills.shape[1]-1 > self.config.max_steps:
            raise ValueError("Next-target window exceeds max_steps")
        if skills.dtype not in (torch.int32, torch.int64) or mask.dtype != torch.bool or responses.is_complex():
            raise ValueError("Skills must be integer, mask boolean and responses real")
        if skills.device != responses.device or skills.device != mask.device:
            raise ValueError("Batch tensors must share a device")
        if bool((mask[:, 1:] & ~mask[:, :-1]).any()):
            raise ValueError("Only contiguous prefix padding is supported")
        if bool(((skills[mask] < 0) | (skills[mask] >= self.config.n_skills)).any()):
            raise ValueError("Observed skill outside the fixed catalog")
        observed = responses[mask]
        if bool(((observed != 0) & (observed != 1)).any()):
            raise ValueError("Observed responses must be binary and finite")

    def forward(self, skills, responses, mask):
        self._validate(skills, responses, mask)
        steps = skills.shape[1]-1
        if steps == 0:
            return self.head.weight.new_zeros((skills.shape[0], 0))
        valid = mask[:, :-1] & mask[:, 1:]
        # Exclude the final raw event of EACH window, including shorter padded
        # windows. It cannot be history for any valid target in that window.
        past_skill = torch.where(valid, skills[:, :-1], 0).long()
        past_response = torch.where(valid, responses[:, :-1], 0).long()
        interactions = torch.where(valid, 1+past_skill+past_response*self.config.n_skills, 0)
        queries = torch.where(valid, 1+skills[:, 1:], 0).long()
        positions = torch.arange(steps, device=skills.device)
        history = self.interaction_embedding(interactions)+self.position_embedding(positions)[None]
        history = history.masked_fill(~valid[..., None], 0.)
        query = self.query_embedding(queries)
        padding = ~valid
        # MHA softmax must not see an all-masked row. Prefix padding ensures all
        # valid queries have key0; empty windows use a zero dummy key at position0
        # and their outputs are discarded below. No future key is ever unmasked.
        padding = padding.clone()
        padding[~valid.any(dim=1), 0] = False
        causal = torch.ones((steps, steps), device=skills.device, dtype=torch.bool).triu(1)
        attended, _ = self.attention(query, history, history, attn_mask=causal,
                                     key_padding_mask=padding, need_weights=False)
        hidden = self.attention_norm(query+self.dropout(attended))
        hidden = self.ff_norm(hidden+self.dropout(self.ff(hidden)))
        logits = self.head(hidden).squeeze(-1)
        return logits.masked_fill(~valid, 0.)

    def next_response(self, batch):
        logits = self(batch["skills"], batch["responses"], batch["mask"])
        targets = batch["responses"][:, 1:].float()
        valid = batch["mask"][:, :-1] & batch["mask"][:, 1:]
        return logits, targets, valid

    def losses(self, batch):
        logits, targets, valid = self.next_response(batch)
        # Padding values are semantically absent, even if a caller supplied NaN
        # there; valid target responses were checked by forward().
        losses = F.binary_cross_entropy_with_logits(logits, torch.where(valid, targets, 0.), reduction="none")
        return losses, valid

    def metadata(self):
        return {"architecture": "SAKT", **asdict(self.config)}
