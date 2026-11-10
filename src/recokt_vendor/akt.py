"""PyKT AKT, plain qid embeddings and n_pid=0, with the baseline batch API.

Derived from pykt-team/pykt-toolkit, commit
ccbaf37b07e3e7ff7b2415475b2faacb967d54ea, pykt/models/akt.py.
Only the official plain-qid branch is supported.

MIT License
Copyright (c) 2022 pykt-team

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from dataclasses import asdict, dataclass
import math
from numbers import Real

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class AKTConfig:
    n_skills: int
    embedding_dim: int = 32
    n_heads: int = 2
    ff_dim: int = 64
    dropout: float = 0.0
    max_steps: int = 50
    blocks: int = 1
    final_fc_dim: int = 512
    kq_same: int = 1
    separate_qa: bool = False
    n_pid: int = 0

    def __post_init__(self):
        for name in ("n_skills", "embedding_dim", "n_heads", "ff_dim",
                     "max_steps", "blocks", "final_fc_dim"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(name + " must be a positive integer")
        if self.embedding_dim % self.n_heads:
            raise ValueError("embedding_dim must be divisible by n_heads")
        if (isinstance(self.dropout, bool) or not isinstance(self.dropout, Real)
                or not math.isfinite(self.dropout) or not 0 <= self.dropout < 1):
            raise ValueError("dropout must be finite and in [0,1)")
        if type(self.kq_same) is not int or self.kq_same not in (0, 1):
            raise ValueError("kq_same must be 0 or 1")
        if type(self.separate_qa) is not bool:
            raise ValueError("separate_qa must be boolean")
        if type(self.n_pid) is not int or self.n_pid != 0:
            raise ValueError("Only n_pid=0 is supported: there are no item IDs")


def _attention(q, k, v, d_k, allowed, dropout, gamma):
    """Official detached content-distance and negative-softplus decay."""
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    length = q.shape[2]
    positions = torch.arange(length, device=q.device)
    distance = (positions[None, :] - positions[:, None]).abs().to(scores.dtype)
    # -1e32 is the upstream sentinel. The finite fallback also allows half
    # tensors without an overflow converting the mask scalar.
    sentinel = max(-1e32, torch.finfo(scores.dtype).min)
    with torch.no_grad():
        initial = F.softmax(scores.masked_fill(~allowed, sentinel), dim=-1)
        initial = initial * allowed.to(initial.dtype)
        cumulative = torch.cumsum(initial, dim=-1)
        total = torch.sum(initial, dim=-1, keepdim=True)
        content_distance = torch.clamp((total - cumulative) * distance,
                                       min=0.).sqrt().detach()
    decay = -F.softplus(gamma).unsqueeze(0)
    effect = torch.clamp((content_distance * decay).exp(), min=1e-5, max=1e5)
    scores = F.softmax((scores * effect).masked_fill(~allowed, sentinel), dim=-1)
    # For strict retrieval this zeros the first row exactly as upstream's
    # zero_pad=True. It additionally handles completely padded examples.
    scores = scores.masked_fill(~allowed, 0.)
    return torch.matmul(dropout(scores), v)


class _MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        d = config.embedding_dim
        self.d_model = d
        self.d_k = d // config.n_heads
        self.h = config.n_heads
        self.kq_same = config.kq_same == 1
        self.v_linear = nn.Linear(d, d)
        self.k_linear = nn.Linear(d, d)
        if not self.kq_same:
            self.q_linear = nn.Linear(d, d)
        self.dropout = nn.Dropout(config.dropout)
        self.out_proj = nn.Linear(d, d)
        self.gammas = nn.Parameter(torch.zeros(self.h, 1, 1))
        nn.init.xavier_uniform_(self.gammas)
        nn.init.xavier_uniform_(self.k_linear.weight)
        nn.init.xavier_uniform_(self.v_linear.weight)
        if not self.kq_same:
            nn.init.xavier_uniform_(self.q_linear.weight)
            nn.init.constant_(self.q_linear.bias, 0.)
        nn.init.constant_(self.k_linear.bias, 0.)
        nn.init.constant_(self.v_linear.bias, 0.)
        nn.init.constant_(self.out_proj.bias, 0.)

    def forward(self, q, k, v, allowed):
        batch_size = q.shape[0]
        k = self.k_linear(k).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        q_linear = self.k_linear if self.kq_same else self.q_linear
        q = q_linear(q).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        v = self.v_linear(v).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        values = _attention(q, k, v, self.d_k, allowed, self.dropout, self.gammas)
        concat = values.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.out_proj(concat)


class _TransformerLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        d = config.embedding_dim
        self.masked_attn_head = _MultiHeadAttention(config)
        self.layer_norm1 = nn.LayerNorm(d)
        self.dropout1 = nn.Dropout(config.dropout)
        self.linear1 = nn.Linear(d, config.ff_dim)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(config.dropout)
        self.linear2 = nn.Linear(config.ff_dim, d)
        self.layer_norm2 = nn.LayerNorm(d)
        self.dropout2 = nn.Dropout(config.dropout)

    def forward(self, query, key, values, allowed, apply_pos=True):
        attended = self.masked_attn_head(query, key, values, allowed)
        query = self.layer_norm1(query + self.dropout1(attended))
        if apply_pos:
            update = self.linear2(self.dropout(self.activation(self.linear1(query))))
            query = self.layer_norm2(query + self.dropout2(update))
        return query


class _Architecture(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.blocks_1 = nn.ModuleList([_TransformerLayer(config)
                                       for _ in range(config.blocks)])
        self.blocks_2 = nn.ModuleList([_TransformerLayer(config)
                                       for _ in range(2 * config.blocks)])

    def forward(self, questions, interactions, mask):
        length = questions.shape[1]
        lower = torch.ones((length, length), dtype=torch.bool,
                           device=questions.device).tril()
        # Prefix padding means every valid query has only valid causal keys.
        inclusive = lower[None, None] & mask[:, None, None, :]
        strict = lower.tril(-1)[None, None] & mask[:, None, None, :]
        y = interactions
        for block in self.blocks_1:
            y = block(y, y, y, inclusive)
        x = questions
        for index, block in enumerate(self.blocks_2):
            if index % 2 == 0:
                x = block(x, x, x, inclusive, apply_pos=False)
            else:
                x = block(x, x, y, strict)
        return x


class AKT(nn.Module):
    """Predict response[t] using skills[:t+1] and responses[:t].

    forward()/next_response() return columns t=1..L-1, matching DKT/SAKT.
    max_steps bounds the number of next-response columns (raw L <= max_steps+1).
    Skills are zero-based atomic KCs; padding is exclusively indicated by mask.
    Only contiguous right padding is accepted. No item difficulty is inferred.
    """

    def __init__(self, config: AKTConfig):
        super().__init__()
        if not isinstance(config, AKTConfig):
            raise TypeError("AKT requires AKTConfig")
        self.config = config
        self.n_pid = config.n_pid
        self.n_question = config.n_skills
        d = config.embedding_dim
        # Match PyKT's zero-based catalog exactly: row0 is a real KC.
        self.q_embed = nn.Embedding(config.n_skills, d)
        qa_rows = 2 * config.n_skills + 1 if config.separate_qa else 2
        self.qa_embed = nn.Embedding(qa_rows, d)
        self.model = _Architecture(config)
        self.out = nn.Sequential(
            nn.Linear(2 * d, config.final_fc_dim), nn.ReLU(), nn.Dropout(config.dropout),
            nn.Linear(config.final_fc_dim, 256), nn.ReLU(), nn.Dropout(config.dropout),
            nn.Linear(256, 1),
        )

    @property
    def head(self):
        """The actual last Linear, exposed once for FedAU's feature hook."""
        return self.out[-1]

    def _validate(self, skills, responses, mask):
        if not all(isinstance(x, torch.Tensor) for x in (skills, responses, mask)):
            raise ValueError("skills, responses and mask must be tensors")
        if (skills.ndim != 2 or skills.shape != responses.shape
                or skills.shape != mask.shape or min(skills.shape) < 1):
            raise ValueError("Expected equally shaped nonempty [B,L] tensors")
        if skills.shape[1] - 1 > self.config.max_steps:
            raise ValueError("Next-target window exceeds max_steps")
        if (skills.dtype not in (torch.int32, torch.int64)
                or mask.dtype != torch.bool or responses.is_complex()):
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

    def _predict(self, skills, responses, mask):
        self._validate(skills, responses, mask)
        safe_skills = torch.where(mask, skills, 0).long()
        safe_responses = torch.where(mask, responses, 0).long()
        questions = self.q_embed(safe_skills)
        if self.config.separate_qa:
            interactions = self.qa_embed(safe_skills + self.n_question * safe_responses)
        else:
            interactions = self.qa_embed(safe_responses) + questions
        hidden = self.model(questions, interactions, mask)
        # Run the actual head once. These are its inputs after both MLP layers,
        # ReLUs and dropout, rather than the earlier 2*d concatenation.
        features = self.out[:-1](torch.cat([hidden, questions], dim=-1))
        logits = self.head(features).squeeze(-1)
        valid = mask[:, :-1] & mask[:, 1:]
        return features[:, 1:], logits[:, 1:].masked_fill(~valid, 0.)

    def forward(self, skills, responses, mask):
        return self._predict(skills, responses, mask)[1]

    def prediction_features(self, batch):
        """Return (features, logits, targets, valid) from ONE forward pass.

        features has shape [B,L-1,256] and is aligned with responses[:,1:].
        An external head pre-hook instead receives [B,L,256]; slice [:,1:].
        As with logits/targets, callers must use valid to select real examples.
        """
        features, logits = self._predict(batch["skills"], batch["responses"], batch["mask"])
        valid = batch["mask"][:, :-1] & batch["mask"][:, 1:]
        return features, logits, batch["responses"][:, 1:].float(), valid

    def next_response(self, batch):
        logits = self(batch["skills"], batch["responses"], batch["mask"])
        valid = batch["mask"][:, :-1] & batch["mask"][:, 1:]
        return logits, batch["responses"][:, 1:].float(), valid

    def losses(self, batch):
        logits, targets, valid = self.next_response(batch)
        targets = torch.where(valid, targets, 0.)
        return F.binary_cross_entropy_with_logits(logits, targets, reduction="none"), valid

    def metadata(self):
        return {"architecture": "AKT", **asdict(self.config)}
