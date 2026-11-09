from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class ModelConfig:
    n_skills: int
    embedding_dim: int = 32
    hidden_dim: int = 64
    dropout: float = 0.0


class DKT(nn.Module):
    """Response at time t predicts response to the known item at t+1.

    Skill-response embeddings and a one-layer LSTM are a minimal DKT variant.
    This is not the original quality-weighted/personalized FDKT implementation.
    Padding is encoded as interaction 0, real interactions as 1..2K.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        if config.n_skills < 1 or min(config.embedding_dim, config.hidden_dim) < 1:
            raise ValueError("Model dimensions must be positive")
        self.config = config
        self.embedding = nn.Embedding(2 * config.n_skills + 1,
                                      config.embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(config.embedding_dim, config.hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(config.dropout)
        self.head = nn.Linear(config.hidden_dim, config.n_skills)

    def forward(self, skills, responses, mask):
        if skills.shape != responses.shape or skills.shape != mask.shape:
            raise ValueError("skills, responses, mask must have the same [B,L] shape")
        tokens = 1 + skills + responses.long() * self.config.n_skills
        tokens = torch.where(mask, tokens, torch.zeros_like(tokens))
        hidden, _ = self.lstm(self.embedding(tokens))
        return self.head(self.dropout(hidden))

    def next_response(self, batch):
        all_logits = self(batch["skills"], batch["responses"], batch["mask"])
        logits = all_logits[:, :-1].gather(2, batch["skills"][:, 1:, None]).squeeze(-1)
        targets = batch["responses"][:, 1:].float()
        mask = batch["mask"][:, 1:] & batch["mask"][:, :-1]
        return logits, targets, mask

    def losses(self, batch):
        logits, targets, mask = self.next_response(batch)
        losses = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        return losses, mask

    def metadata(self):
        return {"architecture": "DKT-LSTM", **asdict(self.config)}
