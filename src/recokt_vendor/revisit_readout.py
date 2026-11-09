"""Causal, component-local revisit features and a small residual readout."""

import torch
from torch import nn


@torch.no_grad()
def extract_revisit_features(model, batch):
    """Cache features for target i using only observations before i in its window.

    Layout: h[i-1] (64), h[j] (64), z_current, z_anchor, response[j],
    log1p(i-j-1), log1p(prior_count). j is the last observed same-KC event.
    Hidden states and logits come from one full-history pass of this component.
    Anchors include adjacent same-KC events; the correction gate requires gap > 0.
    The caller controls the backbone's train/eval mode, as for DKT.next_response.
    """
    skills, responses, mask = batch["skills"], batch["responses"], batch["mask"]
    tokens = 1 + skills + responses.long() * model.config.n_skills
    tokens = torch.where(mask, tokens, torch.zeros_like(tokens))
    hidden, _ = model.lstm(model.embedding(tokens))
    all_logits = model.head(model.dropout(hidden))
    target_skills = skills[:, 1:]
    base_logits = all_logits[:, :-1].gather(2, target_skills.unsqueeze(-1)).squeeze(-1)
    valid_mask = mask[:, :-1] & mask[:, 1:]

    positions = torch.arange(skills.shape[1], device=skills.device)
    target_positions = positions[1:]
    same_past = (
        (target_skills.unsqueeze(-1) == skills.unsqueeze(1))
        & mask.unsqueeze(1)
        & (positions[None, None, :] < target_positions[None, :, None])
    )
    anchor = torch.where(same_past, positions, -1).amax(dim=-1)
    has_anchor = valid_mask & (anchor >= 0)
    prior_count = same_past.sum(dim=-1).masked_fill(~valid_mask, 0)
    gap = (target_positions - anchor - 1).masked_fill(~has_anchor, 0)
    revisit_mask = has_anchor & (gap > 0)

    # Index zero is only a safe gather placeholder when no anchor exists.
    anchor_index = anchor.clamp_min(0)
    anchor_hidden = hidden.gather(
        1, anchor_index.unsqueeze(-1).expand(-1, -1, hidden.shape[-1]))
    anchor_logits = all_logits.gather(
        1, anchor_index.unsqueeze(-1).expand(-1, -1, all_logits.shape[-1]))
    anchor_logits = anchor_logits.gather(2, target_skills.unsqueeze(-1)).squeeze(-1)
    anchor_response = responses.gather(1, anchor_index)
    anchor_hidden = anchor_hidden.masked_fill(~has_anchor.unsqueeze(-1), 0)
    anchor_logits = anchor_logits.masked_fill(~has_anchor, 0)
    anchor_response = anchor_response.masked_fill(~has_anchor, 0)
    scalars = torch.stack((base_logits, anchor_logits, anchor_response,
                           gap.to(hidden.dtype).log1p(),
                           prior_count.to(hidden.dtype).log1p()), dim=-1)
    features = torch.cat((hidden[:, :-1], anchor_hidden, scalars), dim=-1)
    features = features.masked_fill(~valid_mask.unsqueeze(-1), 0)
    return {"features": features.detach(), "base_logits": base_logits.detach(),
            "targets": responses[:, 1:].float().detach(), "valid_mask": valid_mask,
            "revisit_mask": revisit_mask, "gap": gap, "prior_count": prior_count}


class RevisitHead(nn.Module):
    """Bounded logit correction with an exactly zero initial residual."""

    def __init__(self, mode, seed=None):
        super().__init__()
        self.mode = mode
        input_dim, hidden_dim = {
            "current": (68, 16), "metadata": (4, 180), "full": (133, 8)
        }[mode]
        # Initialize privately so adding a readout does not change backbone RNG.
        generator = torch.Generator()
        generator.seed() if seed is None else generator.manual_seed(seed)
        with torch.random.fork_rng(devices=[]):
            torch.random.set_rng_state(generator.get_state())
            self.net = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.Tanh(),
                                     nn.Linear(hidden_dim, 1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, features):
        if self.mode == "current":
            features = torch.cat((features[..., :64], features[..., 128:129],
                                  features[..., 130:133]), dim=-1)
        elif self.mode == "metadata":
            features = torch.cat((features[..., 128:129], features[..., 130:133]), dim=-1)
        return torch.tanh(self.net(features).squeeze(-1))


def corrected_logits(head, pack):
    """Apply the readout only to valid within-window revisits with gap > 0."""
    return torch.where(pack["revisit_mask"],
                       pack["base_logits"] + head(pack["features"]),
                       pack["base_logits"])
