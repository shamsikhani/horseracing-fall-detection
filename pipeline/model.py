"""
Stage 4: AttentionMIL Model Architecture.

Implements the attention-based Multiple Instance Learning localiser:
  (a) Feature projection (768 -> 256, two-layer FFN)
  (b) Attention mechanism (Ilse et al., 2018)
  (c) Bag-level classifier
  (d) Clip-level classifier
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionMIL(nn.Module):
    """Attention-based Multiple Instance Learning model for temporal localisation."""

    def __init__(self, input_dim: int = 768, hidden_dim: int = 256,
                 attention_dim: int = 128, dropout: float = 0.3):
        super().__init__()

        # (a) Feature projection: 768 -> hidden_dim -> hidden_dim
        self.feature_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # (b) Attention mechanism
        self.attention_V = nn.Linear(hidden_dim, attention_dim)
        self.attention_w = nn.Linear(attention_dim, 1)

        # (c) Bag-level classifier
        self.bag_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

        # (d) Clip-level classifier
        self.clip_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def compute_attention(self, h: torch.Tensor, mask: torch.Tensor = None):
        """
        Compute attention weights over clips.
        h: (batch, num_clips, hidden_dim)
        mask: (batch, num_clips) - True for valid positions, False for padding
        Returns: attention weights (batch, num_clips)
        """
        e = self.attention_w(torch.tanh(self.attention_V(h)))  # (batch, num_clips, 1)
        e = e.squeeze(-1)  # (batch, num_clips)

        if mask is not None:
            e = e.masked_fill(~mask, float("-inf"))

        alpha = F.softmax(e, dim=-1)  # (batch, num_clips)
        return alpha

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        """
        Forward pass.
        x: (batch, num_clips, input_dim) - clip feature sequences
        mask: (batch, num_clips) - True for valid clips, False for padding

        Returns dict with:
          - bag_prob: (batch,) bag-level probabilities
          - clip_probs: (batch, num_clips) per-clip probabilities
          - attention: (batch, num_clips) attention weights
          - bag_logit: (batch,) raw bag logits
        """
        # (a) Project features
        h = self.feature_projection(x)  # (batch, num_clips, hidden_dim)

        # (b) Compute attention
        alpha = self.compute_attention(h, mask)  # (batch, num_clips)

        # (c) Bag representation: attention-weighted sum
        r = torch.bmm(alpha.unsqueeze(1), h).squeeze(1)  # (batch, hidden_dim)
        bag_logit = self.bag_classifier(r).squeeze(-1)    # (batch,)
        bag_prob = torch.sigmoid(bag_logit)                # (batch,)

        # (d) Clip-level probabilities
        clip_logits = self.clip_classifier(h).squeeze(-1)  # (batch, num_clips)
        clip_probs = torch.sigmoid(clip_logits)             # (batch, num_clips)

        if mask is not None:
            clip_probs = clip_probs * mask.float()
            alpha = alpha * mask.float()

        return {
            "bag_prob": bag_prob,
            "bag_logit": bag_logit,
            "clip_probs": clip_probs,
            "attention": alpha,
        }
