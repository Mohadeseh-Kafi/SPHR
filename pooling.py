"""
=========================================================
Pooling Layers
KAGB Framework
=========================================================

Contains:

1. MeanPooling
2. MaskedMeanPooling
3. CLSPooling
4. AttentionPooling
5. GeMPooling

Author:
Amir Lakizadeh Lab (KAGB)
=========================================================
"""

import torch
import torch.nn as nn


import torch.nn.functional as F


# =========================================================
# Mean Pooling
# =========================================================

class MeanPooling(nn.Module):

    """
    Simple Mean Pooling

    Input
    -----
    x : (B,L,D)

    Output
    ------
    (B,D)
    """

    def __init__(self):
        super().__init__()

    def forward(self, x, attention_mask=None):

        if attention_mask is None:
            return x.mean(dim=1)

        mask = attention_mask.unsqueeze(-1).float()

        x = x * mask

        pooled = x.sum(dim=1)

        pooled = pooled / mask.sum(dim=1).clamp(min=1e-9)

        return pooled


# =========================================================
# Masked Mean Pooling
# =========================================================

class MaskedMeanPooling(nn.Module):

    """
    Explicit masked mean pooling.

    Always expects attention_mask.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x, attention_mask):

        mask = attention_mask.unsqueeze(-1).float()

        x = x * mask

        pooled = x.sum(dim=1)

        pooled = pooled / mask.sum(dim=1).clamp(min=1e-9)

        return pooled


# =========================================================
# CLS Pooling
# =========================================================

class CLSPooling(nn.Module):

    """
    Use CLS Token

    Output
    ------
    x[:,0,:]
    """

    def __init__(self):
        super().__init__()

    def forward(self, x, attention_mask=None):

        return x[:, 0]


# =========================================================
# Attention Pooling
# =========================================================

class AttentionPooling(nn.Module):

    """
    Learnable Attention Pooling

    score_i = softmax(Wx)

    output = Σ(score_i * token_i)
    """

    def __init__(self, hidden_dim):

        super().__init__()

        self.attention = nn.Linear(hidden_dim, 1)

    def forward(self, x, attention_mask=None):

        scores = self.attention(x)

        if attention_mask is not None:

            scores = scores.masked_fill(
                attention_mask.unsqueeze(-1) == 0,
                -1e9
            )

        weights = torch.softmax(scores, dim=1)

        pooled = torch.sum(weights * x, dim=1)

        return pooled


# =========================================================
# GeM Pooling
# =========================================================

class GeMPooling(nn.Module):

    """
    Generalized Mean Pooling

    Originally proposed for image retrieval.

    Works surprisingly well for protein embeddings.
    """

    def __init__(self,
                 p=3,
                 eps=1e-6):

        super().__init__()

        self.p = nn.Parameter(torch.ones(1) * p)

        self.eps = eps

    def forward(self,
                x,
                attention_mask=None):

        if attention_mask is not None:

            mask = attention_mask.unsqueeze(-1).float()

            x = x * mask

        x = x.clamp(min=self.eps)

        pooled = torch.mean(
            x.pow(self.p),
            dim=1
        ).pow(1.0 / self.p)

        return pooled