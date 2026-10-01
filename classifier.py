"""
==========================================
classifier.py

KAGB Framework

Binary Classifier 

==========================================
"""

import torch
import torch.nn as nn

from config import Config


class MLPClassifier(nn.Module):
    """
    Simple MLP Classifier

    Input:
        (B, SHARED_DIM)

    Output:
        logits (B, 1)
    """

    def __init__(self):

        super().__init__()

        D = Config.SHARED_DIM

        self.classifier = nn.Sequential(

            nn.Linear(D, D // 2),

            nn.GELU(),

            nn.Dropout(Config.DROPOUT),

            nn.Linear(D // 2, D // 4),

            nn.GELU(),

            nn.Dropout(Config.DROPOUT),

            nn.Linear(D // 4, 1)

        )

    def forward(self, x):

        logits = self.classifier(x)

        return logits