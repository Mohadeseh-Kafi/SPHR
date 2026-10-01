# =========================================================
# PhysChem Encoder
# =========================================================
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import Config
import numpy as np

class PhysChemEncoder(nn.Module):

    """
    Physicochemical Feature Encoder

    Input:
        (B,N_features)

    Output:
        (B,Shared_dim)
    """

    def __init__(self):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(
                Config.NUM_PHYS_FEATURES,
                256
            ),

            nn.GELU(),

            nn.Dropout(Config.DROPOUT),

            nn.Linear(
                256,
                Config.SHARED_DIM
            )

        )

    ###################################################

    def forward(self, physchem):
        assert physchem.shape[-1] == Config.NUM_PHYS_FEATURES, \
            f"Expected {Config.NUM_PHYS_FEATURES} features, got {physchem.shape[-1]}"

        embedding = self.network(physchem)

        return {

            "physchem_embedding": embedding

        }