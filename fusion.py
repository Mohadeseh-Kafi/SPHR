import torch
import torch.nn as nn

from config import Config


class GatedFusion(nn.Module):
    VERSION = "V2"

    def __init__(self):

        super().__init__()

        D = Config.SHARED_DIM

        self.gate = nn.Sequential(

            nn.Linear(D * 4, D),
            nn.GELU(),
            nn.Linear(D, 4),
            nn.Softmax(dim=-1)

        )

        self.output_projection = nn.Sequential(

            nn.Linear(D, D),
            nn.LayerNorm(D),
            nn.Dropout(Config.DROPOUT)

        )

    def forward(
        self,
        hub,
        chemical,
        structure,
        physchem
    ):

        fused = torch.cat(
            [hub, chemical, structure, physchem],
            dim=-1
        )

        gates = self.gate(fused)

        g_hub = gates[:, 0].unsqueeze(-1)
        g_chem = gates[:, 1].unsqueeze(-1)
        g_struct = gates[:, 2].unsqueeze(-1)
        g_phys = gates[:, 3].unsqueeze(-1)

        output = (
            g_hub * hub +
            g_chem * chemical +
            g_struct * structure +
            g_phys * physchem
        )

        output = self.output_projection(output)

        return output