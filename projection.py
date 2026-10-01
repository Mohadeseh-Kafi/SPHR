import torch
import torch.nn as nn


class ProjectionBlock(nn.Module):
    """
    Unified Projection Block

    Input:
        (B, input_dim)

    Output:
        (B, shared_dim)
    """

    def __init__(
        self,
        input_dim: int,
        shared_dim: int = 512,
        dropout: float = 0.2
    ):
        super().__init__()

        self.projection = nn.Sequential(

            nn.Linear(input_dim, shared_dim),

            nn.LayerNorm(shared_dim),

            nn.GELU(),

            nn.Dropout(dropout)

        )

    def forward(self, x):

        return self.projection(x)

    # ump = UnifiedModalityProjection(
    #
    #     seq_dim=480,
    #
    #     chem_dim=768,
    #
    #     struct_dim=768,
    #
    #     phys_dim=128,
    #
    #     shared_dim=512,
    #
    # )
    # seq, chem, struct, phys = ump(
    #
    #     seq_embedding,
    #
    #     chem_embedding,
    #
    #     structure_embedding,
    #
    #     phys_embedding,
    #
    # )