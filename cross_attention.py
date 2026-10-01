import torch
import torch.nn as nn

from config import Config
class BaseCrossAttention(nn.Module):

    def __init__(
            self,
            hidden_dim,
            num_heads,
            dropout):

        super().__init__()
        #key_padding_mask = key_padding_mask.to(torch.bool)
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.norm1 = nn.LayerNorm(hidden_dim)

        self.ffn = nn.Sequential(

            nn.Linear(hidden_dim,
                      hidden_dim * 4),

            nn.GELU(),

            nn.Dropout(dropout),

            nn.Linear(hidden_dim * 4,
                      hidden_dim)

        )
        ##########################################################
        # Ablation : Concatenation Projection
        ##########################################################

        self.concat_projection = nn.Sequential(

            nn.Linear(hidden_dim * 4, hidden_dim),

            nn.GELU(),

            nn.Dropout(dropout),

            nn.LayerNorm(hidden_dim)

        )

        self.norm2 = nn.LayerNorm(hidden_dim)


        self.dropout = nn.Dropout(dropout)

    def forward(

            self,

            query,

            key,

            value,

            key_padding_mask=None):
        if key_padding_mask is not None:
            key_padding_mask = (key_padding_mask == 0)

        attended, weights = self.attention(
            query=query,
            key=key,
            value=value,
            key_padding_mask=key_padding_mask
        )

        x = self.norm1(

            query +

            self.dropout(attended)

        )

        ffn = self.ffn(x)

        output = self.norm2(

            x +

            self.dropout(ffn)

        )

        return output, weights
##############################################################
# Hub Cross Attention Version 2
##############################################################

class HubCrossAttention(nn.Module):
    """
    SPHR Version 2

    Inputs
    ------
    seq_feat:
        projected_tokens
        attention_mask

    chem_feat:
        projected_tokens
        attention_mask

    struct_feat:
        projected_tokens
        attention_mask

    phys_feat:
        physchem_embedding

    Output
    ------
    fused_embedding
    """

    ##########################################################
    # Mean Pooling
    ##########################################################

    @staticmethod
    def masked_mean_pooling(tokens, mask):
        """
        tokens : (B,L,D)
        mask   : (B,L)

        return : (B,D)
        """

        mask = mask.float().unsqueeze(-1)

        summed = (tokens * mask).sum(dim=1)

        lengths = mask.sum(dim=1).clamp(min=1e-6)

        pooled = summed / lengths

        return pooled

    ##############################################################
    # Constructor
    ##############################################################

    def __init__(
            self,
            seq_dim=480,
            chem_dim=384,
            struct_dim=512,
            phys_dim=512,
            hidden_dim=512,
            num_heads=8,
            dropout=0.1,
    ):
        super().__init__()

        ##########################################################
        # Dimensions
        ##########################################################

        self.seq_dim = seq_dim
        self.chem_dim = chem_dim
        self.struct_dim = struct_dim
        self.phys_dim = phys_dim

        self.hidden_dim = hidden_dim

        self.ablation_mode = Config.ABLATION_MODE
        ##########################################################
        # Projectors
        ##########################################################

        ##########################################################
        # Projectors
        ##########################################################

        self.seq_proj = (
            nn.Identity()
            if seq_dim == hidden_dim
            else nn.Linear(seq_dim, hidden_dim)
        )

        self.chem_proj = (
            nn.Identity()
            if chem_dim == hidden_dim
            else nn.Linear(chem_dim, hidden_dim)
        )

        self.struct_proj = (
            nn.Identity()
            if struct_dim == hidden_dim
            else nn.Linear(struct_dim, hidden_dim)
        )

        self.phys_proj = (
            nn.Identity()
            if phys_dim == hidden_dim
            else nn.Linear(phys_dim, hidden_dim)
        )
        ##########################################################
        # Concatenation Projection
        ##########################################################

        self.concat_projection = nn.Sequential(

            nn.Linear(hidden_dim * 4, hidden_dim),

            nn.GELU(),

            nn.Dropout(dropout)

        )
        ##########################################################
        # Cross Attention Blocks
        ##########################################################

        self.seq_to_chem = BaseCrossAttention(
            hidden_dim,
            num_heads,
            dropout
        )

        self.seq_to_struct = BaseCrossAttention(
            hidden_dim,
            num_heads,
            dropout
        )

        self.seq_to_phys = BaseCrossAttention(
            hidden_dim,
            num_heads,
            dropout
        )

        ##########################################################
        # LayerNorm
        ##########################################################

        self.norm1 = nn.LayerNorm(hidden_dim)

        self.norm2 = nn.LayerNorm(hidden_dim)

        ##########################################################
        # Feed Forward Network
        ##########################################################

        self.ffn = nn.Sequential(

            nn.Linear(hidden_dim, hidden_dim * 4),

            nn.GELU(),

            nn.Dropout(dropout),

            nn.Linear(hidden_dim * 4, hidden_dim),

        )

        ##########################################################
        # Dropout
        ##########################################################

        self.dropout = nn.Dropout(dropout)

        self.ablation_mode = Config.ABLATION_MODE

    ##############################################################
    # Prepare Features
    ##############################################################

    def _prepare_features(

            self,

            seq_feat,

            chem_feat,

            struct_feat,

            phys_feat,

    ):
        ##########################################################
        # Sequence
        ##########################################################

        seq_tokens = self.seq_proj(

            seq_feat["projected_tokens"]

        )
        seq_embedding = self.masked_mean_pooling(

            seq_tokens,

            seq_feat["attention_mask"]

        )

        ##########################################################
        # Chemical
        ##########################################################

        chem_tokens = self.chem_proj(

            chem_feat["projected_tokens"]

        )

        chem_embedding = self.masked_mean_pooling(

            chem_tokens,

            chem_feat["attention_mask"]

        )

        ##########################################################
        # Structure
        ##########################################################

        struct_tokens = self.struct_proj(

            struct_feat["projected_tokens"]

        )

        struct_embedding = self.masked_mean_pooling(

            struct_tokens,

            struct_feat["attention_mask"]

        )

        ##########################################################
        # PhysChem
        ##########################################################

        phys_embedding = self.phys_proj(

            phys_feat["physchem_embedding"]

        )

        ##########################################################
        # Return
        ##########################################################

        return {

            "seq_tokens": seq_tokens,
            "seq_embedding": seq_embedding,
            "chem_tokens": chem_tokens,

            "chem_embedding": chem_embedding,

            "struct_tokens": struct_tokens,

            "struct_embedding": struct_embedding,

            "phys_embedding": phys_embedding,

            "seq_mask": seq_feat["attention_mask"],

            "chem_mask": chem_feat["attention_mask"],

            "struct_mask": struct_feat["attention_mask"],

        }

    ##############################################################
    # Forward
    ##############################################################

    def forward(

            self,

            seq_feat,

            chem_feat,

            struct_feat,

            phys_feat,

    ):
        mode = self.ablation_mode
        ##########################################################
        # Prepare Features
        ##########################################################

        feat = self._prepare_features(

            seq_feat,

            chem_feat,

            struct_feat,

            phys_feat,

        )
        mode = Config.ABLATION_MODE
        ##########################################################
        # Ablation : ALL_NO_ATTENTION
        ##########################################################

        if mode == "ALL_NO_ATTENTION":
            ######################################################
            # Pool each modality
            ######################################################

            seq_embedding = self.masked_mean_pooling(

                feat["seq_tokens"],

                feat["seq_mask"]

            )

            ######################################################

            concat = torch.cat(

                [

                    seq_embedding,

                    feat["chem_embedding"],

                    feat["struct_embedding"],

                    feat["phys_embedding"]

                ],

                dim=-1

            )

            ######################################################

            fused_embedding = self.concat_projection(

                concat

            )

            ######################################################

            return {

                "embedding": fused_embedding,

                "chem_attention": None,

                "struct_attention": None,

                "phys_attention": None

            }


        ##########################################################
        # Sequence attends to Chemical
        ##########################################################

        ##########################################################
        # Sequence → Chemical
        ##########################################################

        if mode == "SEQ":

            seq_chem = torch.zeros_like(feat["seq_tokens"])

            chem_weights = None

        else:

            seq_chem, chem_weights = self.seq_to_chem(

                feat["seq_tokens"],

                feat["chem_tokens"],

                feat["chem_tokens"],

                feat["chem_mask"]

            )

        ##########################################################
        # Sequence attends to Structure
        ##########################################################

        ##########################################################
        # Sequence → Structure
        ##########################################################

        if mode in [

            "SEQ",

            "SEQ_CHEM"

        ]:

            seq_struct = torch.zeros_like(

                feat["seq_tokens"]

            )

            struct_weights = None

        else:

            seq_struct, struct_weights = self.seq_to_struct(

                feat["seq_tokens"],

                feat["struct_tokens"],

                feat["struct_tokens"],

                feat["struct_mask"]

            )
        ##########################################################
        # PhysChem branch
        ##########################################################

        ##########################################################
        # Sequence → PhysChem
        ##########################################################

        if mode in [

            "SEQ",

            "SEQ_CHEM",

            "SEQ_CHEM_STRUCT"

        ]:

            seq_phys = torch.zeros_like(

                feat["seq_tokens"]

            )

            phys_weights = None

        else:

            phys = feat["phys_embedding"].unsqueeze(1)

            seq_phys, phys_weights = self.seq_to_phys(

                feat["seq_tokens"],

                phys,

                phys,

                None

            )
        ##########################################################
        # Residual Fusion
        ##########################################################

        fused = (

                feat["seq_tokens"]

                + seq_chem

                + seq_struct

                + seq_phys

        )

        fused = self.norm1(fused)

        ##########################################################
        # Feed Forward
        ##########################################################

        fused = fused + self.dropout(

            self.ffn(fused)

        )

        fused = self.norm2(fused)

        ##########################################################
        # CLS / Global Pooling
        ##########################################################

        fused_embedding = fused.mean(dim=1)

        ##########################################################

        return {

            "embedding": fused_embedding,

            "chem_attention": chem_weights,

            "struct_attention": struct_weights,

            "phys_attention": phys_weights

        }

    ##############################################################
    # Extract Embedding
    ##############################################################

    @torch.no_grad()
    def extract_embedding(
            self,
            seq_feat,
            chem_feat,
            struct_feat,
            phys_feat,
    ):
        """
        Returns the fused embedding before classifier.
        Shape:
            (B, hidden_dim)
        """

        self.eval()

        embedding = self.forward(
            seq_feat,
            chem_feat,
            struct_feat,
            phys_feat,
        )

        return embedding

    ##############################################################
    # Feature Dimension
    ##############################################################

    @property
    def output_dim(self):
        return self.hidden_dim

    ##############################################################
    # Debug Shapes
    ##############################################################

    @torch.no_grad()
    def debug_forward(
            self,
            seq_feat,
            chem_feat,
            struct_feat,
            phys_feat,
    ):
        embedding = self.forward(
            seq_feat,
            chem_feat,
            struct_feat,
            phys_feat,
        )

        print("=" * 60)
        print("HubCrossAttention V2")
        print("=" * 60)

        print("Output Shape :", embedding.shape)

        print("NaN :", torch.isnan(embedding).any())

        print("Min :", embedding.min().item())

        print("Max :", embedding.max().item())

        return embedding
##############################################################
# Dry Test for HubCrossAttention
##############################################################

if __name__ == "__main__":

    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ##########################################################
    # Hyperparameters
    ##########################################################

    batch_size = 4

    seq_len = 90

    chem_len = 256

    struct_len = 88

    hidden_dim = 512

    ##########################################################
    # Fake Inputs
    ##########################################################

    seq_feat = {

        "projected_tokens": torch.randn(
            batch_size,
            seq_len,
            480,
            device=device
        ),

        "attention_mask": torch.ones(
            batch_size,
            seq_len,
            dtype=torch.long,
            device=device
        )
    }

    ##########################################################

    chem_feat = {

        "projected_tokens": torch.randn(
            batch_size,
            chem_len,
            384,
            device=device
        ),

        "attention_mask": torch.ones(
            batch_size,
            chem_len,
            dtype=torch.long,
            device=device
        )
    }

    ##########################################################

    struct_feat = {

        "projected_tokens": torch.randn(
            batch_size,
            struct_len,
            512,
            device=device
        ),

        "attention_mask": torch.ones(
            batch_size,
            struct_len,
            dtype=torch.long,
            device=device
        )
    }

    ##########################################################

    phys_feat = {

        "physchem_embedding": torch.randn(
            batch_size,
            512,
            device=device
        )
    }

    ##########################################################
    # Model
    ##########################################################

    model = HubCrossAttention(

        hidden_dim=hidden_dim

    ).to(device)

    model.eval()

    ##########################################################

    with torch.no_grad():
        output = model(
            seq_feat,
            chem_feat,
            struct_feat,
            phys_feat
        )

        embedding = output["embedding"]

        print(embedding.shape)

    ##########################################################
    # Print Results
    ##########################################################

    print("=" * 60)

    print("HubCrossAttention Version 2")

    print("=" * 60)

    print()

    print("Output Shape")

    print(embedding.shape)

    print()

    print("NaN :", torch.isnan(embedding).any())

    print()

    print("Min :", embedding.min().item())

    print("Max :", embedding.max().item())

    print()

    print("Mean :", embedding.mean().item())

    print("Std  :", embedding.std().item())

    print()

    print("Dry Test Passed.")
