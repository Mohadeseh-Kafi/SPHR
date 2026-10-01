"""
=============================================================
model_v2.py

SPHR Version 2

Training using Precomputed Embeddings

Only Fusion Network

=============================================================
"""

import torch
import torch.nn as nn

from config import Config

##############################################################
# Project Modules
##############################################################

from cross_attention import HubCrossAttention

from fusion import GatedFusion

from classifier import MLPClassifier
##############################################################
# SPHR Version 2
##############################################################

class HCMCAModelV2(nn.Module):

    def __init__(self):

        super().__init__()

        ######################################################
        # Dimensions
        ######################################################

        #self.hidden_dim = Config.HIDDEN_DIM

        ######################################################
        # Cross Attention
        ######################################################

        self.cross_attention = HubCrossAttention(

            seq_dim=Config.SHARED_DIM,

            chem_dim=Config.SHARED_DIM,

            struct_dim=Config.SHARED_DIM,

            phys_dim=Config.SHARED_DIM,

            hidden_dim=Config.SHARED_DIM,
            num_heads = Config.NUM_HEADS,
            dropout = Config.DROPOUT

        )

        ######################################################
        # Fusion
        ######################################################

        self.fusion = GatedFusion()

        ######################################################
        # Classifier
        ######################################################

        self.classifier = MLPClassifier()

    ##############################################################
    # Prepare Inputs
    ##############################################################

    def _prepare_inputs(self, batch):
        ##########################################################
        # Sequence
        ##########################################################

        seq_feat = {

            "projected_tokens":

                batch["sequence_tokens"],

            "attention_mask":

                batch["sequence_mask"]

        }

        ##########################################################
        # Chemical
        ##########################################################

        chem_feat = {

            "projected_tokens":

                batch["chemical_tokens"],

            "attention_mask":

                batch["chemical_mask"]

        }

        ##########################################################
        # Structure
        ##########################################################

        struct_feat = {

            "projected_tokens":

                batch["structure_tokens"],

            "attention_mask":

                batch["structure_mask"]

        }

        ##########################################################
        # PhysChem
        ##########################################################

        phys_feat = {

            "physchem_embedding":

                batch["physchem"]

        }

        ##########################################################

        return (

            seq_feat,

            chem_feat,

            struct_feat,

            phys_feat

        )

    ##############################################################
    # Forward
    ##############################################################

    def forward(self, batch):
        ##########################################################
        # Prepare inputs
        ##########################################################

        seq_feat, chem_feat, struct_feat, phys_feat = self._prepare_inputs(batch)

        ##########################################################
        # Hub Cross Attention
        ##########################################################

        hub_outputs = self.cross_attention(
            seq_feat,
            chem_feat,
            struct_feat,
            phys_feat
        )

        ##########################################################
        embedding = hub_outputs["embedding"]

        logits = self.classifier(

            embedding

        )

        ##########################################################

        return logits, embedding

    ##############################################################
    # Extract Embedding
    ##############################################################

    @torch.no_grad()
    def extract_embedding(self, batch):
        self.eval()

        seq_feat, chem_feat, struct_feat, phys_feat = self._prepare_inputs(batch)

        hub_output = self.cross_attention(

            seq_feat,

            chem_feat,

            struct_feat,

            phys_feat

        )

        fused_embedding = self.fusion(

            hub_output["embedding"]

        )

        return fused_embedding

    ##############################################################
    # Predict Probability
    ##############################################################

    @torch.no_grad()
    def predict_proba(self, batch):
        self.eval()

        logits = self.forward(batch)

        return torch.sigmoid(logits)

    ##############################################################
    # Predict Label
    ##############################################################

    @torch.no_grad()
    def predict(self, batch, threshold=0.5):
        probs = self.predict_proba(batch)

        return (probs > threshold).float()
##############################################################
# Dry Test
##############################################################

if __name__ == "__main__":

    from EmbeddingDataset import EmbeddingDataset
    from dataloader import create_dataloader
    from config import Config

    print("=" * 60)
    print("SPHR Version 2 Dry Test")
    print("=" * 60)

    ##########################################################

    dataset = EmbeddingDataset(

        Config.TRAIN_EMBEDDINGS

    )

    loader = create_dataloader(

        dataset,

        batch_size=4,

        shuffle=False

    )

    ##########################################################

    model = HCMCAModelV2()

    model.eval()

    ##########################################################

    batch = next(iter(loader))

    ##########################################################

    print()

    print("Batch Keys")

    print(batch.keys())

    ##########################################################

    with torch.no_grad():

        logits = model(batch)

    ##########################################################

    print()

    print("Output Shape")

    print(logits.shape)
    print()

    print("NaN :", torch.isnan(logits).any())

    print("Min :", logits.min().item())

    print("Max :", logits.max().item())

    ##########################################################

    print()

    print("Logits")

    print(logits)

    ##########################################################

    print()

    print("Probability")

    print(torch.sigmoid(logits))

    ##########################################################

    print()

    print("Dry Test Passed")

