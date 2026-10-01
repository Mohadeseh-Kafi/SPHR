"""
=========================================================
EmbeddingDataset.py
Version 2
=========================================================

Loads precomputed embeddings.

Author:
Amir Lakizadeh Lab
"""

import torch
from torch.utils.data import Dataset


class EmbeddingDataset(Dataset):

    def __init__(self, embedding_file):

        super().__init__()

        print(f"Loading embeddings : {embedding_file}")

        self.data = torch.load(
            embedding_file,
            map_location="cpu"
        )

        self.length = len(self.data["ids"])

        print(f"Loaded {self.length} samples")
        print()

        print("Embedding Shapes")

        print("-------------------------------")

        for k, v in self.data.items():

            if torch.is_tensor(v):
                print(f"{k:20s} {tuple(v.shape)}")

    ###########################################################

    def __len__(self):

        return self.length

    ###########################################################

    def __getitem__(self, idx):

        sample = {

            "id":
                self.data["ids"][idx],

            ##################################################
            # Sequence
            ##################################################

            "sequence_tokens":
                self.data["sequence_tokens"][idx].contiguous(),

            "sequence_mask":
                self.data["sequence_mask"][idx],

            ##################################################
            # Chemical
            ##################################################

            "chemical_tokens":
                self.data["chemical_tokens"][idx].contiguous(),

            "chemical_mask":
                self.data["chemical_mask"][idx],

            ##################################################
            # Structure
            ##################################################

            "structure_tokens":
                self.data["structure_tokens"][idx].contiguous(),

            "structure_mask":
                self.data["structure_mask"][idx],

            ##################################################
            # PhysChem
            ##################################################

            "physchem":
                self.data["physchem"][idx].float(),

            ##################################################
            # Label
            ##################################################

            "label":
                self.data["labels"][idx].float(),

        }

        return sample