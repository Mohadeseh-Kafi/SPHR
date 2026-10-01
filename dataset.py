import os

import numpy as np
import pandas as pd
import torch

from torch.utils.data import Dataset

from config import Config


class HCMCADataset(Dataset):

    def __init__(self, csv_file):

        super().__init__()

        self.data = pd.read_csv(csv_file)

    def __len__(self):

        return len(self.data)

    def __getitem__(self, idx):

        row = self.data.iloc[idx]

        ##################################################
        # Sequence
        ##################################################

        sequence = row["sequence"]

        ##################################################
        # SMILES
        ##################################################

        smiles = row["smiles"]

        ##################################################
        # PDB
        ##################################################

        pdb_path = row["pdb_path"]

        # -------------------------------
        # Fix NON_ACP numbering
        # -------------------------------
        if "NON_ACP_" in pdb_path:
            filename = os.path.basename(pdb_path)

            idx = int(
                filename.replace("NON_ACP_", "")
                .replace(".pdb", "")
            )

            idx -= Config.NUM_POSITIVE

            filename = f"NON_ACP_{idx}.pdb"

            pdb_path = os.path.join(
                Config.PDB_DIR,
                filename
            )

        ##################################################
        # Physchem
        ##################################################

        if pd.isna(row["physchem"]):

            physchem = torch.empty(0)

        else:

            physchem = torch.tensor(

                np.fromstring(

                    row["physchem"],

                    sep=",",

                    dtype=np.float32

                ),

                dtype=torch.float32

            )

        ##################################################
        # Label
        ##################################################

        label = torch.tensor(

            row["label"],

            dtype=torch.float32

        )

        ##################################################
        # Return
        ##################################################

        return {

            "id": row["id"],

            "sequence": sequence,

            "smiles": smiles,

            "pdb_path": pdb_path,

            "physchem": physchem,

            "label": label

        }