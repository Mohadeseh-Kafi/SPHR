"""
=============================================================
precompute_embeddings.py
SPHR Version 2

Outputs
-------
train_embeddings.pt
val_embeddings.pt
test_embeddings.pt

Author:
Amir Lakizadeh
=============================================================
"""


import os

from pathlib import Path

import torch
from tqdm import tqdm

from SequenceEncoderESM import SequenceEncoderESM

from SequenceEncoderProtBERT import SequenceEncoderProtBERT

from config import Config
from dataset import HCMCADataset
from dataloader import create_dataloader

# ------------------------------------------------------------
# Encoders
# ------------------------------------------------------------

#from SequenceEncoder import SequenceEncoder

from ChemicalEncoder import ChemicalEncoder

#from ChemicalEncoder_MolF import ChemicalEncoder_MolF

from StructureEncoder import StructureEncoder
from PhysChemEncoder import PhysChemEncoder

# ------------------------------------------------------------
# Device
# ------------------------------------------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 60)
print("SPHR Embedding Generator")
print("=" * 60)
print(f"Device : {device}")

# ------------------------------------------------------------
# Output directory
# ------------------------------------------------------------

OUTPUT_DIR = Path("embeddings")

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

print(f"Embedding directory : {OUTPUT_DIR.resolve()}")

def pad_tokens(tokens, max_len):

    """
    tokens:
        (B,L,D)

    returns

        (B,max_len,D)
    """

    B, L, D = tokens.shape

    if L == max_len:
        return tokens

    if L > max_len:
        return tokens[:, :max_len, :]

    pad = torch.zeros(

        B,

        max_len - L,

        D,

        dtype=tokens.dtype,

        device=tokens.device

    )

    return torch.cat(

        [tokens, pad],

        dim=1

    )

def pad_mask(mask, max_len):

    """
    mask:

        (B,L)

    -->

        (B,max_len)

    """

    B, L = mask.shape

    if L == max_len:
        return mask

    if L > max_len:
        return mask[:, :max_len]

    pad = torch.zeros(

        B,

        max_len - L,

        dtype=mask.dtype,

        device=mask.device

    )

    return torch.cat(

        [mask, pad],

        dim=1

    )

# ============================================================
# Load Encoders
# ============================================================

def load_encoders():
    """
    Load all pretrained encoders only once.

    Returns
    -------
    sequence_encoder
    chemical_encoder
    structure_encoder
    physchem_encoder
    """

    print("\nLoading Encoders ...")

    ############################################################
    # Sequence Encoder
    ############################################################

    if Config.SEQUENCE_ENCODER == "ESM":

        sequence_encoder = SequenceEncoderESM().to(device)

    else:

        sequence_encoder = SequenceEncoderProtBERT().to(device)

    ############################################################
    # Chemical Encoder
    ############################################################

    chemical_encoder = ChemicalEncoder().to(device)

    ############################################################
    # Structure Encoder
    ############################################################

    structure_encoder = StructureEncoder().to(device)

    ############################################################
    # PhysChem Encoder
    ############################################################

    physchem_encoder = PhysChemEncoder().to(device)

    ############################################################
    # Evaluation Mode
    ############################################################

    sequence_encoder.eval()
    chemical_encoder.eval()
    structure_encoder.eval()
    physchem_encoder.eval()

    print("Sequence Encoder  : Loaded")
    print("Chemical Encoder  : Loaded")
    print("Structure Encoder : Loaded")
    print("PhysChem Encoder  : Loaded")

    return (
        sequence_encoder,
        chemical_encoder,
        structure_encoder,
        physchem_encoder
    )
# ============================================================
# Encode One Batch
# ============================================================

@torch.no_grad()
def encode_batch(
    batch,
    sequence_encoder,
    chemical_encoder,
    structure_encoder,
    physchem_encoder
):
    ############################################################
    # Sequence
    ############################################################
    seq_output = sequence_encoder(batch["sequence"])

    ############################################################
    # Chemical
    ############################################################
    if Config.ABLATION_MODE in ["FULL", "ALL_NO_ATTENTION", "SEQ_CHEM", "SEQ_CHEM_STRUCT"]:
        chem_output = chemical_encoder(batch["smiles"])
    else:
        chem_output = None

    ############################################################
    # Structure
    ############################################################
    if Config.ABLATION_MODE in ["FULL", "ALL_NO_ATTENTION","SEQ_CHEM_STRUCT"]:
        struct_output = structure_encoder(batch["pdb_path"])
    else:
        struct_output = None

    ############################################################
    # PhysChem
    ############################################################
    if Config.ABLATION_MODE in ["FULL", "ALL_NO_ATTENTION"]:
        phys_output = physchem_encoder(batch["physchem"].to(device))
    else:
        phys_output = None
    ############################################################
    # Move everything to CPU
    ############################################################

    ############################################################
    # Dummy tensors for Ablation
    ############################################################

    B = seq_output["projected_tokens"].shape[0]

    ############################################################
    # Chemical
    ############################################################

    if chem_output is None:

        chemical_tokens = torch.zeros(
            B,
            Config.MAX_SEQ_LENGTH,
            Config.SHARED_DIM
        )

        chemical_mask = torch.zeros(
            B,
            Config.MAX_SEQ_LENGTH,
            dtype=torch.long
        )

    else:

        chemical_tokens = pad_tokens(
            chem_output["projected_tokens"],
            Config.MAX_SEQ_LENGTH
        ).cpu()

        chemical_mask = pad_mask(
            chem_output["attention_mask"],
            Config.MAX_SEQ_LENGTH
        ).cpu()

    ############################################################
    # Structure
    ############################################################

    if struct_output is None:

        structure_tokens = torch.zeros(
            B,
            Config.MAX_SEQ_LENGTH,
            Config.SHARED_DIM
        )

        structure_mask = torch.zeros(
            B,
            Config.MAX_SEQ_LENGTH,
            dtype=torch.long
        )

    else:

        structure_tokens = pad_tokens(
            struct_output["projected_tokens"],
            Config.MAX_SEQ_LENGTH
        ).cpu()

        structure_mask = pad_mask(
            struct_output["attention_mask"],
            Config.MAX_SEQ_LENGTH
        ).cpu()

    ############################################################
    # PhysChem
    ############################################################

    if phys_output is None:

        physchem_embedding = torch.zeros(
            B,
            Config.SHARED_DIM
        )

    else:

        physchem_embedding = phys_output["physchem_embedding"].cpu()

    ############################################################
    # Final Output
    ############################################################

    output = {

        "ids":
            batch["id"],

        ########################################################

        "sequence_tokens":
            pad_tokens(
                seq_output["projected_tokens"],
                Config.MAX_SEQ_LENGTH
            ).cpu(),

        "sequence_mask":
            pad_mask(
                seq_output["attention_mask"],
                Config.MAX_SEQ_LENGTH
            ).cpu(),

        ########################################################

        "chemical_tokens":
            chemical_tokens,

        "chemical_mask":
            chemical_mask,

        ########################################################

        "structure_tokens":
            structure_tokens,

        "structure_mask":
            structure_mask,

        ########################################################

        "physchem":
            physchem_embedding,

        ########################################################

        "labels":
            batch["label"].cpu()

    }

    return output
# ============================================================
# Process One Dataset Split
# ============================================================

def process_split(
    csv_file,
    output_file,
    sequence_encoder,
    chemical_encoder,
    structure_encoder,
    physchem_encoder
):
    """
    Generate embeddings for one dataset split.

    Parameters
    ----------
    csv_file : str

    output_file : str
    """

    print("=" * 60)
    print(f"Processing : {csv_file}")
    print("=" * 60)

    ############################################################
    # Dataset
    ############################################################

    dataset = HCMCADataset(
        csv_file=csv_file
    )

    loader = create_dataloader(
        dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False
    )

    ############################################################
    # Containers
    ############################################################

    ids = []

    sequence_tokens = []
    sequence_masks = []

    chemical_tokens = []
    chemical_masks = []

    structure_tokens = []
    structure_masks = []

    physchem_features = []

    labels = []

    ############################################################
    # Iterate over batches
    ############################################################

    progress = tqdm(loader)

    for batch in progress:

        emb = encode_batch(

            batch,

            sequence_encoder,

            chemical_encoder,

            structure_encoder,

            physchem_encoder

        )

        ########################################################
        # IDs
        ########################################################

        ids.extend(
            emb["ids"]
        )

        ########################################################
        # Sequence
        ########################################################

        sequence_tokens.append(
            emb["sequence_tokens"]
        )

        sequence_masks.append(
            emb["sequence_mask"]
        )

        ########################################################
        # Chemical
        ########################################################

        chemical_tokens.append(
            emb["chemical_tokens"]
        )

        chemical_masks.append(
            emb["chemical_mask"]
        )

        ########################################################
        # Structure
        ########################################################

        structure_tokens.append(
            emb["structure_tokens"]
        )

        structure_masks.append(
            emb["structure_mask"]
        )

        ########################################################
        # PhysChem
        ########################################################

        physchem_features.append(
            emb["physchem"]
        )

        ########################################################
        # Labels
        ########################################################

        labels.append(
            emb["labels"]
        )

    ############################################################
    # Concatenate tensors
    ############################################################

    output = {

        "ids":

            ids,

        ########################################################

        "sequence_tokens":

            torch.cat(sequence_tokens, dim=0),

        "sequence_mask":

            torch.cat(sequence_masks, dim=0),

        ########################################################

        "chemical_tokens":

            torch.cat(chemical_tokens, dim=0),

        "chemical_mask":

            torch.cat(chemical_masks, dim=0),

        ########################################################

        "structure_tokens":

            torch.cat(structure_tokens, dim=0),

        "structure_mask":

            torch.cat(structure_masks, dim=0),

        ########################################################

        "physchem":

            torch.cat(physchem_features, dim=0),

        ########################################################

        "labels":

            torch.cat(labels, dim=0)

    }

    ############################################################
    # Save
    ############################################################

    torch.save(

        output,

        OUTPUT_DIR / output_file

    )

    print()

    print(f"Saved : {OUTPUT_DIR / output_file}")

    print(f"Samples : {len(ids)}")

    print()
    # ============================================================
    # Main
    # ============================================================
if __name__ == "__main__":

    print()
    print("=" * 60)
    print("SPHR Embedding Generator")
    print("=" * 60)
    print()

    ###########################################################
    # Ablation modes
    ###########################################################

    MODES = [
        "SEQ",
        "SEQ_CHEM",
        "SEQ_CHEM_STRUCT",
        "ALL_NO_ATTENTION",
        "FULL"
    ]

    ###########################################################
    #######################################################
    # Load encoders
    #######################################################

    (
        sequence_encoder,
        chemical_encoder,
        structure_encoder,
        physchem_encoder
    ) = load_encoders()
    for mode in MODES:

        print()
        print("=" * 70)
        print(f"Generating Embeddings : {mode}")
        print("=" * 70)

        #######################################################
        # Current mode
        #######################################################

        Config.ABLATION_MODE = mode

        #######################################################
        # Output directory
        #######################################################

        OUTPUT_DIR = Path(f"./embeddings/{mode}")

        OUTPUT_DIR.mkdir(
            parents=True,
            exist_ok=True
        )


        #######################################################
        # Train
        #######################################################

        process_split(

            csv_file=Config.TRAIN_CSV,

            output_file="train_embeddings.pt",

            sequence_encoder=sequence_encoder,

            chemical_encoder=chemical_encoder,

            structure_encoder=structure_encoder,

            physchem_encoder=physchem_encoder

        )

        #######################################################
        # Validation
        #######################################################

        if hasattr(Config, "VAL_CSV"):

            process_split(

                csv_file=Config.VAL_CSV,

                output_file="val_embeddings.pt",

                sequence_encoder=sequence_encoder,

                chemical_encoder=chemical_encoder,

                structure_encoder=structure_encoder,

                physchem_encoder=physchem_encoder

            )

        #######################################################
        # Test
        #######################################################

        if hasattr(Config, "TEST_CSV"):

            process_split(

                csv_file=Config.TEST_CSV,

                output_file="test_embeddings.pt",

                sequence_encoder=sequence_encoder,

                chemical_encoder=chemical_encoder,

                structure_encoder=structure_encoder,

                physchem_encoder=physchem_encoder

            )

        #######################################################

        print(f"{mode} Finished.")

    ###########################################################

    print()
    print("=" * 60)
    print("All Embeddings Generated Successfully.")
    print("=" * 60)