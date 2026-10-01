"""
=============================================================
KAGB Multi-Modal Framework
Config
=============================================================

Author : Amir Lakizadeh Lab
Project: Anticancer Peptide Prediction using
         Sequential Progressive Hub Refinement (SPHR)

=============================================================
"""

import torch


class Config:
    """
    Global configuration of the whole framework.
    """
    # Encoder output dimensions
    #SEQUENCE_EMBED_DIM = 480  # ESM2 t12_35M
    #SEQUENCE_EMBED_DIM = 1280  # esm2_t33_650M_UR50D
    SEQUENCE_EMBED_DIM = 1024 #ProtBERT

    CHEMICAL_EMBED_DIM = 384  # یا مقدار واقعی ChemBERTa
    STRUCTURE_EMBED_DIM = 256  # GCN output
    PHYSCHEM_DIM = 128

    SHARED_DIM = 384 #512
    # ==========================================================
    # Device
    # ==========================================================

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    RANDOM_SEED = 123

    NUM_WORKERS = 4

    PIN_MEMORY = True

    # ==========================================================
    # Dataset
    # ==========================================================

    MAX_SEQ_LENGTH = 256

    MAX_SMILES_LENGTH = 256

    NUM_CLASSES = 2

    # ==========================================================
    # Protein Language Model
    # ==========================================================

    ESM_MODEL = "facebook/esm2_t12_35M_UR50D"
    #ESM_MODEL = "facebook/esm2_t33_650M_UR50D"
    ProtBERT_MODEL = "Rostlab/prot_bert"


    SEQUENCE_ENCODER = "ProtBERT"#defulat seq encoder
    #SEQUENCE_ENCODER = "ESM"

    # گزینه‌های دیگر:
    # facebook/esm2_t30_150M_UR50D
    # facebook/esm2_t33_650M_UR50D
    # facebook/esm2_t36_3B_UR50D

    # ==========================================================
    # Chemical Language Model
    # ==========================================================

    CHEM_MODEL = "DeepChem/ChemBERTa-77M-MLM"
    #CHEM_MODEL = "DeepChem/MoLFormer-c3-1.1B"
    #CHEM_MODEL = "DeepChem/ChemBERTa-77M-MTR"

    #CHEM_EMBED_DIM = 768

    # گزینه‌های جایگزین
    # ChemBERTa-3
    # ChemFormer
    # MolFormer

    # ==========================================================
    # Structure Encoder
    # (Future Work)

    # ==========================================================
    # Physicochemical Features
    # ==========================================================

    PHYSCHEM_DIM = 128
    NUM_PHYS_FEATURES = 61

    # ==========================================================
    # Shared Latent Space
    # ==========================================================

    PROJECTION_DROPOUT = 0.20

    # ==========================================================
    # Multi-head Cross Attention
    # ==========================================================

    NUM_HEADS = 8

    ATTENTION_DROPOUT = 0.20

    FFN_EXPANSION = 4

    # ==========================================================
    # Classifier
    # ==========================================================

    # MLP_HIDDEN1 = 256
    #
    # MLP_HIDDEN2 = 128
    # HIDDEN_DIM = 512

    CLASSIFIER_DROPOUT = 0.10
    NUM_POSITIVE = 376

    ##########################################################
    # Embedding Files
    ##########################################################

    TRAIN_EMBEDDINGS = "./embeddings/FULL/train_embeddings.pt"

    VAL_EMBEDDINGS = "./embeddings/FULL/val_embeddings.pt"

    TEST_EMBEDDINGS = "./embeddings/FULL/test_embeddings.pt"
    # ==========================================================
    # Training
    # ==========================================================

    BATCH_SIZE = 16

    EPOCHS = 250
    MIN_EPOCH = 100

    MIN_DELTA = 1e-4

    LEARNING_RATE = 3e-5
    LR_FACTOR = 0.1


    LR_PATIENCE = 5

    WEIGHT_DECAY = 1e-4
    DROPOUT = 0.1
    PDB_DIR = "PDB"

    DATASET_CSV = "./dataset/dataset_all.csv"
    TRAIN_CSV = "./dataset/train.csv"
    VAL_CSV = "./dataset/val.csv"
    TEST_CSV = "./dataset/test.csv"
    EARLY_STOPPING = 20

    THRESHOLD = 0.50
    ##########################################################
    # Ablation Study
    ##########################################################

    ABLATION_MODE = "FULL"

    # گزینه‌ها:
    # "SEQ"
    # "SEQ_CHEM"
    # "SEQ_CHEM_STRUCT"
    # "ALL_NO_ATTENTION"
    # "FULL"

    CHECKPOINT_PATH = "./checkpoints/best_model.pt"
    CHECKPOINT_DIR = "./checkpoints"

    # ==========================================================
    # Contrastive Learning
    # (Future Extension)
    # ==========================================================

    TEMPERATURE = 0.10

    CONTRASTIVE_WEIGHT = 0.10

    # ==========================================================
    # Saving
    # ==========================================================

    MODEL_NAME = "HCMCA_ACP"

    SAVE_DIR = "./checkpoints"

    LOG_DIR = "./logs"

    # ==========================================================
    # Print configuration
    # ==========================================================

    @staticmethod
    def print():

        print("=" * 60)

        print("KAGB Configuration")

        print("=" * 60)

        for key, value in Config.__dict__.items():

            if not key.startswith("__") and not callable(value):

                print(f"{key:25s}: {value}")

        print("=" * 60)