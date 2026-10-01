"""
=========================================================
Chemical Encoder
ChemBERTa
=========================================================
"""

import torch
import torch.nn as nn

from transformers import AutoTokenizer
from transformers import AutoModel

from config import Config
from projection import ProjectionBlock
from pooling import MeanPooling


class ChemicalEncoder_MolF(nn.Module):

    """
    Chemical Encoder based on ChemBERTa

    Input
    -----
    List[str]
        SMILES strings

    Output
    ------
    token_embeddings
    projected_tokens
    chemical_embedding
    attention_mask
    """

    def __init__(self):

        super().__init__()

        ##############################################
        # Tokenizer
        ##############################################

        self.tokenizer = AutoTokenizer.from_pretrained(
            Config.CHEM_MODEL,
            trust_remote_code=True
        )

        ##############################################
        # Backbone
        ##############################################

        self.encoder = AutoModel.from_pretrained(
            Config.CHEM_MODEL,
            trust_remote_code=True
        )

        ##############################################
        # Projection
        ##############################################
        hidden_dim = self.encoder.config.hidden_size
        self.projection = ProjectionBlock(

            input_dim=hidden_dim,

            shared_dim=Config.SHARED_DIM,

            dropout=Config.PROJECTION_DROPOUT

        )

        ##############################################
        # Pooling
        ##############################################

        self.pooling = MeanPooling()

    ##################################################
    # Tokenizer
    ##################################################

    def tokenize(self, smiles):

        return self.tokenizer(

            smiles,

            padding=True,

            truncation=True,

            max_length=Config.MAX_SMILES_LENGTH,

            return_tensors="pt"

        )

    ##################################################
    # Forward
    ##################################################

    def forward(self, smiles):

        ##############################################

        inputs = self.tokenize(smiles)

        ##############################################

        device = next(self.parameters()).device

        inputs = {
            k: v.to(device)
            for k, v in inputs.items()
        }

        ##############################################

        outputs = self.encoder(

            input_ids=inputs["input_ids"],

            attention_mask=inputs["attention_mask"]

        )
        print(type(outputs))
        print(outputs)

        ##############################################

        token_embeddings = outputs.last_hidden_state

        ##############################################
        # Project every token
        ##############################################

        projected_tokens = self.projection(

            token_embeddings

        )

        ##############################################

        chemical_embedding = self.pooling(

            projected_tokens,

            inputs["attention_mask"]

        )

        ##############################################

        return {

            "token_embeddings": token_embeddings,

            "projected_tokens": projected_tokens,

            "chemical_embedding": chemical_embedding,

            "attention_mask": inputs["attention_mask"]

        }