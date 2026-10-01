from torch import nn

from pooling import MeanPooling
from projection import ProjectionBlock
from config import Config


class SequenceEncoderBase(nn.Module):

    def __init__(self):

        super().__init__()

        self.pooling = MeanPooling()

        self.projection = ProjectionBlock(
            input_dim=Config.SEQUENCE_EMBED_DIM,
            shared_dim=Config.SHARED_DIM,
            dropout=Config.PROJECTION_DROPOUT
        )

    ########################################################
    # Tokenizer
    ########################################################

    def tokenize(self, sequences):

        raise NotImplementedError

    ########################################################
    # Forward
    ########################################################

    def forward(self, sequences):

        inputs = self.tokenize(sequences)

        device = next(self.parameters()).device

        inputs = {
            k: v.to(device)
            for k, v in inputs.items()
        }

        outputs = self.encoder(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"]
        )

        token_embeddings = outputs.last_hidden_state

        sequence_embedding = self.pooling(
            token_embeddings,
            inputs["attention_mask"]
        )

        projected_tokens = self.projection(
            token_embeddings
        )

        return {

            "token_embeddings": token_embeddings,

            "sequence_embedding": sequence_embedding,

            "projected_tokens": projected_tokens,

            "attention_mask": inputs["attention_mask"]

        }