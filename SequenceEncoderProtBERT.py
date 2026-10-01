from transformers import BertTokenizer
from transformers import BertModel

from config import Config
from SequenceEncoderBase import SequenceEncoderBase


class SequenceEncoderProtBERT(SequenceEncoderBase):

    def __init__(self):

        super().__init__()

        self.tokenizer = BertTokenizer.from_pretrained(
            "Rostlab/prot_bert",
            do_lower_case=False
        )

        self.encoder = BertModel.from_pretrained(
            Config.ProtBERT_MODEL
        )

    ########################################################
    # ProtBERT tokenizer
    ########################################################

    def tokenize(self, sequences):

        if isinstance(sequences, str):
            sequences = " ".join(list(sequences))
        else:
            sequences = [
                " ".join(list(seq))
                for seq in sequences
            ]

        return self.tokenizer(

            sequences,

            truncation=True,

            padding="max_length",

            max_length=Config.MAX_SEQ_LENGTH,

            return_tensors="pt"

        )