from transformers import AutoTokenizer
from transformers import EsmModel

from config import Config
from SequenceEncoderBase import SequenceEncoderBase


class SequenceEncoderESM(SequenceEncoderBase):

    def __init__(self):

        super().__init__()

        self.tokenizer = AutoTokenizer.from_pretrained(
            Config.ESM_MODEL
        )

        self.encoder = EsmModel.from_pretrained(
            Config.ESM_MODEL
        )

    def tokenize(self, sequences):

        return self.tokenizer(

            sequences,

            truncation=True,

            padding="max_length",

            max_length=Config.MAX_SEQ_LENGTH,

            return_tensors="pt"

        )