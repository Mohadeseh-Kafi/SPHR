from config import Config

if Config.SEQUENCE_ENCODER == "ProtBERT":
    from SequenceEncoderProtBERT import SequenceEncoderProtBERT as SequenceEncoder
else:
    from SequenceEncoderESM import SequenceEncoderESM as SequenceEncoder