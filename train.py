import config
import numpy as np
import sentencepiece as spm
import tensorflow as tf
from typing import List


def load_corpus() -> List[str]:
    with open(f"{config.SPM_TRAINER_CONFIG['input']}", "r") as file:
        return file.readlines()


def load_tokenizer() -> spm.SentencePieceProcessor:
    model_path = f"{config.SPM_TRAINER_CONFIG['model_prefix']}.model"
    return spm.SentencePieceProcessor(model_file=model_path)


def postprocess_token_sequences(
        sequences: List[List[int]],
        max_length: int = config.PADDED_SEQUENCE_LENGTH,
        pad_id: int = config.SPM_TRAINER_CONFIG["pad_id"],
        bos_id: int = config.SPM_TRAINER_CONFIG["bos_id"],
        eos_id: int = config.SPM_TRAINER_CONFIG["eos_id"],
) -> np.array:
    """
    Attach BOS, EOS token to each sequence and pad each to maximum length
    :param sequences: list of token sequences
    :param max_length: length of padded sequence
    :param pad_id: id of pad token
    :param bos_id: id of bos token
    :param eos_id: id of eos token
    :return: array of processed token sequences
    """
    real_token_max_num = max_length - 2  # room for bos_id, eos_id
    processed_sequences = []
    for seq in sequences:
        trimmed_sequence = seq[:real_token_max_num][:]
        trimmed_sequence.append(eos_id)
        processed_sequences.append([bos_id] + trimmed_sequence)
    return tf.keras.utils.pad_sequences(
        sequences=processed_sequences,
        maxlen=max_length,
        dtype="int32",
        padding="post",
        truncating="post",
        value=pad_id,
    )
