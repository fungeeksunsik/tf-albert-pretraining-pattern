import config
import numpy as np
import sentencepiece as spm
import tensorflow as tf
import time
from typing import List, Tuple


def load_corpus() -> List[str]:
    with open(f"{config.SPM_TRAINER_CONFIG['input']}", "r") as file:
        return file.readlines()


def load_tokenizer() -> spm.SentencePieceProcessor:
    model_path = f"{config.SPM_TRAINER_CONFIG['model_prefix']}.model"
    return spm.SentencePieceProcessor(model_file=model_path)


def postprocess_token_sequences(sequences: List[List[int]]) -> np.array:
    """
    Attach BOS, EOS token to each sequence and pad each to maximum length
    :param sequences: list of token sequences
    :return: array of processed token sequences
    """
    real_token_max_num = config.PADDED_SEQUENCE_LENGTH - 2  # room for bos_id, eos_id
    processed_sequences = []
    for seq in sequences:
        trimmed_sequence = seq[:real_token_max_num][:]
        trimmed_sequence.append(config.SPM_TRAINER_CONFIG["eos_id"])
        processed_sequences.append([config.SPM_TRAINER_CONFIG["bos_id"]] + trimmed_sequence)
    return tf.keras.utils.pad_sequences(
        sequences=processed_sequences,
        maxlen=config.PADDED_SEQUENCE_LENGTH,
        dtype="int32",
        padding="post",
        truncating="post",
        value=config.SPM_TRAINER_CONFIG["pad_id"],
    )


def apply_mlm_mask(
        inputs: np.array,
        rg: tf.random.Generator,
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    """
    Takes special-token(ex. bos, eos, pad) added sequences as input and returns tuple of tensors. Specifically, it
    returns following tensors consecutively to be passed into tf.data.Dataset API:
      1. mlm mask applied input tensor
      2. original input tensor
      3. float-casted boolean mask which indicates whether certain token was mlm mask
      4. 3-dimensional attention mask where each dimension represents (batch_size, num_heads, max_length) respectively
    :param inputs: trimmed, padded input sequences(output of `postprocess_token_sequences` method)
    :param rg: random number generator initiated from random_seed
    :return: tuple of output tensors
    """
    is_mlm_mask = tf.logical_and(
        x=inputs >= len(config.SPECIAL_TOKENS),
        y=rg.uniform(inputs.shape) <= 0.15,
        name="select_masked_token",
    )
    mask_type_probs = rg.uniform(inputs.shape)
    attention_mask = tf.expand_dims(
        input=inputs != config.SPECIAL_TOKENS["pad_token"]["id"],
        axis=1,  # dimension corresponds to num_heads will later be broadcasted within MHA layer
        name="generate_attention_mask",
    )
    random_tokens = tf.cast(
        x=rg.uniform(inputs.shape) * config.VOCAB_SIZE,
        dtype=tf.dtypes.int32,
        name="generate_random_tokens",
    )
    masked_inputs = tf.where(
        condition=is_mlm_mask & (mask_type_probs > 0.9),
        x=random_tokens,
        y=inputs,
        name="allocate_random_token"
    )
    masked_inputs = tf.where(
        condition=is_mlm_mask & (mask_type_probs <= 0.8),
        x=config.SPECIAL_TOKENS["mask_token"]["id"],
        y=masked_inputs,
        name="allocate_mask_token",
    )
    return (
        masked_inputs,
        tf.constant(inputs),
        tf.cast(is_mlm_mask, tf.dtypes.float32),
        attention_mask,
    )


def masked_data_generator() -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    """
    Generator that yields batch of masked sequences at every iteration
    :return: result of `apply_mlm_mask` method
    """
    corpus = load_corpus()
    tokenizer = load_tokenizer()
    random_seed = time.time()
    random_number_generator = tf.random.Generator.from_seed(random_seed)
    num_full_batches, remainder_exists = divmod(len(corpus), config.BATCH_SIZE)
    for batch_index in range(num_full_batches):
        index_lb = batch_index * config.BATCH_SIZE
        index_ub = (batch_index + 1) * config.BATCH_SIZE
        batch_output = tokenizer.encode(corpus[index_lb: index_ub])
        batch_output = postprocess_token_sequences(batch_output)
        yield apply_mlm_mask(batch_output, random_number_generator)
    if remainder_exists:
        index_lb = config.BATCH_SIZE * num_full_batches
        batch_output = tokenizer.encode(corpus[index_lb:])
        batch_output = postprocess_token_sequences(batch_output)
        yield apply_mlm_mask(batch_output, random_number_generator)


def create_masked_dataset() -> tf.data.Dataset:
    """
    Wrap overall masked data generating logic with tf.data.Dataset API
    :return: Tensorflow Dataset which returns batch of masked data in every iteration
    """
    return tf.data.Dataset.from_generator(
        generator=masked_data_generator,
        output_signature=(
            tf.TensorSpec(
                shape=(None, config.PADDED_SEQUENCE_LENGTH),
                dtype=tf.dtypes.int32,
                name="masked_inputs"
            ),
            tf.TensorSpec(
                shape=(None, config.PADDED_SEQUENCE_LENGTH),
                dtype=tf.dtypes.int32,
                name="original_inputs"
            ),
            tf.TensorSpec(
                shape=(None, config.PADDED_SEQUENCE_LENGTH),
                dtype=tf.dtypes.float32,
                name="sample_weights"
            ),
            tf.TensorSpec(
                shape=(None, 1, config.PADDED_SEQUENCE_LENGTH),
                dtype=tf.dtypes.float32,
                name="attention_mask"
            ),
        ),
        name="generate_masked_token_sequences"
    )
