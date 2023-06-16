import config
import tensorflow as tf
import time
import numpy as np
from typing import Union, Dict


class MLMDataCollateLayer(tf.keras.layers.Layer):

    def __init__(
            self,
            random_seed: Union[float, int] = time.time(),
            name: str = "mlm_data_collate_layer",
    ):
        super(MLMDataCollateLayer, self).__init__(name=name, trainable=False)
        self._vocab_size = config.SPM_TRAINER_CONFIG["vocab_size"]
        self._plain_token_range_lb = len(config.SPECIAL_TOKENS)
        self._mask_token_id = config.SPECIAL_TOKENS["mask_token"]["id"]
        self._non_mask_id = config.NON_MASK_ID
        self._mask_prob = 0.15
        self._mask_token_replace_prob = 0.8
        self._random_token_replace_threshold = 0.9
        self._rg = tf.random.Generator.from_seed(random_seed)

    def call(self, inputs: np.array) -> Dict[str, tf.Tensor]:
        is_mlm_mask = tf.logical_and(
            x=inputs >= self._plain_token_range_lb,
            y=self._rg.uniform(inputs.shape) <= self._mask_prob,
            name="select_masked_token",
        )
        target_labels = tf.where(
            condition=is_mlm_mask,
            x=inputs,
            y=self._non_mask_id,
            name="generate_target_labels"
        )
        mask_type_probs = self._rg.uniform(inputs.shape)
        random_tokens = tf.cast(
            x=self._rg.uniform(inputs.shape) * self._vocab_size,
            dtype=tf.dtypes.int32,
            name="generate_random_tokens",
        )
        masked_inputs = tf.where(
            condition=is_mlm_mask & (mask_type_probs > self._random_token_replace_threshold),
            x=random_tokens,
            y=inputs,
            name="allocate_random_token"
        )
        masked_inputs = tf.where(
            condition=is_mlm_mask & (mask_type_probs <= self._mask_token_replace_prob),
            x=self._mask_token_id,
            y=masked_inputs,
            name="allocate_mask_token",
        )
        return {"input_ids": masked_inputs, "labels": target_labels}


