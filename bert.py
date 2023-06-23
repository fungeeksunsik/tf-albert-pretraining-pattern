import tensorflow as tf
import numpy as np
import config


class BertEmbeddings(tf.keras.layers.Layer):

    def __init__(self, name: str):
        """
        Additional layer normalization layer is attached as in huggingface transformers TFBertEmbeddings layer. Since
        IMDb corpus is not corpus of pair type documents(e.g. token type ids are always 0), segment embeddings is
        unnecessary. Therefore, they are not defined in this layer.
        :param name: name of the embedding layer
        """
        super(BertEmbeddings, self).__init__(name=name)
        with tf.name_scope(name):
            self.word_embeddings = tf.keras.layers.Embedding(
                input_dim=config.VOCAB_SIZE,
                output_dim=config.EMBED_DIM,
                name="word",
            )
            self.position_embeddings = tf.constant(
                value=self._generate_positional_encodings_weight(),
                dtype=tf.dtypes.float32,
                shape=(config.PADDED_SEQUENCE_LENGTH, config.EMBED_DIM),
                name="positional"
            )
            self.layernorm = tf.keras.layers.LayerNormalization(
                epsilon=config.LAYER_NORM_EPS, name="layer_normalization"
            )

    def call(self, inputs, *args, **kwargs):
        output = self.word_embeddings(inputs) + self.position_embeddings
        return self.layernorm(output)

    @staticmethod
    def _generate_positional_encodings_weight() -> np.array:
        """
        Implement positional encoding value generator as described in clause 3.5 of original Vaswani (2017) paper.
        reference: https://www.tensorflow.org/text/tutorials/transformer#the_embedding_and_positional_encoding_layer
        :return: (max_length, word_emb_dimension) shaped positional encodings
        """
        dimensions = np.repeat(np.arange(config.EMBED_DIM // 2), 2) * 2
        dimensions = 1 / np.power(10000, dimensions / config.EMBED_DIM)
        positions = np.arange(config.PADDED_SEQUENCE_LENGTH)
        encodings = np.outer(positions, dimensions)
        encodings[:, 0::2] = np.sin(encodings[:, 0::2])
        encodings[:, 1::2] = np.cos(encodings[:, 1::2])
        return encodings


class BertAttention(tf.keras.layers.Layer):

    def __init__(self, name: str):
        """
        Implementation of BERT layer contains 2 sublayers as explained in clause 3.1 of original Vaswani(2017) paper.
        In huggingface Transformer, this layer corresponds to TFBertAttention layer.
        :param name: name of current bert attention layer
        """
        super(BertAttention, self).__init__(name=name)
        with tf.name_scope(name):
            self.attention = tf.keras.layers.MultiHeadAttention(
                num_heads=config.NUM_ATTENTION_HEADS,
                key_dim=config.EMBED_DIM // config.NUM_ATTENTION_HEADS,
                value_dim=config.EMBED_DIM // config.NUM_ATTENTION_HEADS,
                name="mha_sublayer"
            )
            self.attention_layernorm = tf.keras.layers.LayerNormalization(
                epsilon=config.LAYER_NORM_EPS, name="mha_sublayer_normalization"
            )
            self.dense = tf.keras.layers.Dense(
                units=config.EMBED_DIM, name="dense_sublayer"
            )
            self.dense_layernorm = tf.keras.layers.LayerNormalization(
                epsilon=config.LAYER_NORM_EPS, name="dense_sublayer_normalization"
            )
            self.addition = tf.keras.layers.Add(name="residual_connection")
            self.dropout = tf.keras.layers.Dropout(rate=0.1, name="dropout")

    def call(self, inputs, *args, **kwargs):
        attention_output = self.attention(
            query=inputs["input_tensor"],
            value=inputs["input_tensor"],
            attention_mask=inputs["attention_mask"],
            return_attention_scores=False,
        )
        attention_output = self.dropout(attention_output)
        attention_output = self.addition([inputs["input_tensor"], attention_output])
        attention_output = self.attention_layernorm(attention_output)
        ffn_output = self.dense(attention_output)
        ffn_output = self.dropout(ffn_output)
        ffn_output = self.addition([attention_output, ffn_output])
        return {
            "input_tensor": self.dense_layernorm(ffn_output),
            "attention_mask": inputs["attention_mask"],
        }


class BertLinearTransform(tf.keras.layers.Layer):

    def __init__(self, name: str):
        """
        Implementation of position-wise feed-forward networks described in clause 3.3 of original Vaswani(2017) paper.
        In huggingface Transformer, this corresponds to combination of TFBertIntermediate and TFBertOutput layers.
        :param name: name of current linear transformation layer
        """
        super(BertLinearTransform, self).__init__(name=name)
        with tf.name_scope(name):
            self.intermediate_dense = tf.keras.layers.Dense(
                units=config.DENSE_DIM, name="intermediate", activation="gelu"
            )
            self.output_dense = tf.keras.layers.Dense(
                units=config.EMBED_DIM, name="output"
            )
            self.dropout = tf.keras.layers.Dropout(rate=0.1, name="dropout")
            self.layernorm = tf.keras.layers.LayerNormalization(
                epsilon=config.LAYER_NORM_EPS, name="layer_normalization"
            )
            self.addition = tf.keras.layers.Add(name="residual_connection")

    def call(self, inputs, *args, **kwargs):
        ffn_output = self.intermediate_dense(inputs["input_tensor"])
        ffn_output = self.output_dense(ffn_output)
        ffn_output = self.dropout(ffn_output)
        ffn_output = self.addition([inputs["input_tensor"], ffn_output])
        return {
            "input_tensor": self.layernorm(ffn_output),
            "attention_mask": inputs["attention_mask"],
        }


class BertEncoder(tf.keras.layers.Layer):

    def __init__(self, name: str):
        """
        3-dimensional attention mask where each dimension represents (batch_size, num_heads, max_length) respectively

        Layers in a Sequential model should only have a single input tensor
        :param name:
        """
        super(BertEncoder, self).__init__(name=name)
        self._pad_token_id = config.SPECIAL_TOKENS["pad_token"]["id"]
        with tf.name_scope(name):
            self.embeddings = BertEmbeddings("embeddings")
            self.layer_group = []
            for index in range(config.NUM_LAYERS * 2):
                layer_index, is_attention_layer = divmod(index, 2)
                if is_attention_layer == 0:
                    self.layer_group.append(BertAttention(f"attention_{layer_index}"))
                else:
                    self.layer_group.append(BertLinearTransform(f"linear_transform_{layer_index}"))

    def call(self, inputs, *args, **kwargs) -> tf.Tensor:
        """
        Encode masked token sequences into embedding tensor
        :param inputs: special token added(including pad token) square tensor of masked token sequences
        :type inputs: tf.Tensor
        :return: batched sequence of BERT encoded embeddings
        """
        inputs = {
            "input_tensor": self.embeddings(inputs),
            "attention_mask": tf.expand_dims(
                input=inputs != self._pad_token_id,
                axis=1,  # dimension corresponds to num_heads(axes=1) will later be broadcasted within MHA layers
                name="generate_attention_mask",
            )
        }
        for layer in self.layer_group:
            inputs = layer(inputs)
        return inputs["input_tensor"]


class BertMLMPreTrainer(tf.keras.Model):

    def __init__(self, name: str):
        super(BertMLMPreTrainer, self).__init__(name)
        with tf.name_scope(name):
            self.encoder = BertEncoder(name="bert_encoder")
            self.mlm_head = tf.keras.layers.Dense(
                units=config.VOCAB_SIZE, name="mlm_head", activation="softmax"
            )
            self.loss_function = tf.keras.losses.SparseCategoricalCrossentropy(
                from_logits=False,  # because activation="softmax"
                # ignore_class=-100  # rather than sample_weight, labeling non-mask token in y_true is also possible
                reduction=tf.keras.losses.Reduction.NONE,
                name="sparse_categorical_cross_entropy"
            )
            self.loss_tracker = tf.keras.metrics.Mean(name="mean_loss")

    def call(self, inputs, training=None, mask=None):
        x = self.encoder(inputs)
        return self.mlm_head(x)

    def train_step(self, data):
        masked_inputs, original_inputs, sample_weight = data
        with tf.GradientTape() as tape:
            prediction = self(masked_inputs, training=True)
            loss = self.loss_function(
                y_true=original_inputs,
                y_pred=prediction,
                sample_weight=sample_weight,
            )
        trainable_variables = self.trainable_variables
        gradients = tape.gradient(target=loss, sources=trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, trainable_variables))
        self.loss_tracker.update_state(loss, sample_weight)
        return {"loss": self.loss_tracker.result()}
