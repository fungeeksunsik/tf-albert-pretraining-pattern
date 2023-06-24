import config
import tensorflow as tf
import tensorflow_datasets as tfds

from bert import BertEncoder
from preprocess import preprocess_review
from train import load_tokenizer, postprocess_token_sequences

_IMDB_DATA_OUTPUT_SIGNATURE = (
    tf.TensorSpec(
        shape=(None, config.PADDED_SEQUENCE_LENGTH), dtype=tf.dtypes.int32, name="token_sequences"
    ),
    tf.TensorSpec(
        shape=(None,), dtype=tf.dtypes.int32, name="labels"
    ),
)


def imdb_sentiment_train_data_generator():
    tokenizer = load_tokenizer()
    dataset = tfds.load("imdb_reviews", split="train", shuffle_files=True)
    sequences = []
    labels = []
    for review in dataset:
        preprocessed_review = preprocess_review(review["text"])
        token_sequence = tokenizer.encode(preprocessed_review)
        labels.append(review["label"].numpy())
        sequences.append(token_sequence)
        if len(sequences) >= config.BATCH_SIZE:
            sequences = postprocess_token_sequences(sequences)
            yield tf.constant(sequences), tf.constant(labels)
            sequences = []
            labels = []
    if sequences:
        sequences = postprocess_token_sequences(sequences)
        yield tf.constant(sequences), tf.constant(labels)


def create_train_imdb_sentiment_data():
    return tf.data.Dataset.from_generator(
        generator=imdb_sentiment_train_data_generator,
        output_signature=_IMDB_DATA_OUTPUT_SIGNATURE,
        name="generate_train_imdb_sentiment_data"
    )


def imdb_sentiment_validation_data_generator():
    tokenizer = load_tokenizer()
    dataset = tfds.load("imdb_reviews", split="test", shuffle_files=True)
    sequences = []
    labels = []
    for review in dataset:
        preprocessed_review = preprocess_review(review["text"])
        token_sequence = tokenizer.encode(preprocessed_review)
        labels.append(review["label"].numpy())
        sequences.append(token_sequence)
        if len(sequences) >= config.BATCH_SIZE:
            sequences = postprocess_token_sequences(sequences)
            yield tf.constant(sequences), tf.constant(labels)
            sequences = []
            labels = []
    if sequences:
        sequences = postprocess_token_sequences(sequences)
        yield tf.constant(sequences), tf.constant(labels)


def create_validation_imdb_sentiment_data():
    return tf.data.Dataset.from_generator(
        generator=imdb_sentiment_validation_data_generator,
        output_signature=_IMDB_DATA_OUTPUT_SIGNATURE,
        name="generate_validation_imdb_sentiment_data"
    )


class IMDbReviewSentimentAnalyzer(tf.keras.Model):

    def __init__(self, name: str):
        super(IMDbReviewSentimentAnalyzer, self).__init__(name)
        self.pad_token_id = config.SPECIAL_TOKENS["pad_token"]["id"]
        self.encoder = _load_bert_encoder()
        self.mean_pool = tf.keras.layers.GlobalAveragePooling1D()
        self.dropout = tf.keras.layers.Dropout(rate=0.1)
        self.hidden_layer = tf.keras.layers.Dense(
            units=config.HIDDEN_LAYER_UNITS, activation="relu"
        )
        self.probability = tf.keras.layers.Dense(units=1, activation="sigmoid")
        self.loss_function = tf.keras.losses.BinaryCrossentropy(
            reduction=tf.keras.losses.Reduction.NONE,
            name="binary_cross_entropy"
        )
        self.loss_tracker = tf.keras.metrics.Mean(name="mean_loss")

    def call(self, inputs, training=None, mask=None):
        pad_mask = inputs != self.pad_token_id
        x = self.encoder(inputs)
        x = self.mean_pool(inputs=x, mask=pad_mask)
        x = self.dropout(x)
        x = self.hidden_layer(x)
        x = self.dropout(x)
        return self.probability(x)

    def train_step(self, data):
        token_sequences, labels = data
        with tf.GradientTape() as tape:
            prediction = self(token_sequences, training=True)
            loss = self.loss_function(y_true=labels, y_pred=prediction)
        trainable_variables = self.trainable_variables
        gradients = tape.gradient(target=loss, sources=trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, trainable_variables))
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}


def _load_bert_encoder() -> BertEncoder:
    archive_path = f"{config.LOCAL_DIR}/{config.SAVED_MODEL_ARCHIVE_NAME}"
    return tf.keras.models.load_model(archive_path).get_layer("bert_encoder")
