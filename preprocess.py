import tensorflow as tf
import tensorflow_datasets as tfds
import sentencepiece as spm
import pathlib
import config
from typing import List


def preprocess_review(review: tf.Tensor) -> str:
    """
    Replace line break HTML tag to single space and uppercase letter to lowercase
    :param review: tf.Tensor of raw review text
    :return: preprocessed review
    """
    x = tf.strings.regex_replace(review, "<[^>]+>", " ", replace_global=True)
    x = tf.strings.lower(x)
    return tf.strings.strip(x).numpy().decode("utf-8")


def fetch_imdb_corpus() -> List[str]:
    """
    Fetch IMDb review data from tensorflow datasets and extract preprocessed reviews as list of string
    :return: list of preprocessed reviews
    """
    imdb_dataset = tfds.load("imdb_reviews", split="train", shuffle_files=True)
    corpus = []
    for review in imdb_dataset:
        corpus.append(preprocess_review(review["text"]))
    return corpus


def save_corpus_to_local(corpus: List[str]):
    """
    Save extracted corpus to configured path
    :param corpus: output of fetch_imdb_corpus method
    :return: None
    """
    corpus_path = config.SPM_TRAINER_CONFIG["input"]
    corpus_path = pathlib.Path(corpus_path)
    corpus_path.parent.mkdir(exist_ok=True, parents=True)
    with open(corpus_path, "w") as file:
        file.writelines("\n".join(corpus))


def train_spm_tokenizer():
    """
    Train Sentencepiece tokenizer and save result to configured local path
    :return: None
    """
    spm_trainer_config = config.SPM_TRAINER_CONFIG  # configured tokenizer settings
    spm.SentencePieceTrainer.Train(**spm_trainer_config)
