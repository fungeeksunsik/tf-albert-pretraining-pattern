import logging
import sys
import typer
import bert
import config
import evaluate
import preprocess
import train
import tensorflow as tf

app = typer.Typer()
formatter = logging.Formatter(
    fmt="%(asctime)s (%(funcName)s) : %(msg)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setFormatter(formatter)
logger = logging.getLogger(__name__)
logger.addHandler(stream_handler)
logger.setLevel(logging.INFO)


@app.command("preprocess")
def run_preprocess():
    logger.info("Fetch IMDb corpus from Tensorflow Dataset")
    corpus = preprocess.fetch_imdb_corpus()

    logger.info("Save fetched corpus to local directory")
    preprocess.save_corpus_to_local(corpus)

    logger.info("Train sentencepiece tokenizer with saved corpus")
    preprocess.train_spm_tokenizer()


@app.command("train")
def run_train():
    logger.info("Create masked dataset for MLM task")
    dataset = train.create_masked_dataset()

    logger.info("Estimate BERT model parameters with masked dataset")
    bert_mlm_model = bert.BertMLMPreTrainer("mlm_model")
    bert_mlm_model.compile(optimizer=tf.keras.optimizers.legacy.Adam())
    bert_mlm_model.fit(dataset, epochs=config.NUM_EPOCHS)

    logger.info("Save pretrained BERT model")
    archive_path = f"{config.LOCAL_DIR}/{config.SAVED_MODEL_ARCHIVE_NAME}"
    bert_mlm_model.save(archive_path, save_format="tf")


@app.command("evaluate")
def run_evaluate():
    logger.info("Compose classification dataset for BERT fine-tuning")
    train_dataset = evaluate.create_train_imdb_sentiment_data()
    test_dataset = evaluate.create_validation_imdb_sentiment_data()

    logger.info("Fine-tune sentiment classification model based on the pretrained model")
    imdb_predictor = evaluate.IMDbReviewSentimentAnalyzer("imdb_predictor")
    imdb_predictor.compile(optimizer=tf.keras.optimizers.legacy.Adam(), metrics=["accuracy"])
    imdb_predictor.fit(x=train_dataset, epochs=config.NUM_EPOCHS, validation_data=test_dataset)

    logger.info("Save fine-tuned sentiment classification model")
    archive_path = f"{config.LOCAL_DIR}/imdb_predictor"
    imdb_predictor.save(archive_path, save_format="tf")


if __name__ == "__main__":
    app()
