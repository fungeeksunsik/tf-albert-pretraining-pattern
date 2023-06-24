import typer
import bert
import config
import evaluate
import preprocess
import train
import tensorflow as tf

app = typer.Typer()


@app.command("preprocess")
def run_preprocess():
    corpus = preprocess.fetch_imdb_corpus()
    preprocess.save_corpus_to_local(corpus)
    preprocess.train_spm_tokenizer()


@app.command("train")
def run_train():
    dataset = train.create_masked_dataset()
    bert_mlm_model = bert.BertMLMPreTrainer("mlm_model")
    bert_mlm_model.compile(optimizer=tf.keras.optimizers.legacy.Adam())
    bert_mlm_model.fit(dataset, epochs=config.NUM_EPOCHS)
    archive_path = f"{config.LOCAL_DIR}/{config.SAVED_MODEL_ARCHIVE_NAME}"
    bert_mlm_model.save(archive_path, save_format="tf")


@app.command("evaluate")
def run_evaluate():
    train_dataset = evaluate.create_train_imdb_sentiment_data()
    test_dataset = evaluate.create_validation_imdb_sentiment_data()
    imdb_predictor = evaluate.IMDbReviewSentimentAnalyzer("imdb_predictor")
    imdb_predictor.compile(optimizer=tf.keras.optimizers.legacy.Adam(), metrics=["accuracy"])
    imdb_predictor.fit(x=train_dataset, epochs=config.NUM_EPOCHS, validation_data=test_dataset)
    archive_path = f"{config.LOCAL_DIR}/imdb_predictor"
    imdb_predictor.save(archive_path, save_format="tf")


if __name__ == "__main__":
    app()
