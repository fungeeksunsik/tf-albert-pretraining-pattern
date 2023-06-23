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


if __name__ == "__main__":
    app()
