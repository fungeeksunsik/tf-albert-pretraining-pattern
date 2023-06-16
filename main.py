import typer
import preprocess

app = typer.Typer()


@app.command("preprocess")
def run_preprocess():
    corpus = preprocess.fetch_imdb_corpus()
    preprocess.save_corpus_to_local(corpus)
    preprocess.train_spm_tokenizer()


if __name__ == '__main__':
    app()
