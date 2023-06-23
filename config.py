# local directory to save files generated from project
LOCAL_DIR = "/tmp/tbpp"

# BERT hyperparameters
PADDED_SEQUENCE_LENGTH = 256
BATCH_SIZE = 32
VOCAB_SIZE = 25000
EMBED_DIM = 128
DENSE_DIM = 512
NUM_ATTENTION_HEADS = 8
NUM_LAYERS = 2
LAYER_NORM_EPS = 1e-6
NUM_EPOCHS = 3
SAVED_MODEL_ARCHIVE_NAME = "bert_mlm_imdb"

# sentencepiece tokenizer parameters
SPECIAL_TOKENS = {
    "pad_token": {"id": 0, "piece": "[PAD]"},
    "unk_token": {"id": 1, "piece": "[UNK]"},
    "bos_token": {"id": 2, "piece": "[CLS]"},
    "eos_token": {"id": 3, "piece": "[SEP]"},
    "mask_token": {"id": 4, "piece": "[MASK]"},
}


def parse_special_tokens() -> str:
    user_defined_symbols = [
        token["piece"]
        for token in sorted(SPECIAL_TOKENS.values(), key=lambda x: x["id"])[4:]
    ]
    return ",".join(user_defined_symbols)


SPM_TRAINER_CONFIG = {
    "input": f"{LOCAL_DIR}/corpus.txt",
    "model_prefix": f"{LOCAL_DIR}/tokenizer",
    "vocab_size": VOCAB_SIZE,
    "model_type": "bpe",
    "pad_id": SPECIAL_TOKENS["pad_token"]["id"],
    "unk_id": SPECIAL_TOKENS["unk_token"]["id"],
    "bos_id": SPECIAL_TOKENS["bos_token"]["id"],
    "eos_id": SPECIAL_TOKENS["eos_token"]["id"],
    "pad_piece": SPECIAL_TOKENS["pad_token"]["piece"],
    "unk_piece": SPECIAL_TOKENS["unk_token"]["piece"],
    "bos_piece": SPECIAL_TOKENS["bos_token"]["piece"],
    "eos_piece": SPECIAL_TOKENS["eos_token"]["piece"],
    "user_defined_symbols": parse_special_tokens(),
    "split_by_number": True,
    "add_dummy_prefix": True,
    "train_extremely_large_corpus": False,
    "minloglevel": 3
}

# sentiment classifier parameters
HIDDEN_LAYER_UNITS = 64
