# local directory to save files generated from project
LOCAL_DIR = "/tmp/tbpp"

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
    "vocab_size": 25000,
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

# BERT hyperparameters
NON_MASK_ID = -100
PADDED_SEQUENCE_LENGTH = 350
