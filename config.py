# local directory to save files generated from project
LOCAL_DIR = "/tmp/tbpp"

# sentencepiece tokenizer parameters
MASK_TOKEN = "[MASK]"
SPM_TRAINER_CONFIG = {
    "input": f"{LOCAL_DIR}/corpus.txt",
    "model_prefix": f"{LOCAL_DIR}/tokenizer",
    "vocab_size": 25000,
    "model_type": "bpe",
    "pad_id": 0,
    "unk_id": 1,
    "bos_id": 2,
    "eos_id": 3,
    "pad_piece": "[PAD]",
    "unk_piece": "[UNK]",
    "bos_piece": "[CLS]",
    "eos_piece": "[SEP]",
    "user_defined_symbols": MASK_TOKEN,
    "split_by_number": True,
    "add_dummy_prefix": True,
    "train_extremely_large_corpus": False,
    "minloglevel": 3
}
