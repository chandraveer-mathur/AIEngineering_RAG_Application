from transformers import AutoTokenizer

from config import EMBEDDING_MODEL

def load_tokenizer():
    return AutoTokenizer.from_pretrained(EMBEDDING_MODEL)