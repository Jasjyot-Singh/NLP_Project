import json
import re
import numpy as np


class OneHotEncoder:
    def __init__(self, lowercase=True, include_special_tokens=False):
        self.lowercase = lowercase
        self.include_special_tokens = include_special_tokens
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.token_to_idx = {}
        self.idx_to_token = {}
        self.vocab = []
        self.vocab_size = 0

    def tokenize(self, text):
        if self.lowercase:
            text = text.lower()
        return re.findall(r"\b\w+\b", text)

    def fit(self, texts):
        if isinstance(texts, str):
            texts = [texts]

        unique_tokens = set()
        for text in texts:
            for token in self.tokenize(text):
                unique_tokens.add(token)

        self.vocab = sorted(list(unique_tokens))
        self.token_to_idx = {}
        self.idx_to_token = {}
        idx = 0

        if self.include_special_tokens:
            self.token_to_idx[self.pad_token] = idx
            self.idx_to_token[idx] = self.pad_token
            idx += 1
            self.token_to_idx[self.unk_token] = idx
            self.idx_to_token[idx] = self.unk_token
            idx += 1

        for token in self.vocab:
            self.token_to_idx[token] = idx
            self.idx_to_token[idx] = token
            idx += 1

        self.vocab_size = len(self.token_to_idx)
        return self

    def fit_transform(self, texts):
        self.fit(texts)
        if isinstance(texts, str):
            return self.encode_sequence(texts)
        return [self.encode_sequence(text) for text in texts]

    def encode_word(self, word):
        if self.lowercase:
            word = word.lower()
        vec = np.zeros(self.vocab_size, dtype=int)
        idx = self.token_to_idx.get(word)
        if idx is None and self.include_special_tokens:
            idx = self.token_to_idx.get(self.unk_token)
        if idx is not None:
            vec[idx] = 1
        return vec, idx

    def decode_word(self, vector):
        idx = int(np.argmax(vector))
        return self.idx_to_token.get(idx)

    def encode_sequence(self, text):
        tokens = self.tokenize(text)
        matrix = np.zeros((len(tokens), self.vocab_size), dtype=int)
        indices = []
        for i, token in enumerate(tokens):
            vec, idx = self.encode_word(token)
            matrix[i] = vec
            indices.append(idx)
        return matrix, tokens, indices

    def decode_sequence(self, matrix):
        words = []
        for row in matrix:
            idx = int(np.argmax(row))
            token = self.idx_to_token.get(idx)
            if token is not None and token != self.pad_token:
                words.append(token)
        return " ".join(words)

    def encode_document(self, text):
        tokens = set(self.tokenize(text))
        vec = np.zeros(self.vocab_size, dtype=int)
        for token in tokens:
            idx = self.token_to_idx.get(token)
            if idx is None and self.include_special_tokens:
                idx = self.token_to_idx.get(self.unk_token)
            if idx is not None:
                vec[idx] = 1
        return vec

    def save_vocab(self, filepath):
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump({
                "token_to_idx": self.token_to_idx,
                "vocab": self.vocab,
                "vocab_size": self.vocab_size,
                "lowercase": self.lowercase,
                "include_special_tokens": self.include_special_tokens
            }, f, indent=2)

    def load_vocab(self, filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
            self.token_to_idx = data["token_to_idx"]
            self.idx_to_token = {int(v): k for k, v in self.token_to_idx.items()}
            self.vocab = data.get("vocab", [])
            self.vocab_size = data["vocab_size"]
            self.lowercase = data["lowercase"]
            self.include_special_tokens = data.get("include_special_tokens", False)
