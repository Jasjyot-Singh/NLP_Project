import json
import re
import numpy as np
import pandas as pd


class OneHotEncoder:
    def __init__(self, lowercase=True, handle_unknown=True):
        self.lowercase = lowercase
        self.handle_unknown = handle_unknown
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.token_to_idx = {}
        self.idx_to_token = {}
        self.vocab_size = 0

    def tokenize(self, text):
        if self.lowercase:
            text = text.lower()
        return re.findall(r"\b\w+\b", text)

    def fit(self, texts):
        unique_tokens = []
        for text in texts:
            for token in self.tokenize(text):
                if token not in unique_tokens:
                    unique_tokens.append(token)

        self.token_to_idx = {}
        self.idx_to_token = {}

        idx = 0
        self.token_to_idx[self.pad_token] = idx
        self.idx_to_token[idx] = self.pad_token
        idx += 1

        if self.handle_unknown:
            self.token_to_idx[self.unk_token] = idx
            self.idx_to_token[idx] = self.unk_token
            idx += 1

        for token in sorted(unique_tokens):
            if token not in self.token_to_idx:
                self.token_to_idx[token] = idx
                self.idx_to_token[idx] = token
                idx += 1

        self.vocab_size = len(self.token_to_idx)
        return self

    def fit_transform(self, texts):
        self.fit(texts)
        return [self.encode_sequence(text) for text in texts]

    def encode_word(self, word):
        if self.lowercase:
            word = word.lower()
        vec = np.zeros(self.vocab_size, dtype=int)
        idx = self.token_to_idx.get(word, self.token_to_idx.get(self.unk_token))
        if idx is not None:
            vec[idx] = 1
        return vec

    def decode_word(self, vector):
        idx = int(np.argmax(vector))
        return self.idx_to_token.get(idx, self.unk_token)

    def encode_sequence(self, text, max_len=None):
        tokens = self.tokenize(text)
        if max_len is not None:
            if len(tokens) > max_len:
                tokens = tokens[:max_len]
            elif len(tokens) < max_len:
                tokens = tokens + [self.pad_token] * (max_len - len(tokens))

        matrix = np.zeros((len(tokens), self.vocab_size), dtype=int)
        for i, token in enumerate(tokens):
            idx = self.token_to_idx.get(token, self.token_to_idx.get(self.unk_token))
            if idx is not None:
                matrix[i, idx] = 1
        return matrix

    def decode_sequence(self, matrix):
        words = []
        for row in matrix:
            idx = int(np.argmax(row))
            token = self.idx_to_token.get(idx, self.unk_token)
            if token != self.pad_token:
                words.append(token)
        return " ".join(words)

    def encode_document(self, text):
        tokens = set(self.tokenize(text))
        vec = np.zeros(self.vocab_size, dtype=int)
        for token in tokens:
            idx = self.token_to_idx.get(token, self.token_to_idx.get(self.unk_token))
            if idx is not None:
                vec[idx] = 1
        return vec

    def to_dataframe(self, matrix, text):
        tokens = self.tokenize(text)
        columns = [self.idx_to_token[i] for i in range(self.vocab_size)]
        return pd.DataFrame(matrix, index=tokens, columns=columns)

    def save_vocab(self, filepath):
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump({
                "token_to_idx": self.token_to_idx,
                "vocab_size": self.vocab_size,
                "lowercase": self.lowercase,
                "handle_unknown": self.handle_unknown
            }, f, indent=2)

    def load_vocab(self, filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
            self.token_to_idx = data["token_to_idx"]
            self.idx_to_token = {int(v): k for k, v in self.token_to_idx.items()}
            self.vocab_size = data["vocab_size"]
            self.lowercase = data["lowercase"]
            self.handle_unknown = data["handle_unknown"]
