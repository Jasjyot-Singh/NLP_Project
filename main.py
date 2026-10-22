import sys
from encoder import OneHotEncoder

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")


def display_one_hot(corpus):
    encoder = OneHotEncoder(lowercase=True, include_special_tokens=False)
    encoder.fit(corpus)
    matrix, tokens, indices = encoder.encode_sequence(corpus)

    print(f'Corpus: "{corpus}"')
    print(f"Step 1: Lowercase & Tokenize ➔ [{', '.join(tokens)}]")
    print(f"Step 2: Build Unique Sorted Vocab ➔ V = {{{', '.join(encoder.vocab)}}} (|V| = {len(encoder.vocab)})")
    print("Step 3 & 4: Word Vectors:")

    max_len = max(len(t) for t in tokens) if tokens else 0
    for token, row, idx in zip(tokens, matrix, indices):
        vec_str = str(row.tolist())
        print(f"{token:<{max_len}} ➔ {vec_str}   (Index {idx})")


def main():
    if len(sys.argv) > 1:
        corpus = " ".join(sys.argv[1:])
    else:
        corpus = "The cat sat on the mat"

    display_one_hot(corpus)


if __name__ == "__main__":
    main()
