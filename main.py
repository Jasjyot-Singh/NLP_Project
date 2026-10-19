import sys
import pandas as pd
from encoder import OneHotEncoder

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 1000)


def run_demonstration():
    corpus = [
        "Natural language processing with Python",
        "One hot vector encoding for text data",
        "Machine learning models require numerical representation"
    ]

    print("=" * 60)
    print("ONE-HOT VECTOR ENCODING PROJECT")
    print("=" * 60)

    encoder = OneHotEncoder(lowercase=True, handle_unknown=True)
    encoder.fit(corpus)

    print("\n[1] TRAINING CORPUS:")
    for i, doc in enumerate(corpus, 1):
        print(f"  Sentence {i}: {doc}")

    print(f"\n[2] VOCABULARY (Size = {encoder.vocab_size}):")
    for token, idx in sorted(encoder.token_to_idx.items(), key=lambda item: item[1]):
        print(f"  {idx:2d} -> {token}")

    sample_word = "encoding"
    word_vector = encoder.encode_word(sample_word)
    decoded_word = encoder.decode_word(word_vector)

    print("\n[3] WORD LEVEL ONE-HOT ENCODING:")
    print(f"  Target Word: '{sample_word}'")
    print(f"  One-Hot Vector shape: {word_vector.shape}")
    print(f"  One-Hot Vector: {word_vector.tolist()}")
    print(f"  Decoded back: '{decoded_word}'")

    sample_sentence = "one hot vector encoding"
    seq_matrix = encoder.encode_sequence(sample_sentence)
    decoded_sentence = encoder.decode_sequence(seq_matrix)

    print("\n[4] SEQUENCE LEVEL ONE-HOT MATRIX:")
    print(f"  Sentence: '{sample_sentence}'")
    print(f"  Matrix shape: {seq_matrix.shape} (tokens x vocab_size)")
    print(f"  Decoded back: '{decoded_sentence}'")
    print("\n  Tabular Representation:")
    df_seq = encoder.to_dataframe(seq_matrix, sample_sentence)
    print(df_seq)

    oov_sentence = "deep learning generates representation"
    oov_matrix = encoder.encode_sequence(oov_sentence)
    decoded_oov = encoder.decode_sequence(oov_matrix)

    print("\n[5] OUT-OF-VOCABULARY (OOV) ENCODING:")
    print(f"  Sentence with unseen tokens: '{oov_sentence}'")
    print(f"  Decoded with <UNK>: '{decoded_oov}'")
    print("\n  Tabular Representation:")
    df_oov = encoder.to_dataframe(oov_matrix, oov_sentence)
    print(df_oov)

    doc_vector = encoder.encode_document(sample_sentence)
    print("\n[6] DOCUMENT MULTI-HOT (BAG-OF-WORDS) VECTOR:")
    print(f"  Text: '{sample_sentence}'")
    print(f"  Multi-Hot Vector: {doc_vector.tolist()}")

    encoder.save_vocab("vocab.json")
    print("\n[7] VOCABULARY PERSISTENCE:")
    print("  Saved vocabulary to 'vocab.json'.")

    return encoder


def process_user_text(encoder, text):
    print("\n" + "-" * 50)
    print(f"Input: '{text}'")
    seq_matrix = encoder.encode_sequence(text)
    print(f"Matrix Shape: {seq_matrix.shape}")
    print("\nMatrix View:")
    print(encoder.to_dataframe(seq_matrix, text))
    print(f"\nDecoded Text: '{encoder.decode_sequence(seq_matrix)}'")
    doc_vec = encoder.encode_document(text)
    print(f"Document Binary Vector: {doc_vec.tolist()}")
    print("-" * 50)


def main():
    encoder = run_demonstration()

    if len(sys.argv) > 1:
        custom_text = " ".join(sys.argv[1:])
        process_user_text(encoder, custom_text)
        return

    print("\n" + "=" * 60)
    print("INTERACTIVE MODE (Type a sentence or 'exit' to quit)")
    print("=" * 60)

    while True:
        try:
            user_input = input("\nEnter text: ").strip()
            if not user_input or user_input.lower() in ("exit", "quit"):
                break
            process_user_text(encoder, user_input)
        except (KeyboardInterrupt, EOFError):
            break


if __name__ == "__main__":
    main()
