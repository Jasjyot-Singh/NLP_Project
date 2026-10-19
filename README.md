# NLP_Project: One-Hot Vector Encoding

A complete Natural Language Processing (NLP) implementation of One-Hot Vector Encoding from scratch in Python.

## Features
- **Tokenization & Vocabulary Mapping**: Automatically builds vocabulary with `<PAD>` and `<UNK>` (Out-of-Vocabulary) support.
- **Word-Level One-Hot Encoding**: Converts individual tokens into 1D one-hot vectors.
- **Sequence-Level Encoding**: Transforms sentences into 2D matrices `(sequence_length, vocab_size)` with tabular Pandas DataFrame visualization.
- **Document-Level Encoding**: Multi-hot (binary bag-of-words) representations.
- **Decoding / Reconstruction**: Uses `argmax` to invert one-hot representations back to text.
- **Vocabulary Persistence**: Save and reload vocabulary mappings via JSON.

## Project Structure
- `encoder.py`: Implementation of the `OneHotEncoder` class.
- `main.py`: Demonstration script, CLI tool, and interactive interface.
- `requirements.txt`: Python package dependencies (`numpy`, `pandas`).

## Installation
```bash
pip install -r requirements.txt
```

## Usage

### Run Demonstration and Interactive Mode
```bash
python main.py
```

### Pass Custom Text via Command Line
```bash
python main.py "machine learning and natural language processing"
```
