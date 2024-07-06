# Data Directory

This directory contains the prepared datasets for the sentiment analysis tasks.

## Files

- **prepare_data.py**: Script to download and preprocess the IMDB dataset.
- **train_texts.txt**: Preprocessed training texts.
- **train_labels.txt**: Preprocessed training labels.
- **test_texts.txt**: Preprocessed test texts.
- **test_labels.txt**: Preprocessed test labels.

## Usage

1. Run `prepare_data.py` to download and preprocess the IMDB dataset:
    ```bash
    python prepare_data.py
    ```

2. The preprocessed data will be saved in this directory as `train_texts.txt`, `train_labels.txt`, `test_texts.txt`, and `test_labels.txt`.
