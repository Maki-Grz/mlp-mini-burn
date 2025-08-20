# MLP Mini Burn

**MLP Mini Burn** is a Rust-based learning project demonstrating a Multi-Layer Perceptron (MLP) implementation using
the [Burn](https://github.com/burn-rs/burn) deep learning framework. This repository is designed as a hands-on base for
exploring neural networks, data preprocessing, training, and model evaluation.

## Features

- Multi-Layer Perceptron with configurable hidden layers
- Cross-entropy loss and SGD optimizer with gradient clipping
- Data preprocessing using [Polars](https://pola-rs.github.io/polars/) for CSV files
- Automatic handling of categorical features (genres) and numeric normalization
- Model saving and loading via Burn’s `CompactRecorder`
- Early stopping to prevent overfitting
- Flexible backend using `burn-autodiff` and `burn-ndarray`

## Usage

1. **Prepare datasets**:
    - `anime.csv` - anime metadata (id, genre, episodes, etc.)
    - `rating.csv` - user ratings linked by `anime_id`

2. **Run the training**:
   ```bash
   cargo run
