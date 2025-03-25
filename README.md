# Makemore Name Generator

## Overview

This script implements a character-level name generator inspired by the Makemore model. It trains a neural network on a dataset of names to generate new names based on learned patterns.

## Dependencies

Ensure you have the following Python libraries installed:

```bash
pip install torch matplotlib
```

## Steps in the Script

### 1. Data Preparation

- Downloads a list of names from GitHub.
- Tokenizes the dataset into character-based sequences.
- Converts characters into indices using `stoi` (String-to-Index) and `itos` (Index-to-String) mappings.
- Splits data into training (80%), validation (10%), and test (10%) sets.

### 2. Model Architecture

- Implements a custom neural network with:
  - Character embeddings.
  - Consecutive token flattening.
  - Fully connected layers with batch normalization.
  - Tanh activation for non-linearity.

### 3. Training Process

- Uses cross-entropy loss for optimization.
- Runs for `200000` epochs with batch size `32`.
- Adjusts learning rate dynamically (`0.1` initially, then `0.01`).
- Prints loss updates every `10000` iterations.
- Evaluates the final model on training and validation data.

### 4. Name Generation

- Uses the trained model to generate names based on learned character sequences.
- Ensures batch normalization layers are set to inference mode before generation.

## Usage

1. Run the script:
   ```bash
   python makemore_name_generator.py
   ```
2. The model will train and display loss progression.
3. Modify `block_size`, `embedding_size`, and `n_hidden` for different variations.

## Future Improvements

- Implement dropout for better regularization.
- Experiment with different activation functions.
- Extend the model to generate words in other languages.

# Attention is all you need paper implementaion (Shakespere text generation)

## Overview

This script implements a Transformer model for text generation based on the "Attention is All You Need" paper. It uses Shakespeare's text dataset for training and generates new text sequences based on learned patterns.

## Dependencies

Ensure you have the following Python libraries installed:

```bash
pip install torch matplotlib kaggle
```

## Steps in the Script

### 1. Data Preparation

- Downloads the Shakespeare dataset from Kaggle.
- Prepares character-based tokenization using `stoi` (String-to-Index) and `itos` (Index-to-String) mappings.
- Splits data into training (90%) and validation (10%) sets.
- Creates batches of sequences with `block_size = 128` for training.

### 2. Transformer Model Architecture

- Implements self-attention and masked multi-head attention.
- Includes feedforward layers with ReLU activation.
- Uses layer normalization to stabilize training.
- Embeds input tokens and positional encodings.
- Outputs logits for token prediction.

### 3. Training Process

- Uses AdamW optimizer with a learning rate of `1e-3`.
- Runs for `2500` iterations, evaluating loss every epoch.
- Uses cross-entropy loss for training.
- Saves the trained model (`model.py`) and model weights (`model_state_dict.py`).

### 4. Text Generation

- Generates new text sequences by sampling tokens iteratively.
- Uses a context window to make predictions.
- Prints a sample generated text of `2000` characters.

## Usage

1. Place your Kaggle API key in `~/.kaggle/kaggle.json`.
2. Run the script:
   ```bash
   python attention_is_all_you_need.py
   ```
3. The trained model weights will be saved in `model_state_dict.py`.
4. Modify `max_new_tokens` in `generate()` for different output lengths.

## Future Improvements

- Implement additional text preprocessing steps.
- Optimize hyperparameters using grid search.
- Experiment with different text datasets for broader generalization.

# LSTM Model for d3code

## Overview

This script implements a Long Short-Term Memory (LSTM) model for time series prediction using PyTorch. The data is preprocessed, normalized, and converted into sequences before training the LSTM model to predict future values based on historical patterns.

## Dependencies

Ensure you have the following Python libraries installed:

```bash
pip install pandas numpy torch scikit-learn joblib
```

## Steps in the Script

### 1. Data Preparation

- Loads data from `dataset-1.csv`.
- Checks for NaN or infinite values and fills them appropriately.
- Normalizes the data using `MinMaxScaler` and saves the scaler.
- Converts data into sequences of specified length.
- Splits the data into training and testing sets.

### 2. LSTM Model Definition

- Defines a PyTorch-based LSTM model with:
  - Two LSTM layers.
  - Fully connected layers with ReLU activation.
  - Xavier initialization for weights.
- Outputs a prediction for the given sequence input.

### 3. Model Training

- Defines training parameters, including:
  - Mean Squared Error (MSE) loss function.
  - Adam optimizer with learning rate `0.00001`.
  - Gradient clipping to prevent exploding gradients.
- Trains the model for `100000` epochs, printing the loss every 10 epochs.
- Saves the trained model to `lstm_model.pth`.

## Usage

1. Place `dataset-1.csv` in the same directory as the script.
2. Run the script:
   ```bash
   python lstm_model.py
   ```
3. The trained model weights will be saved in `lstm_model.pth`.

## Notes

- Ensure the dataset contains a column named `Month` to set it as the index.
- Modify the `seq_length` parameter as needed for different time series forecasting needs.

## Future Improvements

- Implement early stopping to prevent overfitting.
- Optimize hyperparameters using grid search.
- Use more advanced data augmentation techniques to enhance model performance.
