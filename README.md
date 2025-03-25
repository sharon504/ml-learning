# LSTM Model for Time Series Prediction

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
