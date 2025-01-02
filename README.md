# Thai Baht to USD Exchange Rate Prediction

This repository contains a Python-based implementation of a linear regression model to predict the exchange rate of the Thai Baht (THB) to the US Dollar (USD). The project leverages a Kaggle dataset and basic time series techniques to train and evaluate the model.

## Dataset

The dataset is sourced from Kaggle:
- **Dataset Name**: [Foreign Exchange Rates Per Dollar 2000-2019](https://www.kaggle.com/datasets/brunotly/foreign-exchange-rates-per-dollar-20002019)
- **Columns Used**:
  - `Time Serie`: Date of the exchange rate.
  - `THAILAND - BAHT/US$`: Thai Baht to USD exchange rate.

## Methodology

1. **Data Preprocessing**:
   - Extracted relevant columns (`Time Serie` and `THAILAND - BAHT/US$`).
   - Converted the exchange rate data to numerical values.
   - Created lagged features (`Lag_1` and `Lag_2`) for time series modeling.
   - Split the dataset chronologically into training (80%) and testing (20%) sets.

2. **Model Training**:
   - A **Linear Regression** model was trained using the lagged features as predictors.

3. **Evaluation**:
   - Metrics used for evaluation:
     - **Mean Squared Error (MSE)**: 0.0086
     - **Mean Absolute Error (MAE)**: 0.0679
     - **R-squared (R²)**: 0.9971
   - The model demonstrated excellent performance with high accuracy.

4. **Future Prediction**:
   - The model predicts the next exchange rate based on the most recent lagged values.

## Results

The model's evaluation metrics indicate a strong performance:
- **MSE**: 0.0086
- **MAE**: 0.0679
- **R-squared**: 0.9971

This suggests that the model captures the underlying patterns of the exchange rate data very effectively.

## Installation

To run the project locally, follow these steps:

1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/thb-usd-exchange-rate-prediction.git
   cd thb-usd-exchange-rate-prediction
   ```

2. Install the required Python libraries:
   ```bash
   pip install -r requirements.txt
   ```

3. Download the dataset from Kaggle:
   - Use the `kaggle` CLI or the provided `kagglehub` code snippet to download the dataset.

4. Run the script:
   ```bash
   python main.py
   ```

## Usage

### Code Walkthrough

1. **Preprocessing**:
   - Extract relevant columns.
   - Create lagged features.
   - Handle missing values and split data chronologically.

2. **Training**:
   - Train the linear regression model using lagged features.

3. **Evaluation**:
   - Compute performance metrics such as MSE, MAE, and R-squared.

4. **Prediction**:
   - Forecast future exchange rates using the trained model.

### Example Output

```text
Mean Squared Error: 0.0086492570072786
Mean Absolute Error: 0.06790474134533907
R-squared: 0.9971741217154321
Next Predicted THB/USD Exchange Rate: <value>
```

## File Structure

```
.
├── main.py              # Main script for preprocessing, training, and evaluating the model
├── requirements.txt     # List of required Python libraries
├── README.md            # Project documentation
└── dataset/             # Directory to store the downloaded dataset
```

## Future Improvements

- Add more external features (e.g., macroeconomic indicators) to improve robustness.
- Experiment with advanced models like ARIMA, XGBoost, or LSTMs.
- Implement walk-forward validation for better time series evaluation.

## License

This project is licensed under the Apache License. See the `LICENSE` file for details.

## Acknowledgements

- Dataset by [Brunotly on Kaggle](https://www.kaggle.com/datasets/brunotly/foreign-exchange-rates-per-dollar-20002019)
- Python libraries: Scikit-learn, Pandas, Matplotlib

Feel free to fork, contribute, or raise issues. Happy coding!
