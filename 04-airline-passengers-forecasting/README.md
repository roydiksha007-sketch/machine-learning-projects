# Airline Passengers: LSTM Time Series Model

Class project: uses an LSTM neural network to predict the next month's airline passenger count from the previous 12 months.

## Dataset
- `airline_passengers.csv`: monthly airline passenger totals, 144 months (January 1949 to December 1960)
- Source: *(add the source, e.g. the classic Box & Jenkins airline passengers dataset)*

## Method
1. Scaled the passenger counts to the range 0 to 1 with `MinMaxScaler`
2. Built sliding windows: each input is 12 consecutive months, and the target is the following month (132 windows)
3. Split the windows 80% training and 20% testing
4. Built a Keras model with four stacked LSTM layers (50 units each) and a Dense output layer
5. Trained for 100 epochs with the Adam optimizer and mean squared error loss
6. Converted predictions back to passenger counts and plotted predicted against actual values

## Results
The notebook plots the model's predictions against the actual passenger counts. This was a learning exercise, so no error metrics (MAE or RMSE) are reported.

## Limitations and next steps
- The train/test split is random. For time series, a chronological split (train on earlier months, test on later months) gives a more realistic forecast
- The scaler is fitted on the whole series; it should be fitted on training data only
- Add MAE and RMSE on the test set, and compare against a naive baseline ("same as last month")
- The model is large for this amount of data; a single LSTM layer may do as well

## How to run
The notebook loads the data from Google Drive. To run it elsewhere, change the file path in the first data cell to `airline_passengers.csv`.
```bash
pip install pandas scikit-learn tensorflow matplotlib
```

## Tools
Python, pandas, scikit-learn, Keras/TensorFlow, matplotlib
