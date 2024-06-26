import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.ensemble import RandomForestRegressor
import pandas as pd

# Load AQI data from CSV file
df = pd.read_csv('aqi_data.csv')  # Update 'your_aqi_data.csv' with the path to your CSV file
aqi_values = df['AQI'].values  # Assuming the AQI values are stored in a column named 'AQI'

def prepare_data(data, n_steps):
    X, y = [], []
    for i in range(len(data)):
        end_ix = i + n_steps
        if end_ix > len(data)-1:
            break
        seq_x, seq_y = data[i:end_ix], data[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

n_steps = 10

X_lstm, y_lstm = prepare_data(aqi_values, n_steps)
X_lstm = X_lstm.reshape((X_lstm.shape[0], X_lstm.shape[1], 1))

X_train_lstm, X_test_lstm, y_train_lstm, y_test_lstm = train_test_split(X_lstm, y_lstm, test_size=0.2, random_state=42)

lstm_model = Sequential()
lstm_model.add(LSTM(50, activation='relu', input_shape=(n_steps, 1)))
lstm_model.add(Dense(1))
lstm_model.compile(optimizer='adam', loss='mse')
lstm_model.fit(X_train_lstm, y_train_lstm, epochs=100, validation_data=(X_test_lstm, y_test_lstm), verbose=0)

predicted_aqi_lstm = []
for i in range(10):
    x_input_lstm = aqi_values[-n_steps:]  
    x_input_lstm = np.array(x_input_lstm).reshape((1, n_steps, 1))  
    yhat_lstm = lstm_model.predict(x_input_lstm, verbose=0)
    predicted_aqi_lstm.append(yhat_lstm[0][0])
    aqi_values = np.append(aqi_values, yhat_lstm[0][0])

X_rf = np.arange(len(aqi_values[:300])).reshape(-1, 1)
y_rf = aqi_values[:300]

rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_rf, y_rf)

next_ten_days = np.arange(300, 300 + 24*10)  
predicted_aqi_rf = rf_model.predict(next_ten_days.reshape(-1, 1))

# Plot the actual and predicted AQI values
plt.figure(figsize=(10, 6))
plt.plot(timestamps[:300], aqi_values[:300], label='Actual AQI')
plt.plot(np.concatenate((timestamps[:290], np.arange(timestamps[299] + 1, timestamps[299] + 11))), np.concatenate((aqi_values[:290], predicted_aqi_lstm)), label='Predicted AQI (LSTM)', linestyle='--')
plt.plot(next_ten_days, predicted_aqi_rf, label='Predicted AQI (Random Forest)', linestyle='-.')
plt.xlabel('Timestamp')
plt.ylabel('AQI')
plt.title('AQI Prediction for Next Ten Days (Ensemble)')
plt.legend()
plt.grid(True)
plt.show()

# Plot predicted AQI versus day
plt.figure(figsize=(10, 6))
plt.plot(np.arange(len(predicted_aqi_lstm)) + 1, predicted_aqi_lstm, label='Predicted AQI (LSTM)', linestyle='--')
plt.plot(np.arange(len(predicted_aqi_rf)) + 1, predicted_aqi_rf, label='Predicted AQI (Random Forest)', linestyle='-.')
plt.xlabel('Day')
plt.ylabel('Predicted AQI')
plt.title('Predicted AQI for Next Ten Days')
plt.legend()
plt.grid(True)
plt.show()

# Create DataFrame for predicted AQI values
day_numbers = np.arange(1, 11)
predicted_aqi_table = pd.DataFrame({'Day': day_numbers,
                                    'Predicted AQI (LSTM)': predicted_aqi_lstm,
                                    'Predicted AQI (Random Forest)': predicted_aqi_rf})
print("Predicted AQI for Next Ten Days:")
print(predicted_aqi_table)