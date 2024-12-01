import pandas as pd

import matplotlib.pyplot as plt

# Load the data
data = pd.read_csv('training.csv')

date = data.iloc[:, 0]
avg_temperature = data.iloc[:, 1]
max_temperature = data.iloc[:, 2]
min_temperature = data.iloc[:, 3]
avg_humidity = data.iloc[:, 4]
max_humidity = data.iloc[:, 5]
min_humidity = data.iloc[:, 6]
power = data.iloc[:, 7]

# Plot the data
plt.figure(figsize=(10, 5))
plt.plot(date, avg_temperature, label='Average Temperature')
plt.plot(date, max_temperature, label='Max Temperature')
plt.plot(date, min_temperature, label='Min Temperature')
plt.plot(date, avg_humidity, label='Average Humidity')
plt.plot(date, max_humidity, label='Max Humidity')
plt.plot(date, min_humidity, label='Min Humidity')
plt.plot(date, power, label='Power')
plt.xlabel('Date')
plt.ylabel('Values')
plt.title('Weather Data')
plt.legend()
plt.show()