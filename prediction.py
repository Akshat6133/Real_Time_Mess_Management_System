import pandas as pd

# Read the transaction data
data = pd.read_csv("card_transaction.csv")

# Assuming data is stored in a DataFrame named 'data'
# Convert 'transaction_date' column to datetime format
data['transaction_date'] = pd.to_datetime(data['transaction_date'], format='%d-%m-%Y %H:%M')

# Group by 'transaction_date' and count unique student IDs
count_per_timestamp = data.groupby('transaction_date')['account_from'].nunique()

# print(count_per_timestamp)

print ("-------------------------------------------------------------------------------------------------")

# Group data by transaction_date and count the number of transactions
footprint_data = data.groupby('transaction_date').size().reset_index(name='footprint')
print(footprint_data)
print ("-------------------------------------------------------------------------------------------------")

# Rename columns as required by Prophet
footprint_data = footprint_data.rename(columns={'transaction_date': 'ds', 'footprint': 'y'})
print(footprint_data)
# print(type(footprint_data))

import matplotlib.pyplot as plt
plt.plot(footprint_data)
plt.show()
# from fbprophet import Prophet # type: ignore
# Initialize and fit Prophet model
# model = Prophet()
# model.fit(footprint_data)

# Make future dataframe for predictions
# future = model.make_future_dataframe(periods=30)  # Example: Predict for next 30 days

# Make predictions
# forecast = model.predict(future)

# Plot the forecast
# fig = model.plot(forecast)


from prophet import Prophet # type: ignore 

m = Prophet()
m.fit(footprint_data)
future = m.make_future_dataframe(periods=10)
print(future.tail())
forecast = m.predict(future) # new dataframe with predictions included
# forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']] 
# print(type(forecast))
print(forecast)