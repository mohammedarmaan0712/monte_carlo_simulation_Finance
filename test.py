import numpy as np
import pandas as pd
import requests
import json
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from helper import *  # Assuming helper.py contains necessary functions or classes
import datetime



'''
TODO:
1. Request inflation data from FRED API
2. Request stock data from yahoo finance
3. Find the correlation R constant between the two data sets.
4. Repeat the same for Interest rate and unemployment rate.
5. Create a function to perform Monte Carlo simulation.
'''


def regression(stock_data, variable_data):
    model = LinearRegression()
    common_dates = set(variable_data.keys()).intersection(stock_data.index.strftime('%Y-%m-%d'))
    common_dates = list(common_dates)
    common_dates = sorted([date for sublist in common_dates for date in sublist])
    variable_values = [variable_data[date] for date in common_dates]
    stock_values = [stock_data.loc[date, "Close"] for date in common_dates]

    # Ensure that the lengths of variable_values and stock_values match
    if len(variable_values) != len(stock_values):
        raise ValueError("Mismatch in lengths of variable_values and stock_values")
    X = np.array(common_dates).reshape(-1, 1)  # Dates as dependent variable
    print(common_dates)
    return
    variable = np.array(variable_values).reshape(-1, 1)  # Independent variable
    stock = np.array(stock_values)  # Dependent variable

    # Fit the regression model
    model.fit(variable, stock)

    # Plot the scatter plot
    plt.scatter(X, stock, color='blue', label='Data Points')
#    plt.plot(variable, model.predict(variable), color='red', label='Regression Line')
    plt.xlabel('Dates')
    plt.ylabel('Stock Price')
    plt.title('Scatter Plot with Regression Line')
    plt.legend()
    plt.show()
    # Show the scatter plot
    coef, intercept = model.coef_, model.intercept_  # Get the coefficients and intercept of the regression line
    print(f"Regression Coefficient: {coef}")
    print(f"Intercept: {intercept}")
    # Use the model to predict future values
    future_values = np.array(float(input("future indicator value:"))).reshape(-1, 1)
    future_values = model.predict(future_values)  # Predict future stock values based on the model
    #print(model.predict(np.array(future_values)))  # Predict future values based on the model
    # Predict future values for the stock based on the variable data
    return future_values


def main():
    # Get variable data from FRED
    series_id = input("Enter the FRED series ID (e.g., CPIAUCNS for Consumer Price Index): ")
    variable_data = getVariableData(series_id)
    
    # Get stock data from Yahoo Finance
    ticker = input("Enter the stock ticker symbol (e.g., ^GSPC for S&P 500): ")
    if not ticker:
        ticker = '^GSPC'  # Default to S&P 500 if no input is given
    stock_data = getStockData(ticker)  # S&P 500 Index
    # Convert stock_data to a dictionary with date and close value
    stock_data = stock_data[['Close']].to_dict('index')
    stock_data = {date.strftime('%Y-%m-%d'): data['Close'] for date, data in stock_data.items()}
    
    common_dates = set(variable_data.keys()) & set(stock_data.keys())
    # Create list of structs (dictionaries)
    structured_data = {
        date: { "stock": stock_data[date], "indicator": variable_data[date]} 
        for date in sorted(common_dates)
    }
    correlation = np.corrcoef( [data['stock'] for data in structured_data.values()], [data['indicator'] for data in structured_data.values()] )[0, 1]
    print(f"Correlation between {series_id} and {ticker}: {correlation}")
    plt.plot(list(structured_data.keys()), [data['stock']*3 for data in structured_data.values()], color='blue', label='Stock Price')
    plt.plot(list(structured_data.keys()), [data['indicator'] for data in structured_data.values()], color='red', label='Indicator Value')
    plt.show()
    return
    for date in structured_data:
        plt.plot_date(date, structured_data[date]['stock'], color='blue', label='Stock Price')
        plt.plot_date(date, structured_data[date]['indicator'], color='red', label='Indicator Value')
    
    plt.show()  # Show the plot of stock prices and indicator values
    plt.show()  # Show the plot of stock prices and indicator values
      # Debugging line to check structured data

    return
    

    # Calculate correlation
    correlations = []
    i=0
    '''for data in variable_data:
        correlation = getCorrelation(data, stock_data, i)
        correlations.append(correlation)
        i+=1
        print(f"Correlation: {correlation}")'''
    '''for data in variable_data:
        prediction = regression(stock_data, data)
        i+=1
        print(f"Prediction: {prediction} for {series_id_list[i]}")'''
    prediction = regression(stock_data, variable_data)
    i+=1
    print(f"Prediction: {prediction} for {series_id_list[i]}")
    
    
    # Print all correlations

if __name__ == "__main__":
    main()