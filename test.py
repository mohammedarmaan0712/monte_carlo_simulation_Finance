import numpy as np
import pandas as pd
import requests
import json
import yfinance as yf
import matplotlib.pyplot as plt


# Your FRED API Key
API_KEY = "f40a14b7a960cba7b015625ba80b308b"


series_id_list = ['^GSPC',  # S&P 500 Index
    'UNRATE',  # U.S. Unemployment Rate
    'GDP',  # U.S. GDP Growth
    'CPIAUCNS',  # U.S. Inflation Rate (CPI)
    'GS10',  # U.S. 10-Year Treasury Yield
    'FEDFUNDS',  # U.S. Federal Funds Rate
    'CONCCONF',  # U.S. Consumer Confidence Index
    'RSXFS',  # U.S. Retail Sales
    'ISM/MAN_PMI',  # U.S. Manufacturing PMI
    'HOUST',  # U.S. Housing Starts
    'DJIA',  # Dow Jones Industrial Average
    'INDPRO',  # Industrial Production Index
    'M2SL',  # U.S. Money Supply (M2)
    'MORTGAGE30US',  # 30-Year Fixed Mortgage Rate
    'PPIACO',  # Producer Price Index (PPI)
    'CIVPART',  # Civilian Participation Rate
    'PCE',  # Personal Consumption Expenditures
    'T10YIE',  # U.S. 10-Year Inflation Expectation
    'BAMLH0A0HYM2',  # High Yield Corporate Bonds
    'WTI',  # West Texas Intermediate Crude Oil Price
    ]

'''
TODO:
1. Request inflation data from FRED API
2. Request stock data from yahoo finance
3. Find the correlation R constant between the two data sets.
4. Repeat the same for Interest rate and unemployment rate.
5. Create a function to perform Monte Carlo simulation.
'''


def monte_carlo():
    pass
    '''
    Monte carlo simulation:
    Numpy is used to generate random numbers and perform calculations.
    find considerable random numbers to simulate a process.
    The simulation is run for a large number of iterations to get an accurate estimate.
    Then the results are used to plot a graph.
    The graph shows the distribution of the random numbers generated.
    Finding statistics on this graph can help in predicting the outcome of the process.

   '''
    #generate random numbers
    num_samples = 1000000 #(1 million samples)
    A = np.random.uniform(3,5,num_samples)
    B = np.random.uniform(1,4,num_samples)
    C = A + B
    # Plot the random numbersp
    plt.hist(C.flatten(), bins=50, color='blue', alpha=0.7)
    plt.title("Distribution of Random Numbers")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.show()
    #find the statistics:
    mean = np.mean(C)
    std_dev = np.std(C)
    variance = np.var(C)
    print(f"Mean: {mean}")
    print(f"Standard Deviation: {std_dev}")
    print(f"Variance: {variance}")

def getVariableData():
    
    todo = [] # List to store the data for each series_id
    for series_id in series_id_list:
        url = f"https://api.stlouisfed.org/fred/series/observations?series_id={series_id}&api_key={API_KEY}&file_type=json" 
        response = requests.get(url)
        data = response.json()
        #if the request was successful
        if 'observations' in data:
            # Extract the observations from the response
            # and convert them to a dictionary with date as key and value as float
            observations = data["observations"]
            # Convert to tuples with date and value
            date_values = {}
            for obs in observations: 
                try:
                    date_values[obs["date"]] = float(obs["value"])
                except ValueError:
                    # Handle cases where value is not a float
                    continue
            todo.append(date_values)
            
    
    return todo

def getStockData(ticker):
    stock = yf.Ticker(ticker)  # stock data
    hist = stock.history(period="max")  # Retrieve max of historical data
    return hist  # Convert to dictionary with dates as keys and prices as values

def getCorrelation(variable_data, stock_data, series_id_index):
    

    # Align data by dates
    common_dates = set(variable_data.keys()).intersection(stock_data.index.strftime('%Y-%m-%d'))
    variable_values = [variable_data[date] for date in common_dates]
    stock_values = [stock_data.loc[date, "Close"] for date in common_dates]

    # Calculate correlation
    correlation = np.corrcoef(variable_values, stock_values)[0, 1]
    print(f"Correlation between {series_id_list[series_id_index]}  and stock prices: {correlation}")
    return correlation

def main():
    # Get variable data from FRED
    variable_data = getVariableData()
    
    # Get stock data from Yahoo Finance
    ticker = input("Enter the stock ticker symbol (e.g., ^GSPC for S&P 500): ")
    if not ticker:
        ticker = '^GSPC'  # Default to S&P 500 if no input is given
    stock_data = getStockData(ticker)  # S&P 500 Index
    
    # Calculate correlation
    correlations = []
    i=0
    for data in variable_data:
        correlation = getCorrelation(data, stock_data, i)
        correlations.append(correlation)
        i+=1
        print(f"Correlation: {correlation}")
    # Print all correlations


if __name__ == "__main__":
    main()