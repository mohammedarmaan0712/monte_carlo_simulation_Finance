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
Get inflation data from FRED API
Get Consumer Confidence, government spending, investment, wage reports, wti crude, money supply, interest Index data from FRED API
form a matrix with all the data on the left and the inflation on the right
perform multivariate regression on the matrix to find the coefficients for each variable
'''

def getVariableData(series_id):
    
    url = f"https://api.stlouisfed.org/fred/series/observations?series_id={series_id}&api_key={API_KEY}&file_type=json" 
    response = requests.get(url)
    data = response.json()
        #if the request was successful
    if 'observations' in data:
        # Extract the observations from the response
        #             # and convert them to a dictionary with date as key and value as float
        observations = data["observations"]
        # Convert to tuples with date and value
        date_values = {}
        for obs in observations: 
            try:
                date_values[obs["date"]] = float(obs["value"])
            except ValueError:
                # Handle cases where value is not a float
                continue
        return date_values
    

def getConsumerConfidence(series_id):
    url = f"https://api.stlouisfed.org/fred/series/observations?series_id={series_id}&api_key={API_KEY}&file_type=json" 
    data = requests.get(url).json()['observations']
    date_values = {}
    for obs in data:
        try:
            date_values[obs['date']] = float(obs["value"])  # Convert value to float
        except ValueError:
            pass
     # Debugging line to check the response
        #if the request was successful
    
 
    return date_values


def main():
    inflation_data = getVariableData('CPIAUCNS')  # U.S. Inflation Rate (CPI)
    #inflation data is a dictonary with date as key and value as float
    consumer_confidence_data = getConsumerConfidence('UMCSENT')  # U.S. Consumer Confidence Index
    #consumer_confidence data is a dictonary with date as key and value as float
    m2 = getVariableData('M2SL')  # U.S. Money Supply (M2)
    #m2 data is a dictonary with date as key and value as float
    ppi = getVariableData('PPIACO')  # Producer Price Index (PPI)
    #ppi data is a dictonary with date as key and value as float
    wti = getVariableData('IR14200')  # West Texas Intermediate Crude Oil Price
    #wti data is a dictonary with date as key and value as float
    interest = getVariableData('FEDFUNDS') # U.S. Federal Funds Rate
    print(interest)

    pass


if __name__ == "__main__":
    main()