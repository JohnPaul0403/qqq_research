import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose

# Train-Test Split for Time Series Data
def train_test_split(df, test_size=0.2):
    """
    Perform train-test split on time series data.
    
    Args:
        df (pd.DataFrame): Input DataFrame containing the time series data.
        test_size (float): Proportion of the data to include in the test split.
        
    Returns:
        train (pd.DataFrame): DataFrame containing training data.
        test (pd.DataFrame): DataFrame containing test data.
    """
    split_index = int((1 - test_size) * len(df))
    train, test = df.iloc[:split_index], df.iloc[split_index:]
    
    return train, test

# Function to plot ACF and PACF
def plot_correlations(df, column, lags=20):
    """
    Plot ACF and PACF for the given time series column.
    
    Args:
        df (pd.DataFrame): Input DataFrame containing the time series data.
        column (str): Name of the column to analyze.
        lags (int): Number of lags to include in the plots.
    """
    # Ensure the column is stationary
    ts = df[column].diff().dropna()

    # Plot ACF and PACF
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    plot_acf(ts, ax=axes[0], lags=lags, title='Autocorrelation (ACF)')
    plot_pacf(ts, ax=axes[1], lags=lags, title='Partial Autocorrelation (PACF)')

    plt.tight_layout()
    plt.show()

# Function to decompose time series
def decompose(df: pd.DataFrame, column: str, model='additive', period=None):
    """
    Decompose a time series into trend, seasonal, and residual components.
    
    Args:
        df (pd.DataFrame): Input DataFrame containing the time series data.
        column (str): Name of the column to decompose.
        model (str): Type of decomposition ('additive' or 'multiplicative').
        period (int): Period of the seasonal component.
        
    Returns:
        decomposition (pd.DataFrame): DataFrame containing trend, seasonal, and residual components.
    """
    # Ensure the column is stationary
    ts = df[column].diff().dropna()
    
    # Perform decomposition
    decomposition = seasonal_decompose(ts, model=model, period=period)
    
    return decomposition

# Function to visualize decomposition
def plot_decomposition(decomposition):
    """
    Visualize the components of a time series decomposition.
    
    Args:
        decomposition: Result of time series decomposition.
    """
    plt.figure(figsize=(12, 8))
    plt.subplot(411)
    plt.plot(decomposition.observed, label='Original', color='black')
    plt.legend(loc='upper left')
    
    plt.subplot(412)
    plt.plot(decomposition.trend, label='Trend', color='blue')
    plt.legend(loc='upper left')
    
    plt.subplot(413)
    plt.plot(decomposition.seasonal, label='Seasonal', color='green')
    plt.legend(loc='upper left')
    
    plt.subplot(414)
    plt.plot(decomposition.resid, label='Residual', color='red')
    plt.legend(loc='upper left')
    
    plt.tight_layout()
    plt.show()