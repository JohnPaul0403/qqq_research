import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler

class Preprocess:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def rsi(self, period=14):
        delta = self.df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def macd(self, short=12, long=26, signal=9):
        short_ema = self.df['Close'].ewm(span=short, adjust=False).mean()
        long_ema = self.df['Close'].ewm(span=long, adjust=False).mean()
        macd = short_ema - long_ema
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        return macd, signal_line
    
    def sma(self, window=20):
        return self.df['Close'].rolling(window=window).mean()
    
    def cci(self, period=20):
        tp = (self.df['High'] + self.df['Low'] + self.df['Close']) / 3
        sma = tp.rolling(window=period).mean()
        mad = tp.rolling(window=period).apply(lambda x: np.mean(np.abs(x - np.mean(x))))
        return (tp - sma) / (0.015 * mad)
    
    def aroon(self, period=25):
        aroon_up = 100 * (self.df['High'].rolling(window=period).apply(lambda x: x.argmax()) - period) / period
        aroon_down = 100 * (self.df['Low'].rolling(window=period).apply(lambda x: x.argmin()) - period) / period
        return aroon_up, aroon_down
    
    def add_features(self):
        self.df['RSI'] = self.rsi()
        self.df['MACD'], self.df['Signal Line'] = self.macd()
        self.df['SMA'] = self.sma()
        self.df['CCI'] = self.cci()
        self.df['Aroon Up'], self.df['Aroon Down'] = self.aroon()
        return self.df
    
    def fill_missing(self):
        imputer = SimpleImputer(strategy='mean')
        self.df = pd.DataFrame(imputer.fit_transform(self.df), columns=self.df.columns)
        return self.df
    
    def singleStepSampler(self, window):
        xRes = []
        yRes = []
        for i in range(0, len(self.df) - window):
            res = []
            for j in range(0, window):
                r = []
                for col in self.df.columns:
                    r.append(self.df[col][i + j])
                res.append(r)
            xRes.append(res)
            yRes.append(self.df[['Open', 'Close']].iloc[i + window].values)
        return np.array(xRes), np.array(yRes)
    
    def get_processed_data(self):
        self.df.reset_index(inplace=True)
        self.df = self.add_features()
        self.df = self.fill_missing()
        scaler = MinMaxScaler()
        self.df[self.df.columns] = scaler.fit_transform(self.df[self.df.columns])
        return self.df
    
    def get_data(self, train_split, window):
        x, y = self.singleStepSampler(window)
        split_idx = int(len(x) * train_split)
        x_train, y_train = x[:split_idx], y[:split_idx]
        x_test, y_test = x[split_idx:], y[split_idx:]
        return x_train, y_train, x_test, y_test