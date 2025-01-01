import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

class ARIMAModel:
    def __init__(self, train_data, test_data, exog_train=None, exog_test=None):
        """
        Initialize the ARIMAModel class.

        Args:
            train_data (pd.Series): Training data for the time series model.
            test_data (pd.Series): Test data for the time series model.
            exog_train (pd.DataFrame): Exogenous variables for training.
            exog_test (pd.DataFrame): Exogenous variables for testing.
        """
        self.model = None
        self.train_data = train_data
        self.test_data = test_data
        self.exog_train = exog_train
        self.exog_test = exog_test
        
    def train(self, order=(1, 1, 1)):
        # ARIMA Model
        self.model = ARIMA(self.train_data, order=order, exog=self.exog_train if self.exog_train is not None else None)
        self.fitted_model = self.model.fit()
        print(self.fitted_model.summary())
    
    def evaluate(self):
        # Make predictions
        predictions = self.fitted_model.forecast(steps=len(self.test_data), exog=self.exog_test if self.exog_test is not None else None)
        
        # Reverse the scaling for actual and predicted values
        scaler = StandardScaler()
        scaler.fit(self.train_data.reshape(-1, 1))  # Re-fit scaler on training data
        actual = scaler.inverse_transform(self.test_data.reshape(-1, 1))
        predictions = scaler.inverse_transform(predictions.reshape(-1, 1))
        
        # Evaluate performance using MAE, RMSE, etc.
        mae = mean_absolute_error(actual, predictions)
        rmse = np.sqrt(mean_squared_error(actual, predictions))
        
        print(f'Mean Absolute Error: {mae}')
        print(f'Root Mean Squared Error: {rmse}')
        
        # Plot actual vs predicted values
        plt.figure(figsize=(10, 6))
        plt.plot(actual, label='Actual', color='blue')
        plt.plot(predictions, label='Predicted', color='red')
        plt.legend()
        plt.title('ARIMA Model - Actual vs Predicted')
        plt.show()

    def save_model(self, filename='arima_model.pkl'):
        self.fitted_model.save(filename)
        print(f'Model saved as {filename}')
