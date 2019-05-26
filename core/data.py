import numpy as np
import pandas as pd
import requests
import fxcmpy
from sklearn.preprocessing import StandardScaler
import dask.dataframe as dd

class DataModel():

    def __init__(self):
        self.con = None
        self.data = []
        self.dataX = np.array([])
        self.dataY = np.array([])
        self.scaler = StandardScaler()

    def req_data(self, currency, cols, api_key, con, period):        
        self.con = con
        data = con.get_candles(currency, period=period, number=250)
        self.data = data[cols].values.tolist()[-249:]
        return True

    def get_train_data(self, data, seq_len):
        self.data = data
        data_x, data_y = [], []
        for i in range(0, len(self.data) - seq_len):
            x, y = self._next_window(i, seq_len)
            data_x.append(x)
            data_y.append(y)
        self.dataX = np.array(data_x)
        self.dataY = np.array(data_y)
        return self.dataX 
    
    def get_predict_data(self, seq_len):
        data_x, data_y = [], []
        x = self._next_window_predict(0, seq_len)
        data_x.append(x)
        self.dataX = np.array(data_x)
        return self.dataX 

    def _next_window_predict(self, i, seq_len):
        data = self.data[i:i+seq_len]
        data = self.data_scaling(data)        
        if len(data) > 0:
            x = data[0:len(data)]
            return x

    def _next_window(self, i, seq_len):
        '''Generates the next data window from the given index location i'''
        data = self.data[i:i+seq_len]
        data = self.data_scaling(data)        
        if len(data) > 0:
            x = data[0:len(data)-1]
            y = data[-1, [0]]
            return x, y

    def data_scaling(self, data):
        data = np.array(data)
        self.scaler.fit(data.reshape(-1, 1))
        data = self.scaler.transform(data.reshape(-1, 1))
        return data

    def data_scaling_inv(self, data):
        return self.scaler.inverse_transform(data)

    def predict_sequences_multiple(self, model, data, window_size, prediction_len):
        predictions = []
        for i in range(prediction_len):
            pred = model.predict(data)
            predictions.append(pred)
            data = data.tolist()
            del data[0][0]
            data[0].append(pred.reshape(1, 1, 1))
            data = np.array(data)
        return data[0][:prediction_len]