import os
import numpy as np
import datetime as dt
from keras.layers import Dense, Activation, Dropout, LSTM
from keras.models import Sequential
from keras.callbacks import EarlyStopping, ModelCheckpoint
class Model():  
    def __init__(self, model_config):
        self.model = Sequential()
        self.loss = model_config['loss']
        self.optimizer = model_config['optimizer']
        self.metrics = model_config['metrics']
        self.step_size = model_config['step_size']
        self.filename = model_config['filename']
        self.callbacks = model_config['callbacks']

    def build_model(self, configs):
        self.model.add(LSTM(100, input_shape=(
                    self.step_size, 1), return_sequences=True))
        self.model.add(Dropout(0.2))
        self.model.add(LSTM(100, input_shape=(
                    None, None), return_sequences=True))
        self.model.add(LSTM(100, input_shape=(
                    None, None), return_sequences=False))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(1, activation="linear"))
        self.model.compile(
            loss=self.loss, optimizer=self.optimizer, metrics=self.metrics)

    def train(self, X_train, X_test, y_train, y_test, epochs, batch_size, save_dir):
        save_fname = os.path.join(save_dir, self.filename )
        callbacks = [
            EarlyStopping(monitor=self.callbacks["monitor"], patience=self.callbacks["patience"]),
            ModelCheckpoint(filepath=save_fname, monitor=self.callbacks["monitor"], save_best_only=self.callbacks['save_best_only'])
        ]
        self.model.fit(
            X_train, y_train, validation_data=(X_test,y_test),
			epochs=epochs,
			batch_size=batch_size,
			callbacks=callbacks
		)
        self.model.save(save_fname)
    

