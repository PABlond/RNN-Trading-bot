import os
import numpy as np
import datetime as dt
from keras.layers import Dense, Activation, Dropout, LSTM
from keras.models import Sequential
from keras.callbacks import EarlyStopping, ModelCheckpoint
class Model():  
    def __init__(self, model_config):
        self.cfg = model_config
        self.model = Sequential()

    def build_model(self, configs):
        loss, optimizer, metrics, step_size = self.cfg['loss'], self.cfg['optimizer'], self.cfg['metrics'],  self.cfg['step_size']
        self.model.add(LSTM(100, input_shape=(
                    step_size, 1), return_sequences=True))
        self.model.add(Dropout(0.2))
        self.model.add(LSTM(100, input_shape=(
                    None, None), return_sequences=True))
        self.model.add(LSTM(100, input_shape=(
                    None, None), return_sequences=False))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(1, activation="linear"))
        self.model.compile(
            loss=loss, optimizer=optimizer, metrics=metrics)

    def train(self, X_train, X_test, y_train, y_test, epochs, batch_size, save_dir):
        callbacks, loss, optimizer, metrics, filename = self.cfg['callbacks'], self.cfg['loss'], self.cfg['optimizer'], self.cfg['metrics'], self.cfg['filename']
        monitor, save_best_only, patience = callbacks['monitor'], callbacks['save_best_only'], callbacks['patience']
        save_fname = os.path.join(save_dir, filename )
        callbacks = [
            EarlyStopping(monitor=monitor, patience=patience),
            ModelCheckpoint(filepath=save_fname, monitor=monitor, save_best_only=save_best_only)
        ]
        self.model.fit(
            X_train, y_train, validation_data=(X_test,y_test),
			epochs=epochs,
			batch_size=batch_size,
			callbacks=callbacks
		)
        self.model.save(save_fname)
    

