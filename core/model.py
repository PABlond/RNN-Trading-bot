import os
import numpy as np
import datetime as dt
from keras.layers import Dense, Activation, Dropout, LSTM
from keras.models import Sequential
from keras.callbacks import EarlyStopping, ModelCheckpoint
class Model():  
    def __init__(self):
        self.model = Sequential()

    def build_model(self, configs):
        loss, optimizer, metrics = 'mean_squared_error', "adam", ['mean_squared_error']
        self.model.add(LSTM(100, input_shape=(
                    249, 1), return_sequences=True))
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
        save_fname = os.path.join(save_dir, 'model.h5' )
        callbacks = [
            EarlyStopping(monitor='loss', patience=1),
            ModelCheckpoint(filepath=save_fname, monitor='loss', save_best_only=True)
        ]
        self.model.fit(
            X_train, y_train, validation_data=(X_test,y_test),
			epochs=epochs,
			batch_size=batch_size,
			callbacks=callbacks
		)
        self.model.save(save_fname)
    

