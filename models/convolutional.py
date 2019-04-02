import numpy as np
from keras.layers import Conv2D, BatchNormalization, Flatten
from sklearn.model_selection import train_test_split
import os

from NeuralNetworks.TicTacToe.framework.data_manager import DataManager
from NeuralNetworks.TicTacToe.framework.frame import Frame


class ConvolutionalModel:

    def __init__(self, model_name):
        self.name = model_name
        self.model_path = f'{os.curdir}/{self.name}.h5'
        self.model = self.get_model()

    def get_model(self):
        if os.path.exists(self.model_path):
            from keras.models import load_model
            model = load_model(self.model_path)
            print("model found at "+self.model_path)
        else:
            print("model not found!")
            from keras.models import Sequential
            from keras.layers import Dense
            model = Sequential()
            model.add(Conv2D(64, (2, 2), activation='relu', input_shape=(3, 3, 2)))
            model.add(Conv2D(64, (2, 2), activation='relu'))
            model.add(Flatten())
            model.add(Dense(9, activation='softmax'))
        model.compile(optimizer='rmsprop', loss='mse', metrics=['accuracy'])
        # model.compile(loss='mean_squared_error', optimizer='Adam', metrics=['accuracy'])
        return model

    @staticmethod
    def get_inputs_and_outputs(matches):
        inputs = []
        outputs = []
        for match in matches:
            for insert in match['inserts']:
                if not insert['best']:
                    continue
                inputs.append(insert['frame'])
                outputs.append(insert['position'])

        inputs = np.array([Frame.categorize_inputs(input_frame) for input_frame in inputs])
        outputs = np.array(Frame.categorize_outputs(outputs))

        return train_test_split(inputs, outputs, test_size=0.2)

    def train(self, epochs=250, data_manager=None):
        data_manager = data_manager if data_manager else DataManager()
        matches = data_manager.get()
        x_train, x_test, y_train, y_test = self.get_inputs_and_outputs(matches)
        self.model.fit(x_train, y_train, batch_size=128, epochs=epochs, verbose=1, validation_data=(x_test, y_test))
        self.model.save(self.model_path)
