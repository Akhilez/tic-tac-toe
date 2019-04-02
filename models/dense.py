import os

import numpy as np
from sklearn.model_selection import train_test_split

from NeuralNetworks.TicTacToe.framework.data_manager import DataManager
from NeuralNetworks.TicTacToe.framework.frame import Frame


class DenseModel:

    def __init__(self, model_name):
        self.name = model_name
        self.model_path = f'{os.curdir}/{self.name}.h5'
        self.model = self.get_model()

    def train(self, epochs=250, data_manager=None):
        data_manager = data_manager if data_manager else DataManager()
        matches = data_manager.get()
        x_train, x_test, y_train, y_test = self.get_inputs_and_outputs(matches)
        self.model.fit(x_train, y_train, batch_size=128, epochs=epochs, verbose=1, validation_data=(x_test, y_test))
        self.model.save(self.model_path)

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
            model.add(Dense(9, activation='relu', input_shape=(18,)))
            model.add(Dense(9, activation='softmax'))
        model.compile(loss='mean_squared_error', optimizer='Adam', metrics=['accuracy'])
        return model

    def get_inputs_and_outputs(self, matches):
        inputs = []
        outputs = []
        for match in matches:
            for insert in match['inserts']:
                if not insert['best']:
                    continue
                inputs.append(Frame.categorize_inputs(insert['frame']))
                outputs.append(insert['position'])
        outputs = np.array(Frame.categorize_outputs(outputs))
        inputs = np.array(inputs).reshape(len(inputs), 18)

        return train_test_split(inputs, outputs, test_size=0.2)


if __name__ == '__main__':
    DenseModel('Dense_1').train()
