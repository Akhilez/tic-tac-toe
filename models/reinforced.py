import os
from time import time

import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.callbacks import TensorBoard

from framework.data_manager import DataManager
from framework.frame import Frame


class ReinforcedModel:

    def __init__(self, model_name):
        self.name = model_name
        self.model_path = f'{os.curdir}/{self.name}.h5'
        self.model = self.get_model()

    def train(self, epochs=250, data_manager=None):
        data_manager = data_manager if data_manager else DataManager(file_name=f'{os.curdir}/../data.json')
        matches = data_manager.get()
        inputs, outputs = self.get_inputs_and_outputs(matches)
        self.model.fit(inputs, outputs, epochs=epochs, verbose=1)
        self.model.save(self.model_path)

    def get_model(self):
        if os.path.exists(self.model_path):
            from keras.models import load_model
            model = load_model(self.model_path)
            print("model found at " + self.model_path)
        else:
            print("model not found!")
            model = Sequential()
            model.add(Dense(9, activation='sigmoid', input_shape=(27,)))
            model.add(Dense(9, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
        return model

    @staticmethod
    def get_inputs_and_outputs(matches):
        inputs = []
        outputs = []
        for match in matches:

            if match['winner'] == Frame.X:  # MAKING AN ASSUMPTION THAT Reinforced Player is X
                for insert in match['inserts']:
                    inputs.append(Frame.categorize_inputs(insert['frame']))
                    outputs.append(insert['position'])
        outputs = np.array(Frame.categorize_outputs(outputs))
        inputs = np.array(inputs).reshape(len(inputs), 27)

        return inputs, outputs


if __name__ == '__main__':
    ReinforcedModel('Reinforced_1').train()
