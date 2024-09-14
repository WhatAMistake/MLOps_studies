from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


class DisorderCLF:
    def __init__(self, input_shape, layers): 
        # input_shape = X.shape[1]
        # layers = [64, 32, 4]
        self.input_shape = input_shape
        self.layers = layers

    def model(self):
        model = Sequential()
        model.add(Dense(self.layers[0], activation='relu', input_shape=(self.input_shape,)))
        model.add(Dense(self.layers[1], activation='relu'))
        model.add(Dense(self.layers[2], activation='sigmoid'))

        return model
