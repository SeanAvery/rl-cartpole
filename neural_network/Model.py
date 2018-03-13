import os
import h5py
import numpy as np
import keras
from keras.models import Sequential, model_from_json
from keras.layers import Dense
from keras.optimizers import Adam

'''
    MODEL CLASS
'''

class Model():
    def __init__(self, model_name):
        self.model_name = model_name
        # nonce for intermediate weight writes
        self.weights_counter = 0

        # hyperparemters
        self.alpha=0.1
        self.alpha_decay=0.01
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_decay = 0.95
        self.epsilon_min = 0.05
        self.batch_size = 32

        # int model
        self.build_model()


    '''
        returns keras model
            12 --> 12 --> 2 Dense Network
    '''
    def build_model(self):
        model = Sequential()
        model.add(Dense(12, activation='relu', input_dim=4))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(2))
        model.compile(optimizer=Adam(lr=self.alpha, decay=self.alpha_decay), loss='mse')
        self.model = model

    '''
        params
            model: [json] keras model
            model_name: [string]
        writes json to ./models/model_name.json
    '''
    def save_model(self):
        if not os.path.exists('./models'):
            os.makedirs('./models')
        model_file = open('./models/{}.json'.format(self.model_name), 'w')
        model_file.write(self.model.to_json())
        model_file.close()

    def save_weights(self):
        if not os.path.exists('./models/{}'.format(self.model_name)):
            os.makedirs('./models/{}'.format(self.model_name))
        file_path = './models/{0}/{1}.h5'.format(self.model_name, self.weights_counter)
        self.model.save_weights(file_path)

    def checkpoint_model(self):
        keras.callbacks.ModelCheckpoint(
            '{0}/{1}'.format(self.model_name, self.weights_counter),
            monitor='val_loss',
            verbose=0,
            save_best_only=False,
            save_weights_only=False,
            mode='auto',
            period=1)

    def extract_layers(self):
        for layer in self.model.layers:
            weights = layer.get_weights()
            print('weights', weights, type(weights), np.array(weights[0]).shape)

    def load_model_graph(self, nonce):
        weights = h5py.File('models/{0}/{1}.h5'.format(self.model_name, nonce), 'r+')
        for thing in weights.keys():
            data = list(weights[thing])[0]
            print('data', data)
            # print('thing', thing, type(thing))

        print('weights', weights)
        # self.model.load_weights('models/{0}/{1}.h5'.format(self.model_name, nonce))


''' TESTING '''
if __name__ == '__main__':
    model = Model('dense_network')
    model.build_model()
    model.save_model()
    model.save_weights()
    model.extract_layers()
