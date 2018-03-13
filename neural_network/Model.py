import os
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

'''
    MODEL CLASS
'''

class Model():
    def __init__(self, model_name):
        self.model_name = model_name
        self.build_model()
        # nonce for intermediate weight writes
        self.weights_counter = 0

    '''
        returns keras model
            12 --> 12 --> 2 Dense Network
    '''
    def build_model(self):
        model = Sequential()
        model.add(Dense(12, activation='relu', input_dim=4))
        model.add(Dense(12, activation='relu'))
        model.add(Dense(2))
        model.compile(Adam(lr=0.001), 'mse')
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

''' TESTING '''
if __name__ == '__main__':
    model = Model('dense_network')
    model.build_model()
    model.save_model()
    model.save_weights()