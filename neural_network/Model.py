import os
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

'''
    MODEL UTILS
    from Model import *
'''

'''
    returns keras model
        12 --> 12 --> 2 Dense Network
'''
def build_model():
    model = Sequential()
    model.add(Dense(12, activation='relu', input_dim=4))
    model.add(Dense(12, activation='relu'))
    model.add(Dense(2))
    model.compile(Adam(lr=0.001), 'mse')
    return model

'''
    params
        model: [json] keras model
        model_name: [string]
    writes json to ./models/model_name.json
'''
def save_model(model, model_name):
    if not os.path.exists('./models'):
        os.makedirs('./models')
    model_file = open('./models/{}.json'.format(model_name), 'w')
    model_file.write(model)
    model_file.close()

''' TESTING '''
if __name__ == '__main__':
    model = build_model()
    save_model(model.to_json(), 'dense_model_1')
