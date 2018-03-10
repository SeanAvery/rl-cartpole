from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

'''
    MODEL UTILS
'''

def build_model():
    model = Sequential()
    model.add(Dense(12, activation='relu', input_dim=4))
    model.add(Dense(2))
    model.compile(Adam(lr=0.001), 'mse')
    return model

if __name__ == '__main__':
    model = build_model()
    print('model', model)
