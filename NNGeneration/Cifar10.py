import save_keras_model as saver
import time
import matplotlib.pyplot as plt
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.layers import Activation
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras import backend as K
from keras import constraints

import pickle
import h5py
from pathlib import Path

if K.backend()=='tensorflow':
    K.set_image_dim_ordering("th")
 
# Import Tensorflow with multiprocessing
import tensorflow as tf
import multiprocessing as mp
 
# Loading the CIFAR-10 datasets
from keras.datasets import cifar10


save_folder = 'data/CIFAR-10-CNN-PositiveWeights/'


batch_size = 32 
num_classes = 10 
epochs = 100 

(x_train, y_train), (x_test, y_test) = cifar10.load_data() 

y_train = np_utils.to_categorical(y_train, num_classes)
y_test = np_utils.to_categorical(y_test, num_classes)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train  /= 255
x_test /= 255

def base_model():
    model = Sequential()
 
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=x_train.shape[1:]))
    model.add(Dropout(0.2))
 
    model.add(Conv2D(32,(3,3),padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
 
    model.add(Conv2D(64,(3,3),padding='same',activation='relu'))
    model.add(Dropout(0.2))
 
    model.add(Conv2D(64,(3,3),padding='same',activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
 
    model.add(Conv2D(128,(3,3),padding='same',activation='relu'))
    model.add(Dropout(0.2))
 
    model.add(Conv2D(128,(3,3),padding='same',activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
 
    model.add(Flatten())
    model.add(Dropout(0.2))
    model.add(Dense(512,activation='relu',\
        kernel_constraint=constraints.NonNeg()))
    model.add(Dropout(0.2))
    model.add(Dense(num_classes, activation='softmax',\
        kernel_constraint=constraints.NonNeg()))

    #sgd = SGD(lr = 0.1, decay=1e-6, momentum=0.9 nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
    return model

def load_model_at_epoch(model, epoch):
    ret = []
    fn = f'MODEL_Epoch{epoch}_' 
    fn_end = '.npy'
    weight_names = ['W' + str(i+1) for i in range(9)]
    bias_names = ['b' + str(i+1) for i in range(9)]
    for w_num, b_num in zip(weight_names,bias_names):
        ret.append(np.load(save_folder + fn + w_num + fn_end))
        ret.append(np.load(save_folder + fn + b_num + fn_end))
    model.set_weights(ret)

# saves the model into save_folder directory
def saveModel(model, history, fn):
    model_save_directory = save_folder
    saver.save_model(model,model_save_directory,fn)

    #save score into the same model directionry
    fn = model_save_directory + 'scores.txt'
    my_file = Path(fn)
    try:
        my_abs_path = my_file.resolve(strict=True)
    except FileNotFoundError:
        createScoreFile(fn)
    with open(fn, 'a') as f:
        f.write(f"{history['acc']}\t{history['loss']}\t{history['val_acc']}\t{history['val_loss']}\n")

# helper function to initialize the file for recording the scores
def createScoreFile(fn):
    header =  'acc \t loss \t val_acc \t val_loss\n'
    f = open(fn, 'w+')
    f.write(header)
    f.close()

def saveInitModel(model):
    history = {'acc': 0, 'val_acc': 0, 'loss':0, 'val_loss':0 }
    fn = 'MODEL_Epoch0'
    saveModel(model, history, fn)


    
model = base_model()
model.summary()
#saveModel(model, {'acc': 1, 'val_acc': 2, 'loss':3, 'val_loss':4 },'test_save')
#saveInitModel(model)
for e in range(1, epochs+1):
    print('Epoch',e)
    history = model.fit(x_train, y_train, batch_size=batch_size, epochs=1, validation_data=(x_test,y_test),shuffle=True) 
    print('\nTest accuracy:',history.history['val_acc'],'\n','Train accuracy:',history.history['acc'])
    #saver.save_model(model,model_save_directory,f'MODEL_Epoch{e}')
    saveModel(model, history.history, f'MODEL_Epoch{e}')

    


