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
if K.backend()=='tensorflow':
    K.set_image_dim_ordering("th")
 
# Import Tensorflow with multiprocessing
import tensorflow as tf
import multiprocessing as mp
 
# Loading the CIFAR-10 datasets
from keras.datasets import cifar10

batch_size = 32 
# 32 examples in a mini-batch, smaller batch size means more updates in one epoch
num_classes = 10 #
epochs = 100 # repeat 100 times

(x_train, y_train), (x_test, y_test) = cifar10.load_data() 
# x_train - training data(images), y_train - labels(digits)

'''
fig = plt.figure(figsize=(8,3))
for i in range(num_classes):
    ax = fig.add_subplot(2, 5, 1 + i, xticks=[], yticks=[])
    idx = np.where(y_train[:]==i)[0]
    features_idx = x_train[idx,::]
    img_num = np.random.randint(features_idx.shape[0])
    im = np.transpose(features_idx[img_num,::],(1,2,0))
    ax.set_title(class_names[i])
    plt.imshow(im)
plt.show()
'''
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
    model.add(Dense(1024,activation='relu',kernel_constraint=maxnorm(3)))
    model.add(Dropout(0.2))
    model.add(Dense(num_classes, activation='softmax'))

    #sgd = SGD(lr = 0.1, decay=1e-6, momentum=0.9 nesterov=True)
 
    model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
    return model

def load_model_at_epoch(model, epoch):
    ret = []
    folder = 'Saved Models/CIFAR-10/'
    fn = 'MODEL_Epoch' 
    fn += str(epoch) + '_'
    fn_end = '.npy'
    weight_names = ['W' + str(i+1) for i in range(8)]
    bias_names = ['b' + str(i+1) for i in range(8)]
    for w_num, b_num in zip(weight_names,bias_names):
        ret.append(np.load(folder + fn + w_num + fn_end))
        ret.append(np.load(folder + fn + b_num + fn_end))
    model.set_weights(ret)

model = base_model()
# comment out next two lines to start from fresh model
#starting_epoch = 65
#load_model_at_epoch(model, starting_epoch)
model.summary()
model_save_directory = 'Saved Models/CIFAR-10/'
scores = []

###round 0 
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=1)
train_loss, train_acc = model.evaluate(x_train, y_train, verbose=1)
print('\nTest accuracy:',test_acc,'\n','Train accuracy:',train_acc)
scores.append((test_acc,train_acc))
print('Saving model to', model_save_directory)
saver.save_model(model,model_save_directory,'MODEL_Epoch0')
for e in range(epochs):
    res = model.fit(x_train, y_train, batch_size=batch_size, epochs=1, validation_data=(x_test,y_test),shuffle=True)
    test_acc = res.history['val_acc']
    train_acc = res.history['acc']
    print('\nTest accuracy:',test_acc,'\n','Train accuracy:',train_acc)
    scores.append((test_acc,train_acc))
    print('Saving model to', model_save_directory)
    saver.save_model(model,model_save_directory,'MODEL_Epoch'+str(e+1))

    



#Plotting testing and training
# plt.figure(0)
# plt.plot(cnn.history['acc'],'r')
# plt.plot(cnn.history['val_acc'],'g')
# plt.xticks(np.arange(0, 101, 2.0))
# plt.rcParams['figure.figsize'] = (8, 6)
# plt.xlabel("Num of Epochs")
# plt.ylabel("Accuracy")
# plt.title("Training Accuracy vs Validation Accuracy")
# plt.legend(['train','validation'])
  
# plt.figure(1)
# plt.plot(cnn.history['loss'],'r')
# plt.plot(cnn.history['val_loss'],'g')
# plt.xticks(np.arange(0, 101, 2.0))
# plt.rcParams['figure.figsize'] = (8, 6)
# plt.xlabel("Num of Epochs")
# plt.ylabel("Loss")
# plt.title("Training Loss vs Validation Loss")
# plt.legend(['train','validation'])

# plt.show()