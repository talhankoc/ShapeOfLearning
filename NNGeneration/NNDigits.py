import tensorflow as tf
import NNGeneration.save_keras_model as saver
import numpy as np
from tensorflow import keras 
from NNGeneration.nnUtil import randomUpdateModel
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Activation
from keras import regularizers
from pathlib import Path

'''
Config is a dictionary passed by master.py 

Generates a simple NN defined by layerWidths, and trains
on MNIST for "epochs" epochs. Model is saved at each epoch, 
and accuracies are saved at the end.
'''
def makeAndRun(config):
	batch_size = 64
	from keras.datasets import cifar10 as dataset
	#from keras.datasets import mnist as dataset
	#from keras.datasets import fashion_mnist as dataset
	(x_train, y_train),(x_test, y_test) = dataset.load_data()
	y_train = np_utils.to_categorical(y_train, 10)
	y_test = np_utils.to_categorical(y_test, 10)
	x_train = x_train.astype('float32')
	x_test = x_test.astype('float32')
	x_train, x_test = x_train / 255.0, x_test / 255.0
	x_train = np.array(x_train)
	x_test = np.array(x_test)
	if 'cifar10' in str(dataset):
		rgb2gray = lambda imgs: 0.2989 * imgs[:,:,:,0] + 0.5870 * imgs[:,:,:,1] + 0.1140 * imgs[:,:,:,2]
		x_train = rgb2gray(x_train)
		x_test = rgb2gray(x_test)
	
	model = baseModel(config)
	model.summary()
	train_accuracies = []
	train_losses = []
	test_accuracies = []
	test_losses = []
	for epoch in epochs:
		model.fit(x_train,y_train,verbose=1)
		train_loss, train_acc = model.evaluate(x_train, y_train, verbose=0)
		test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
		train_losses.append(train_loss)
		train_accuracies.append(train_acc)
		test_losses.append(test_loss)
		test_accuracies.append(test_acc)
		print(f"Finished update {epoch}. Train: {train_acc}, Test:{test_acc}")
		saver.save_model(model,nnSaveFn(epoch),"")

	saveAccuracies(accSaveFn(),train_accuracies,test_accuracies,train_losses,test_losses)
return 

'''
Returns an untrained model without any initialized parameters. 
'''
def baseModel(config):
	node_count_list = config["layerWidths"]
	model = Sequential()
	model.add(Flatten(input_shape=(32,32)))
	model.add(Dropout(0.2))
	for number_of_units in node_count_list:
		model.add(Dense(number_of_units, activation=tf.nn.relu))
	model.add(Dense(10, activation=tf.nn.softmax))
	model.compile(optimizer='adam',
	          loss='categorical_crossentropy',
	          metrics=['accuracy'])
	return model

'''
Trains a model that does random updates.
'''
def randomModel(config):
	model = randomUpdateModel(emptyModel(config))
	epochs = config["epochs"]
	nnSaveFn = config["nnSaveFn"] 
	accSaveFn = config["accSaveFn"]

	mnist = tf.keras.datasets.mnist
	(x_train, y_train),(x_test, y_test) = mnist.load_data()
	x_train, x_test = x_train / 255.0, x_test / 255.0

	train_accuracies = []
	train_losses = []
	test_accuracies = []
	test_losses = []
	for epoch in epochs:
		model = randomUpdateModel(model)
		train_loss, train_acc = model.evaluate(x_train, y_train, verbose=0)
		test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
		train_losses.append(train_loss)
		train_accuracies.append(train_acc)
		test_losses.append(test_loss)
		test_accuracies.append(test_acc)
		print(f"Finished update {epoch}. Train: {train_acc}, Test:{test_acc}")
		saver.save_model(model,nnSaveFn(epoch),"")

	saveAccuracies(accSaveFn(),train_accuracies,test_accuracies,train_losses,test_losses)
	return

def saveAccuracies(path,train_acc,test_acc,train_losses,test_losses):
	with open(path,"w") as f:
		f.write("acc \t loss \t val_acc \t val_loss\n")
		for epoch in range(len(test_acc)):
			train, test = train_acc[epoch], test_acc[epoch]
			train_loss, test_loss = train_losses[epoch], test_losses[epoch]
			f.write(f"{train}\t{train_loss}\t{test}\t{test_loss}\n")
	return 

def load_model_at_epoch(model, epoch, config):
	ret = []
	fn = config['nnSaveFn'](epoch) 
	fn_end = '.npy'
	weight_names = ['W' + str(i+1) for i in range(4)]
	bias_names = ['b' + str(i+1) for i in range(4)]
	for w_num, b_num in zip(weight_names,bias_names):
		ret.append(np.load(save_folder + fn + w_num + fn_end))
		ret.append(np.load(save_folder + fn + b_num + fn_end))
	model.set_weights(ret)

# saves the model into save_folder directory
def saveModel(model, epoch, config, history):
	saver.save_model(model,config['nnSaveFn'](epoch))
	score_fn = config['accSaveFn']()
	my_file = Path(score_fn)
	try:
		my_abs_path = my_file.resolve(strict=True)
	except FileNotFoundError:
		createScoreFile(score_fn)
	with open(score_fn, 'a') as f:
		f.write(f"{history['acc'][0]}\t{history['loss'][0]}\t{history['val_acc'][0]}\t{history['val_loss'][0]}\n")

# helper function to initialize the file for recording the scores
def createScoreFile(fn):
	header =  'acc \t loss \t val_acc \t val_loss\n'
	f = open(fn, 'w+')
	f.write(header)
	f.close()

def saveInitModel(model, config):
	history = {'acc': [0.0], 'val_acc': [0.0], 'loss':[0.0], 'val_loss':[0.0] }
	saveModel(model, history, config, 0)