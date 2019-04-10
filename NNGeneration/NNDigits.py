import tensorflow as tf
import NNGeneration.save_keras_model as saver
from tensorflow import keras 
from NNGeneration.nnUtil import randomUpdateModel
'''
Config is a dictionary passed by master.py 

Generates a simple NN defined by layerWidths, and trains
on MNIST for "epochs" epochs. Model is saved at each epoch, 
and accuracies are saved at the end.
'''
def makeAndRun(config):
	epochs = config["epochs"]
	nnSaveFn = config["nnSaveFn"] 
	accSaveFn = config["accSaveFn"]

	mnist = tf.keras.datasets.mnist
	(x_train, y_train),(x_test, y_test) = mnist.load_data()
	x_train, x_test = x_train / 255.0, x_test / 255.0
	
	model = emptyModel(config)

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
def emptyModel(config):
	node_count_list = config["layerWidths"]
	model = keras.models.Sequential()
	model.add(keras.layers.Flatten(input_shape=(28,28)))
	for number_of_units in node_count_list:
		model.add(keras.layers.Dense(number_of_units, activation=tf.nn.relu))
	model.add(keras.layers.Dense(10, activation=tf.nn.softmax))
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
