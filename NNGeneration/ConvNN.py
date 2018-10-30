import tensorflow as tf
from tensorflow import keras 
import save_keras_model as saver




''' Not used yet 
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
'''
mnist = None
def feedFashionDataset():
	global mnist
	mnist = tf.keras.datasets.fashion_mnist

def feedDigitsDataset():
	global mnist
	mnist = keras.datasets.mnist

def makeAndRunModel(node_count_list, model_save_directory, e=5):
	global mnist
	#assert mnist not None, 'Data has not been fed. Call either feedFashionDataset() or feedDigitDataset().'
	(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
	train_images, test_images = train_images.astype('float32') / 255, test_images.astype('float32') / 255
	print("Nodes by Layer:", node_count_list)
	save_name = "NN-Fashion-"+"_".join(map(str, node_count_list))

	model = tf.keras.Sequential()
	# Must define the input shape in the first layer of the neural network
	model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=2, padding='same', activation='relu', input_shape=(28,28,1))) 
	model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
	model.add(tf.keras.layers.Dropout(0.3))
	model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=2, padding='same', activation='relu'))
	model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
	model.add(tf.keras.layers.Dropout(0.3))
	model.add(tf.keras.layers.Flatten())
	for number_of_units in node_count_list:
		model.add(tf.keras.layers.Dense(number_of_units, activation='relu'))
		model.add(tf.keras.layers.Dropout(0.5))
	model.add(tf.keras.layers.Dense(10, activation='softmax'))
	# Take a look at the model summary
	model.summary()
	###

	model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
	model.fit(train_images, train_labels, epochs=e)
	test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=1)
	train_loss, train_acc = model.evaluate(train_images, train_labels, verbose=1)
	print('\nTest accuracy:',test_acc)
	print('Train accuracy:',train_acc)
	
	if model_save_directory is not None:
		print('Saving model to', model_save_directory)
		saver.save_model(model,model_save_directory,save_name)
	else:
		print('Not saving model.')
	return test_acc, train_acc
