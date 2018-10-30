import tensorflow as tf
from tensorflow import keras 
import save_keras_model as saver
from keras import backend as K





''' Not used yet 
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
'''
img_rows, img_cols = 28, 28
mnist = None
def feedFashionDataset():
	global mnist
	mnist = tf.keras.datasets.fashion_mnist

def feedDigitsDataset():
	global mnist
	mnist = keras.datasets.mnist

def makeAndRunModel(node_count_list, model_save_directory, e=5, divider=1, model_type=None):
	global mnist
	#assert mnist not None, 'Data has not been fed. Call either feedFashionDataset() or feedDigitDataset().'
	(train_images, train_labels), (test_images, test_labels) = prepareDataSet(divider, model_type)
	print("Nodes by Layer:", node_count_list)
	save_name = "NN-Fashion-"+"_".join(map(str, node_count_list))
	model = makeCNN(node_count_list) if model_type == 'cnn' else makeSimpleNN(node_count_list)
	model.summary()
	scores = []
	for i in range(e):
		model.fit(train_images, train_labels, epochs=1)
		test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=0)
		train_loss, train_acc = model.evaluate(train_images, train_labels, verbose=0)
		print('\nTest accuracy:',test_acc)
		print('Train accuracy:',train_acc)
		scores.append((test_acc,train_acc))
		if model_save_directory is not None:
			print('Saving model to', model_save_directory)
			saver.save_model(model,model_save_directory,save_name+'__Epoch'+str(i+1))

	print(scores)
	return scores

def prepareDataSet(divider, model_type):
	(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
	train_images, test_images = train_images / 255.0, test_images / 255.0
	train_images, train_labels = train_images[:int(len(train_images)/divider)], train_labels[:int(len(train_labels)/divider)]
	if model_type == 'cnn':
		if K.image_data_format() == 'channels_first':
			train_images = train_images.reshape(train_images.shape[0], 1, img_rows, img_cols)
			test_images = test_images.reshape(test_images.shape[0], 1, img_rows, img_cols)
			input_shape = (1, img_rows, img_cols)
		else:
			train_images = train_images.reshape(train_images.shape[0], img_rows, img_cols, 1)
			test_images = test_images.reshape(test_images.shape[0], img_rows, img_cols, 1)
			input_shape = (img_rows, img_cols, 1)
	return (train_images, train_labels), (test_images, test_labels)

def makeSimpleNN(node_count_list):
	model = tf.keras.models.Sequential()
	model.add(keras.layers.Flatten(input_shape=(28, 28)))
	for number_of_units in node_count_list:
		model.add(keras.layers.Dense(number_of_units, activation=tf.nn.relu))
	model.add(keras.layers.Dense(10, activation=tf.nn.softmax))
	model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
	return model
def makeCNN(node_count_list):
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
	model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
	return model

############
parameters = [8,16,24,32,40,48,56,64]
parameters = parameters[::-1]
model_save_directory = 'Saved Models/Fashion - Each Epoch/'
feedFashionDataset()
for p in parameters:
	scores_list = makeAndRunModel([p],model_save_directory,e=50,divider=10)
	f = open(model_save_directory+'HiddenLayerNodeCount'+str(p)+"_Scores.txt","w+")
	count = 1
	f.write('Epoch \tTest Acc. \tTrain Acc.')
	for s in scores_list:
		f.write('\n'+str(count)+':\t'+str(s[0])+'\t'+str(s[1]))
		count += 1
	f.close()

