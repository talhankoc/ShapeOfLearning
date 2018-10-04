import tensorflow as tf
from tensorflow import keras 
import save_keras_model as saver

fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
train_images = train_images / 255.0
test_images = test_images / 255.0

#Not used yet
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

def makeAndRunModel(node_count_list, model_save_directory, e=5):

	print("Nodes by Layer:", node_count_list)
	save_name = "NN-Fashion-"+"_".join(map(str, node_count_list))
	model = tf.keras.models.Sequential()
	model.add(keras.layers.Flatten(input_shape=(28, 28)))
	for number_of_units in node_count_list:
		model.add(keras.layers.Dense(number_of_units, activation=tf.nn.relu))
	model.add(keras.layers.Dense(10, activation=tf.nn.softmax))
	model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
	model.fit(train_images, train_labels, epochs=e)
	test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=0)
	print('Test accuracy:',test_acc)
	saver.save_model(model,model_save_directory,save_name)
	return test_acc
