import tensorflow as tf
from tensorflow import keras 
import save_keras_model as saver

mnist = tf.keras.datasets.mnist
(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

'''
node_count_list -> list of ints specifying number of nodes per layer
e -> number of epochs
'''
def makeAndRunModel(node_count_list, model_save_directory, e=5):

	print("Nodes by Layer:", node_count_list)
	save_name = "SimpleNN-"+"_".join(map(str, node_count_list))
	model = tf.keras.models.Sequential()
	model.add(keras.layers.Flatten())
	for number_of_units in node_count_list:
		model.add(keras.layers.Dense(number_of_units, activation=tf.nn.relu))
	model.add(keras.layers.Dense(10, activation=tf.nn.softmax))
	model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
	model.fit(x_train, y_train, epochs=e, verbose=0)
	test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
	print('Test accuracy:',test_acc)
	saver.save_model(model,model_save_directory,save_name)
	return test_acc



