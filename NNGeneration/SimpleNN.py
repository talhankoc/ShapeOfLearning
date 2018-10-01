import tensorflow as tf
import numpy as np
mnist = tf.keras.datasets.mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

number_of_hidden = 1
hidden_1_units = 128
model_save_folder = "Saved Models/Digits/" 
#model_name = "digits_model_"+str(number_of_hidden)+"_hidden_"+str(hidden_1_units)

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(hidden_1_units, activation=tf.nn.relu),
  tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)
test_loss, test_acc = model.evaluate(x_test, y_test)
print('Test accuracy:',test_acc)
#model.save_weights(model_save_folder+model_name)

i = 0
for layer in model.layers:
	weights = layer.get_weights() # list of numpy arrays
	if len(weights) == 2:
		i += 1
		W = np.array(weights[0])
		b = np.array(weights[1])
		print("\t****************")
		print("\t* Saving Layer *")
		print("\t* "+str(W.shape)+" *")
		np.save(model_save_folder+"_W"+str(i),W)
		np.save(model_save_folder+"_b"+str(i),b)



