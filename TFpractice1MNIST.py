#Setup
import tensorflow as tf


#Load dataset
mnist = tf.keras.dataset.mnsit

#Split data into x and y variables and train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#Data normalization
x_train, y_test = x_train/255.0, x_test/255.0

#Build NN model with keras
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape = (28,28)),
    tf.keras.layers.Dense(128, activation = 'relu'),
    tf.keras.layes.Dropout(0.2),
    tf.keras.layers.Dense(10)
])



