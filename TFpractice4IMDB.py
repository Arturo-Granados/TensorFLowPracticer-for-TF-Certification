####################################      Setup      ####################################

#Import the main libraries
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds

print("Version: ", tf.__version__)
print("Eager mode: ", tf.executing_eagerly())
print("Hub version: ", hub.__version__)
print("GPU is", "available" if tf.config.list_physical_devices("GPU") else "NOT AVAILABLE")

####################################      Load the IMDB dataset     ####################################

#Load the IMDB dataset from tensorflow datasets
train_data, validation_data, test_data = tfds.load(
    name='imdb_reviews',
    split=('train[:60%]', 'train[60%:]', 'test'),
    as_supervised=True
)

#Explore the data
train_examples_batch, train_labels_batch = next(iter(train_data.batch(10)))
print(train_examples_batch)

####################################      Build the NN model     ####################################

#Create embedding layer with tensorflow hub
embedding = "https://tfhub.dev/google/nnlm-en-dim50/2"
hub_layer = hub.KerasLayer(embedding, input_shape=[],
                            dtype=tf.string, trainable=True)
#Create the model
model = tf.keras.Sequential()
model.add(hub_layer)
model.add(tf.keras.layers.Dropout(0.3))
model.add(tf.keras.layers.Dense(10, activation='relu'))
model.add(tf.keras.layers.Dropout(0.3))
model.add(tf.keras.layers.Dense(1))
#Model summary
model.summary()
#Compile model
model.compile(
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    optimizer='adam',
    metrics=['accuracy']
)
#Train model
history = model.fit(
    train_data.shuffle(10000).batch(512),
    epochs = 10,
    validation_data = validation_data.batch(512),
    verbose=1
)
####################################      Evaluate the model     ####################################

loss, accuracy = model.evaluate(test_data.batch(512), verbose=2)

print("Loss: ", loss)
print("Accuracy: ", accuracy)

#Binary_accuracy values for train and validation data
binary_accuracy = history.history['accuracy']
val_binary_accuracy = history.history['val_accuracy']

#Loss function values for train and validation data
loss = history.history['loss']
val_loss = history.history['val_loss']

#Find the epochs number
epochs = range(1, len(binary_accuracy) + 1)

#Plot train and validation loss function
# "bo" is for "blue dot"
plt.plot(epochs, loss, 'bo', label='Training loss')
# b is for "solid blue line"
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()


#Plot train and validation loss binary_accuracy
plt.plot(epochs, binary_accuracy, 'bo', label='Training acc')
plt.plot(epochs, val_binary_accuracy, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')

plt.show()


