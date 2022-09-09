####################################      Setup      ####################################
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


####################################      Load data      ####################################

#Load data from keras dataset
fashion_mnist = tf.keras.datasets.fashion_mnist
#Split data into images and targets and train and test sets
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

#Define classes names list
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']



####################################      Preprocess the data      ####################################

#Define input shape
input_shape = train_images.shape[1:]

#Data normalization
train_images = train_images/255.0
test_images = test_images/255.0


####################################      Build NN model      ####################################

#Build model with sequential model keras modul
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape = input_shape),
    tf.keras.layers.Dense(128, activation = 'relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(10, activation = 'softmax')
])
#Check model summary
model.summary()

#Compile model
model.compile(optimizer = 'adam',
              loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics = ['accuracy'])

#Fit model
history = model.fit(train_images, train_labels, epochs = 20, validation_data=(test_images, test_labels))


####################################      evaluation model      ####################################

#Evluate accuracy
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

#MAE values for train and validation data
accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']

#Loss function values for train and validation data
loss = history.history['loss']
val_loss = history.history['val_loss']

# Get number of epochs
epochs   = range(len(accuracy))

#Plot trainig and validation accuracy
plt.plot(epochs, accuracy)
plt.plot(epochs, val_accuracy)
plt.title ('Training and validation accuracy')
plt.show()

#Plot trainin and validation loss function
plt.plot(epochs, loss)
plt.plot(epochs, val_loss)
plt.title ('Training and validation loss function')
plt.show()


####################################      Making predictions      ####################################

#Making prediction with predict method
predictions = model.predict(test_images)
#Apply armax function to get the with maximum probability
print('Predicted value:',  class_names[np.argmax(predictions[0])])
#Print the real value
print('real value:', class_names[test_labels[0]])