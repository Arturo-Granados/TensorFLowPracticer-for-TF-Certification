#Import tensorflow
import tensorflow as tf

#Load MNIST dataset
mnist = tf.keras.datasets.mnist

#Split data into X and y variables and train and test split
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#Data standarization
x_train, x_test = x_train/255.0, x_test/255.0

#Define input shape
input_shape = x_train.shape[1:]

#Create NN model with keras
model = tf.keras.models.Sequential([
    #Flatten layer convert a matrix into vector
    tf.keras.layers.Flatten(input_shape = input_shape),
    #First hidden layers
    tf.keras.layers.Dense(128, activation = 'relu'),
    #Dropout layers
    tf.keras.layers.Dropout(0.2),
    #Second dense layers
    tf.keras.layers.Dense(10, activation = 'softmax')
])

#Compile model
model.compile(optimizer = 'adam',
              loss = 'sparse_categorical_crossentropy',
              metrics = ['accuracy'])

#Fit model
model.fit(x_train, y_train, epochs = 5)

#Model evaluation
model.evaluate(x_test, y_test, verbose = 2)