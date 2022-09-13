####################################      Setup     ####################################

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import sklearn
# Make NumPy printouts easier to read.
np.set_printoptions(precision=3, suppress=True)

####################################      Load dataset     ####################################

url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'
column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight',
                'Acceleration', 'Model Year', 'Origin']

raw_dataset = pd.read_csv(url, names=column_names,
                          na_values='?', comment='\t',
                          sep=' ', skipinitialspace=True)

dataset = raw_dataset.copy()


####################################      Preprocessing data     ####################################

#Drop rows with missing values
dataset = dataset.dropna()

#Encode Origin column
dataset['Origin'] = dataset['Origin'].map({1: 'USA', 2: 'Europe', 3: 'Japan'})


dataset = pd.get_dummies(dataset, columns=['Origin'], prefix='', prefix_sep='')


#Split the data into training and test sets
from sklearn.model_selection import train_test_split
train_dataset, test_dataset = train_test_split(dataset, test_size= 0.2, random_state= 42)


###########################################      Split features from labels     ####################################
#Input variables
train_features = train_dataset.copy()
test_features = test_dataset.copy()

#Output variables
train_labels = train_features.pop('MPG')
test_labels = test_features.pop('MPG')

####################################      Normalization     ####################################

#Normalization
def normalization(data):
    normalizer = tf.keras.layers.Normalization(axis=-1)
    normalizer.adapt(data)
    return normalizer

####################################      Built models     ####################################

#simple linear regression model
def linear_regression(normalizer):
    model = tf.keras.Sequential([
        normalizer,
        tf.keras.layers.Dense(units = 1)
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.1),
        loss='mean_absolute_error'
    )
    return model

#Dnn model
def dnn_model(normalizer):
    model = tf.keras.Sequential([
        normalizer,
        tf.keras.layers.Dense(units = 64, activation ='relu'),
        tf.keras.layers.Dense(units=64, activation='relu'),
        tf.keras.layers.Dense(units=1)
    ])
    model.compile(loss='mean_absolute_error',
                  optimizer=tf.keras.optimizers.Adam(0.001))
    return model

#Plot loss function
def plot_loss(history):
  plt.plot(history.history['loss'], label='loss')
  plt.plot(history.history['val_loss'], label='val_loss')
  plt.ylim([0, 10])
  plt.xlabel('Epoch')
  plt.ylabel('Error [MPG]')
  plt.legend()
  plt.grid(True)
  plt.show()

####################################      Train models     ####################################

#Linear regression
normalizer = normalization(np.array(train_features))
linear_regression = linear_regression(normalizer)
#print(linear_regression.summary())

#Train Linear regression model
history = linear_regression.fit(
    train_features,
    train_labels,
    epochs=100,
    verbose=0,
    validation_split = 0.2)

#plot Linear regression loss function values
plot_loss(history)


#DNN model
normalizer = normalization(np.array(train_features))
dnn_model = dnn_model(normalizer)
print(dnn_model.summary())
#Train DNN model
history = dnn_model.fit(
    train_features,
    train_labels,
    validation_split=0.2,
    verbose=0,
    epochs=100)
#plot DNN model loss function values
plot_loss(history)


####################################      Test models     ####################################

test_results = {}
test_results['linear_model'] = linear_regression.evaluate(
    test_features, test_labels, verbose=0)
test_results['dnn_model'] = dnn_model.evaluate(
    test_features, test_labels, verbose=0)

print(pd.DataFrame(test_results, index=['Mean absolute error [MPG]']).T)


####################################      Prediction and error analysis     ####################################
test_predictions = dnn_model.predict(test_features).flatten()

a = plt.axes(aspect='equal')
plt.scatter(test_labels, test_predictions)
plt.xlabel('True Values [MPG]')
plt.ylabel('Predictions [MPG]')
lims = [0, 50]
plt.xlim(lims)
plt.ylim(lims)
_ = plt.plot(lims, lims)
plt.show()


error = test_predictions - test_labels
plt.hist(error, bins=25)
plt.xlabel('Prediction Error [MPG]')
_ = plt.ylabel('Count')
plt.show()