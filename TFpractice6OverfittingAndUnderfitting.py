####################################      Setup     ####################################
import tensorflow as tf
import tensorflow_docs as tfdocs
import tensorflow_docs.modeling
import tensorflow_docs.plots
from  IPython import display
from matplotlib import pyplot as plt
import numpy as np
import pathlib
import shutil
import tempfile

logdir = pathlib.Path(tempfile.mkdtemp())/"tensorboard_logs"
shutil.rmtree(logdir, ignore_errors=True)
####################################      Get data     ####################################
#The dataset contains 11,000,000 examples, each with 28 features, and a binary class label
gz = tf.keras.utils.get_file('HIGGS.csv.gz', 'http://mlphysics.ics.uci.edu/data/higgs/HIGGS.csv.gz')
#Number of features
FEATURES = 28
#The tf.data.experimental.CsvDataset class can be used to read csv records directly
# from a gzip file with no intermediate decompression step
ds = tf.data.experimental.CsvDataset(gz,[float(),]*(FEATURES+1), compression_type="GZIP")

def pack_row(*row):
  label = row[0]
  features = tf.stack(row[1:],1)
  return features, label

packed_ds = ds.batch(10000).map(pack_row).unbatch()

for features,label in packed_ds.batch(1000).take(1):
  print(features[0])
  plt.hist(features.numpy().flatten(), bins = 101)
  plt.show()

N_VALIDATION = int(1e3)
N_TRAIN = int(1e4)
BUFFER_SIZE = int(1e4)
BATCH_SIZE = 500
STEPS_PER_EPOCH = N_TRAIN//BATCH_SIZE

validate_ds = packed_ds.take(N_VALIDATION).cache()
train_ds = packed_ds.skip(N_VALIDATION).take(N_TRAIN).cache()



validate_ds = validate_ds.batch(BATCH_SIZE)
train_ds = train_ds.shuffle(BUFFER_SIZE).repeat().batch(BATCH_SIZE)

####################################      Demonstrate overfitting     ####################################
#The real challenge is generalization, not fitting
#Training procedure:  Use tf.keras.optimizers.schedules to reduce the learning rate over time

lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
  0.001,
  decay_steps=STEPS_PER_EPOCH*1000,
  decay_rate=1,
  staircase=False)

def get_optimizer():
  return tf.keras.optimizers.Adam(lr_schedule)

def get_callbacks(name):
  return [
    tfdocs.modeling.EpochDots(),
    tf.keras.callbacks.EarlyStopping(monitor='val_binary_crossentropy', patience=200),
    tf.keras.callbacks.TensorBoard(logdir/name),
  ]

def compile_and_fit(model, name, optimizer=None, max_epochs=10000):
  if optimizer is None:
    optimizer = get_optimizer()
  model.compile(optimizer=optimizer,
                loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                metrics=[
                  tf.keras.losses.BinaryCrossentropy(
                      from_logits=True, name='binary_crossentropy'),
                  'accuracy'])

  model.summary()

  history = model.fit(
    train_ds,
    steps_per_epoch = STEPS_PER_EPOCH,
    epochs=max_epochs,
    validation_data=validate_ds,
    callbacks= get_callbacks(name),
    verbose=0)
  return history


####################################      Tiny model     ####################################
tiny_model = tf.keras.Sequential([
  tf.keras.layers.Dense(16, activation = 'elu', input_shape = (FEATURES,)),
  tf.keras.layers.Dense(1)
])

size_histories = {}

size_histories['Tiny'] = compile_and_fit(tiny_model, 'sizes/Tiny')

plotter = tfdocs.plots.HistoryPlotter(metric = 'binary_crossentropy', smoothing_std=10)
plotter.plot(size_histories)
plt.ylim([0.5, 0.7])
plt.show()


####################################      Small model     ####################################

small_model = tf.keras.Sequential([
  tf.keras.layers.Dense(units = 16, activation = 'elu', input_shape = (FEATURES,)),
  tf.keras.layers.Dense(units = 16, activation = 'elu'),
  tf.keras.layers.Dense(units = 1)
])
size_histories['Small'] = compile_and_fit(small_model, 'sizes/Small')


####################################      Medium model     ####################################

medium_model = tf.keras.Sequential([
  tf.keras.layers.Dense(units = 64, activation = 'elu', input_shape = (FEATURES,)),
  tf.keras.layers.Dense(units = 64, activation = 'elu'),
  tf.keras.layers.Dense(units = 64, activation = 'elu'),
  tf.keras.layers.Dense(units = 1)
])
size_histories['Medium'] = compile_and_fit(medium_model, 'sizes/Medium')

####################################      Large model     ####################################

large_model = tf.keras.Sequential([
  tf.keras.layers.Dense(units = 512, activation = 'elu', input_shape = (FEATURES,)),
  tf.keras.layers.Dense(units = 512, activation = 'elu'),
  tf.keras.layers.Dense(units = 512, activation = 'elu'),
  tf.keras.layers.Dense(units = 512, activation = 'elu'),
  tf.keras.layers.Dense(units = 1)
])
size_histories['Large'] = compile_and_fit(large_model, 'sizes/Large')

####################################      Plot the training and validation losses     ####################################

plotter.plot(size_histories)
a = plt.xscale('log')
plt.xlim([5, max(plt.xlim())])
plt.ylim([0.5, 0.7])
plt.xlabel("Epochs [Log Scale]")
plt.show()

####################################      Strategies to prevent overfitting     ####################################

# copy the training logs from the "Tiny" model above, to use as a baseline for comparison.
#Occam's Razor principle is the balance or tradeoff betwen underfitting and overfitting.
shutil.rmtree(logdir/'regularizers/Tiny', ignore_errors=True)
shutil.copytree(logdir/'sizes/Tiny', logdir/'regularizers/Tiny')

regularizer_histories = {}
regularizer_histories['Tiny'] = size_histories['Tiny']

#Add weight regularization

#Model
large_model_l2 = tf.keras.Sequential([
  tf.keras.layers.Dense(units = 512, activation = 'elu', input_shape = (FEATURES,),
                        kernel_regularizer= tf.keras.regularizers.l2(0.001)),
  tf.keras.layers.Dense(units = 512, activation = 'elu',kernel_regularizer= tf.keras.regularizers.l2(0.001)),
  tf.keras.layers.Dense(units = 512, activation = 'elu',kernel_regularizer= tf.keras.regularizers.l2(0.001)),
  tf.keras.layers.Dense(units = 512, activation = 'elu',kernel_regularizer= tf.keras.regularizers.l2(0.001)),
  tf.keras.layers.Dense(units = 1)
])
#Compile and train model
regularizer_histories['l2'] = compile_and_fit(large_model_l2, "regularizers/l2")
#Plot the model performance
plotter.plot(regularizer_histories)
plt.ylim([0.5, 0.7])
plt.show()

result = large_model_l2(features)
regularization_loss = tf.add_n(large_model_l2.losses)

#Add dropout

#Model
large_model_dropout = tf.keras.Sequential([
  tf.keras.layers.Dense(units = 512, activation = 'elu', input_shape = (FEATURES,)),
  tf.keras.layers.Dropout(0.5),
  tf.keras.layers.Dense(units = 512, activation = 'elu'),
  tf.keras.layers.Dropout(0.5),
  tf.keras.layers.Dense(units = 512, activation = 'elu'),
  tf.keras.layers.Dropout(0.5),
  tf.keras.layers.Dense(units = 512, activation = 'elu'),
  tf.keras.layers.Dropout(0.5),
  tf.keras.layers.Dense(units = 1)
])
#Compile and train the model
regularizer_histories['dropout'] = compile_and_fit(large_model_dropout, "regularizers/dropout")
#Plot the model performance
plotter.plot(regularizer_histories)
plt.ylim([0.5, 0.7])
plt.show()

#Combined L2 + dropout
large_model_combined = tf.keras.Sequential([
  tf.keras.layers.Dense(units = 512, activation = 'elu', input_shape = (FEATURES,),
                        kernel_regularizer= tf.keras.regularizers.l2(0.0001)),
  tf.keras.layers.Dropout(0.5),
  tf.keras.layers.Dense(units = 512, activation = 'elu',
                        kernel_regularizer= tf.keras.regularizers.l2(0.0001)),
  tf.keras.layers.Dropout(0.5),
  tf.keras.layers.Dense(units = 512, activation = 'elu',
                        kernel_regularizer= tf.keras.regularizers.l2(0.0001)),
  tf.keras.layers.Dropout(0.5),
  tf.keras.layers.Dense(units = 512, activation = 'elu',
                        kernel_regularizer= tf.keras.regularizers.l2(0.0001)),
  tf.keras.layers.Dropout(0.5),
  tf.keras.layers.Dense(units = 1)
])

#Compile and train the model
regularizer_histories['combined'] = compile_and_fit(large_model_combined, "regularizers/combined")
#Plot the model performance
plotter.plot(regularizer_histories)
plt.ylim([0.5, 0.7])
plt.show()
