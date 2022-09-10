####################################      Setup      ####################################

#Import importand libraries
import tensorflow as tf
import os
import re
import shutil
import string
import matplotlib.pyplot as plt

####################################      Download dataset      ####################################

#Dataset URL
url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
#Get file with keras
dataset = tf.keras.utils.get_file('aclImdb_v1', url,
                                   untar = True, cache_dir = '.',
                                   cache_subdir = '')
#Define dataset directory
dataset_dir = os.path.join(os.path.dirname(dataset), 'aclImdb')

#Define train set directory
train_dir = os.path.join(dataset_dir,'train')


####################################      Load dataset      ####################################

#The IMDB dataset contains additional folders that must be removed(beacause are inese
#Define directory to bo removed
remove_dir = os.path.join(train_dir, 'unsup')
#Remove directory
shutil.rmtree(remove_dir)

#Load the data off disk and prepare it into a format suitable for training.
#To do so, use the helpful text_dataset_from_directory utility.

#Define batch size and random seed
batch_size = 32
seed = 42

#Create  labeled sets with tf.data.dataset
#Train set
raw_train_ds = tf.keras.utils.text_dataset_from_directory(
    'aclImdb/train',
    batch_size = batch_size,
    validation_split = 0.2,
    subset = 'training',
    seed = seed
)
#Validation set
raw_val_ds = tf.keras.utils.text_dataset_from_directory(
    'aclImdb/train',
    batch_size = batch_size,
    validation_split=0.2,
    subset = 'validation',
    seed = seed
)
#Test set
raw_test_ds = tf.keras.utils.text_dataset_from_directory(
    'aclImdb/test',
    batch_size = batch_size,
)


####################################      Preprocessing data      ####################################

#Standarization(remove html tags)
def custom_standardization(input_data):
  lowercase = tf.strings.lower(input_data)
  stripped_html = tf.strings.regex_replace(lowercase, '<br />', ' ')
  return tf.strings.regex_replace(stripped_html,
                                  '[%s]' % re.escape(string.punctuation),
                                  '')

#TextVectorization layer

#Define max features and sequence length
max_features = 10000
sequence_length = 250

vectorize_layer = tf.keras.layers.TextVectorization(
    standardize = custom_standardization,
    max_tokens = max_features,
    output_mode = 'int',
    output_sequence_length = sequence_length
)

#Make a text-only dataset (without labels), then call adapt
train_text = raw_train_ds.map(lambda x, y: x)
vectorize_layer.adapt(train_text)

#Function to use the vectorize layer to preprocess the data
def vectorize_text(text, label):
  text = tf.expand_dims(text, -1)
  return vectorize_layer(text), label

# Apply the TextVectorization layer  created earlier to the train, validation, and test dataset.
train_ds = raw_train_ds.map(vectorize_text)
val_ds = raw_val_ds.map(vectorize_text)
test_ds = raw_test_ds.map(vectorize_text)


####################################      Dataset configuration      ####################################

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().prefetch(buffer_size= AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size= AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size= AUTOTUNE)


####################################      Create NN model      ####################################

#Define embedding dimention
embedding_dim = 16

#Define model
model = tf.keras.models.Sequential([
    tf.keras.layers.Embedding(max_features+1, embedding_dim),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1)
])

#Compile model
model.compile(loss = tf.keras.losses.BinaryCrossentropy(from_logits=True),
              optimizer = 'adam',
              metrics = tf.metrics.BinaryAccuracy(threshold=0.0) )
model.summary()
#Fit model
epochs = 10
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs)

####################################       Model evaluation      ####################################

loss, accuracy = model.evaluate(test_ds)

print("Loss: ", loss)
print("Accuracy: ", accuracy)

#Binary_accuracy values for train and validation data
binary_accuracy = history.history['binary_accuracy']
val_binary_accuracy = history.history['val_binary_accuracy']

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


####################################       Export the model      ####################################
export_model = tf.keras.Sequential([
  vectorize_layer,
  model,
  tf.keras.layers.Activation('sigmoid')
])

export_model.compile(
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=False), optimizer="adam", metrics=['accuracy']
)

# Test it with `raw_test_ds`, which yields raw strings
loss, accuracy = export_model.evaluate(raw_test_ds)
print(accuracy)

#Inference on new data
examples = [
  "The movie was great!",
  "The movie was okay.",
  "The movie was terrible..."
]

export_model.predict(examples)