#Import libraries
import tensorflow as tf
import matplotlib.pyplot as plt
import keras
from tensorflow.keras import datasets
#Import data
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

#show a data example
#uncoment to see the images
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']
'''
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i])
    # The CIFAR labels happen to be arrays,
    # which is why you need the extra index
    plt.xlabel(class_names[train_labels[i][0]])
plt.show()'''

#Create a convolutional base model

model = keras.models.Sequential([
    keras.layers.Conv2D(32, (3,3), activation ='relu', input_shape = (32,32,3)),
    keras.layers.MaxPooling2D((2,2)),
    keras.layers.Dropout(0.4),
    keras.layers.Conv2D(64, (3,3), activation ='relu'),
    keras.layers.MaxPooling2D((2,2)),
    keras.layers.Dropout(0.4),
    keras.layers.Conv2D(64, (3,3), activation ='relu'),
    keras.layers.Flatten(),
    keras.layers.Dense(64, activation = 'relu'),
    keras.layers.Dense(10)

])


#print(model.summary())

#Compile the model
model.compile(
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer = tf.keras.optimizers.Adam(),
    metrics = ['accuracy']
)

#Train the model
history = model.fit(train_images, train_labels, epochs= 10, validation_data=(test_images, test_labels))


#Evaluation model
#plot metrics
plt.plot(history.history['accuracy'], label = 'accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.show()

#Evaluation
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

print('accuracy ', test_acc)