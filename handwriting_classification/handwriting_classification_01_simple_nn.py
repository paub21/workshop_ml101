from tensorflow.keras.datasets import mnist

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

print(train_images.shape)

train_images = train_images.reshape((60000, 28*28))
train_images = train_images.astype('float32')/255

test_images = test_images.reshape((10000, 28*28))
test_images = test_images.astype('float32')/255

# import matplotlib.pyplot as plt
# digit = train_images[510]
# plt.imshow(digit, cmap=plt.cm.binary)
# plt.show()

from tensorflow.keras import models
from tensorflow.keras import layers

network = models.Sequential()
network.add(layers.Dense(512, activation='relu', input_shape=(28*28,)))
network.add(layers.Dense(128, activation='relu'))
network.add(layers.Dense(10, activation = 'softmax'))

network.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

#network.summary()

print(train_images.shape)

from tensorflow.keras.utils import to_categorical

train_labels_encoded = to_categorical(train_labels)
test_labels_encoded = to_categorical(test_labels)

network.fit(train_images, train_labels_encoded, epochs=5, batch_size=128, verbose=1)

test_loss, test_acc = network.evaluate(test_images, test_labels_encoded)
print('test_acc:', test_acc)

print('size of test images: {}'.format(test_images.shape))
print(network.predict_classes(test_images))
print(test_labels)

# result = network.predict(test_images[139:140])
# print('predicted number:', result.argmax())
# print('actual number:', test_labels[139:140])


# Plot confusion matrix

from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

labels = ['0','1','2','3','4','5','6','7','8','9']

cm = confusion_matrix(test_labels, network.predict_classes(test_images))

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)

disp.plot(cmap=plt.cm.Blues)
plt.show() 

