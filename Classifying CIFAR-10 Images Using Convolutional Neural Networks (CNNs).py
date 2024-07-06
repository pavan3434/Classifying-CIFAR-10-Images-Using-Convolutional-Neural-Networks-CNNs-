#!/usr/bin/env python
# coding: utf-8

# In[1]:


from keras import datasets, layers, models
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, Flatten
import matplotlib.pyplot as plt
import numpy as np

(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# Show the shape of (train_images, train_labels), (test_images, test_labels)
print("Shape of training images:", train_images.shape)
print("Shape of training labels:", train_labels.shape)
print("Shape of testing images:", test_images.shape)
print("Shape of testing labels:", test_labels.shape)

# Creating a list of all the class labels
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Visualizing some of the images from the training dataset
plt.figure(figsize=[10,10])
for i in range(10):    # for first 10 images
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i][0]])
plt.show()

# Converting the pixels data to float type
train_images = train_images.astype('float32')
test_images = test_images.astype('float32')

# Normalize input
train_images = train_images / 255
test_images = test_images / 255

# Change target class to one hot encoding
num_classes = 10
train_labels = to_categorical(train_labels, num_classes)
test_labels = to_categorical(test_labels, num_classes)

# Creating a sequential model and adding layers to it
model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(32, 32, 3)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.3))
model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.3))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

# Checking the model summary
model.summary()

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(train_images, train_labels, batch_size=64, epochs=20,
                    validation_data=(test_images, test_labels))

# Loss curve
plt.figure(figsize=[6,4])
plt.plot(history.history['loss'], 'black', linewidth=2.0)
plt.plot(history.history['val_loss'], 'green', linewidth=2.0)
plt.legend(['Training Loss', 'Validation Loss'], fontsize=14)
plt.xlabel('Epochs', fontsize=10)
plt.ylabel('Loss', fontsize=10)
plt.title('Loss Curves', fontsize=12)
plt.show()

# Accuracy of test_images
test_loss, test_accuracy = model.evaluate(test_images, test_labels)
print("Test Accuracy:", test_accuracy)

# Making the Predictions
pred = model.predict(test_images)

# Converting the predictions into label index
pred_classes = np.argmax(pred, axis=1)
print(pred_classes)

# Plotting the Actual vs. Predicted results
fig, axes = plt.subplots(5, 5, figsize=(15,15))
axes = axes.ravel()

for i in np.arange(0, 25):
    axes[i].imshow(test_images[i])
    axes[i].set_title("True: %s \nPredict: %s" % (class_names[np.argmax(test_labels[i])], class_names[pred_classes[i]]))
    axes[i].axis('off')
plt.subplots_adjust(wspace=1)
plt.show()


# In[ ]:


extra


# In[4]:


get_ipython().system('pip install tensorflow-datasets')


# In[ ]:


11


# In[7]:


from keras import datasets, layers, models
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.applications import VGG16
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import numpy as np

(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

print("Shape of train images:", train_images.shape)
print("Shape of train labels:", train_labels.shape)
print("Shape of test images:", test_images.shape)
print("Shape of test labels:", test_labels.shape)

X_train = train_images
X_test = test_images
Y_train = to_categorical(train_labels)
Y_test = to_categorical(test_labels)

print(X_train.shape)
print(X_test.shape)

# display i-th image
idx = 3
image = X_train[idx]
label = Y_train[idx]
plt.figure()
plt.imshow(image, cmap=plt.cm.binary)
plt.title('Label {}'.format(np.argmax(label)))
plt.colorbar()
plt.grid(False)
plt.show()

# Loading VGG16 model
base_model = VGG16(weights="imagenet", include_top=False, input_shape=X_train[0].shape)
base_model.trainable = False  # Not trainable weights

base_model.summary()

flatten_layer = Flatten()
dense_layer_1 = Dense(50, activation='relu')
dense_layer_2 = Dense(20, activation='relu')
prediction_layer = Dense(10, activation='softmax')

model = models.Sequential([
    base_model,
    flatten_layer,
    dense_layer_1,
    dense_layer_2,
    prediction_layer
])

# Making the Predictions

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy'],
)

es = EarlyStopping(monitor='val_accuracy', mode='max', patience=5, restore_best_weights=True)

model.fit(X_train, Y_train, epochs=10, validation_split=0.2, batch_size=32, callbacks=[es])

# Making the Predictions

accuracy = model.evaluate(X_test, Y_test, verbose=0)
print("Accuracy on test data:", accuracy[1])


# In[ ]:


DENSE LAYER


# In[8]:


from keras import datasets, layers, models
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.applications import VGG16
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import numpy as np

(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

print("Shape of train images:", train_images.shape)
print("Shape of train labels:", train_labels.shape)
print("Shape of test images:", test_images.shape)
print("Shape of test labels:", test_labels.shape)

X_train = train_images
X_test = test_images
Y_train = to_categorical(train_labels)
Y_test = to_categorical(test_labels)

print(X_train.shape)
print(X_test.shape)

# display i-th image
idx = 3
image = X_train[idx]
label = Y_train[idx]
plt.figure()
plt.imshow(image, cmap=plt.cm.binary)
plt.title('Label {}'.format(np.argmax(label)))
plt.colorbar()
plt.grid(False)
plt.show()

# Loading VGG16 model
base_model = VGG16(weights="imagenet", include_top=False, input_shape=X_train[0].shape)
base_model.trainable = False  # Not trainable weights

base_model.summary()

flatten_layer = Flatten()
dense_layer_1 = Dense(50, activation='relu')
dense_layer_2 = Dense(20, activation='relu')
dense_layer_3 = Dense(10, activation='relu')  # Additional dense layer
prediction_layer = Dense(10, activation='softmax')

model = models.Sequential([
    base_model,
    flatten_layer,
    dense_layer_1,
    dense_layer_2,
    dense_layer_3,  # Add the additional dense layer here
    prediction_layer
])

# Making the Predictions

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy'],
)

es = EarlyStopping(monitor='val_accuracy', mode='max', patience=5, restore_best_weights=True)

model.fit(X_train, Y_train, epochs=10, validation_split=0.2, batch_size=32, callbacks=[es])

# Making the Predictions

accuracy = model.evaluate(X_test, Y_test, verbose=0)
print("Accuracy on test data:", accuracy[1])


# In[ ]:


BATCHNORMALIZATION


# In[9]:


from keras import datasets, layers, models
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.applications import VGG16
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import numpy as np

(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

print("Shape of train images:", train_images.shape)
print("Shape of train labels:", train_labels.shape)
print("Shape of test images:", test_images.shape)
print("Shape of test labels:", test_labels.shape)

X_train = train_images
X_test = test_images
Y_train = to_categorical(train_labels)
Y_test = to_categorical(test_labels)

print(X_train.shape)
print(X_test.shape)

# display i-th image
idx = 3
image = X_train[idx]
label = Y_train[idx]
plt.figure()
plt.imshow(image, cmap=plt.cm.binary)
plt.title('Label {}'.format(np.argmax(label)))
plt.colorbar()
plt.grid(False)
plt.show()

# Loading VGG16 model
base_model = VGG16(weights="imagenet", include_top=False, input_shape=X_train[0].shape)
base_model.trainable = False  # Not trainable weights

base_model.summary()

flatten_layer = Flatten()
dense_layer_1 = Dense(50, activation='relu')
batchnorm_layer_1 = BatchNormalization()
dense_layer_2 = Dense(20, activation='relu')
batchnorm_layer_2 = BatchNormalization()
dense_layer_3 = Dense(10, activation='relu')
batchnorm_layer_3 = BatchNormalization()
prediction_layer = Dense(10, activation='softmax')

model = models.Sequential([
    base_model,
    flatten_layer,
    dense_layer_1,
    batchnorm_layer_1,
    dense_layer_2,
    batchnorm_layer_2,
    dense_layer_3,
    batchnorm_layer_3,
    prediction_layer
])

# Making the Predictions

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy'],
)

es = EarlyStopping(monitor='val_accuracy', mode='max', patience=5, restore_best_weights=True)

model.fit(X_train, Y_train, epochs=10, validation_split=0.2, batch_size=32, callbacks=[es])

# Making the Predictions

accuracy = model.evaluate(X_test, Y_test, verbose=0)
print("Accuracy on test data:", accuracy[1])

