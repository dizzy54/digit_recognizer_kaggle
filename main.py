from keras.datasets import mnist
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Activation, Dropout, Dense, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D

import numpy as np

(_, _), (X_val, y_val) = mnist.load_data()

train_data = np.loadtxt('data/train.csv', skiprows=1, delimiter=',')
test_data = np.loadtxt('data/test.csv', skiprows=1, delimiter=',')

print('train_data shape = ', train_data.shape)
n_train, _ = train_data.shape
n_test, n_pixel = test_data.shape
height = round((n_pixel)**(1.0 / 2))
width = round(n_pixel / height)
print('height, width = ', height, width)
# to prepare training data
X_train = train_data[:, 1:]
X_train = X_train.reshape(n_train, 1, height, width).astype('float32')
X_train /= 255
# print(X_train.shape)
y_train = train_data[:, 0]
# print(y_train[0:10])

# to prepare testing data
X_test = test_data[:, :]
X_test = X_test.reshape(n_test, 1, height, width).astype('float32')
X_test /= 255
print('X_test shape = ', X_test.shape)

# to prepare validation data
n_val, _, _ = X_val.shape
X_val = X_val.reshape(n_val, 1, height, width).astype('float32')
X_val /= 255

n_classes = 10

y_train = to_categorical(y_train, n_classes)
y_val = to_categorical(y_val, n_classes)

model = Sequential()

# number of convolutional filters
n_filters = 32

# convolution filter size (n_conv x n_conv filter)
n_conv = 3

# pooling window size (n_pool x n_pool window)
n_pool = 2

# first convolutional layer
model.add(
    Convolution2D(
        n_filters, n_conv, n_conv,
        border_mode='valid',    # narrow convolution, no spill over at border
        input_shape=(1, height, width),
    )
)
model.add(Activation('relu'))

# second convolutional layer
model.add(Convolution2D(n_filters, n_conv, n_conv))
model.add(Activation('relu'))

# pooling layer
model.add(MaxPooling2D(pool_size=(n_pool, n_pool)))
model.add(Dropout(0.25))

# flatten data for 1D layers
model.add(Flatten())

# Dense(n_output) fully connected hidden layer
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))

# output layer
model.add(Dense(n_classes))
model.add(Activation('softmax'))

# compile model
model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy'],
)

# number of examples to look at during each training session
batch_size = 128

# number of times to run through full sets of examples
n_epochs = 10

model.fit(
    X_train,
    y_train,
    batch_size=batch_size,
    nb_epoch=n_epochs,
    validation_data=(X_val, y_val)
)

# save model
model.save('large_files/cnn_model.h5')
# to see results
loss, accuracy = model.evaluate(X_val, y_val)
print('loss: ', loss)
print('accuracy: ', accuracy)

# to predict
y_test = model.predict(X_test, batch_size=batch_size)

# create submission file
with open('data/submission_array.csv', 'w') as f:
    # f.write('ImageId,Label')
    i = 0
    for y in y_test:
        i += 1
        f.write(str(i) + ',' + str(y))
        f.write('\n')
