from keras.models import load_model
import numpy as np

test_data = np.loadtxt('data/test.csv', skiprows=1, delimiter=',')
X_test = test_data[:, :]
n_test, n_pixel = test_data.shape
height = round((n_pixel)**(1.0 / 2))
width = round(n_pixel / height)
X_test = X_test.reshape(n_test, 1, height, width).astype('float32')
X_test /= 255

model = load_model('large_files/cnn_model.h5')
y_test = model.predict(X_test, batch_size=128)

y_test_vals = y_test.argmax(axis=1)

with open('data/submission.csv', 'w') as f:
    f.write('ImageId,Label\n')
    i = 0
    for y in y_test_vals:
        i += 1
        f.write(str(i) + ',' + str(y))
        f.write('\n')
