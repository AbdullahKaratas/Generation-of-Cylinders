import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers, datasets
import numpy as np

import numpy as np
import matplotlib.pyplot as plt

from skimage.transform import radon, iradon

#(x, y), (x_val, y_val) = datasets.fashion_mnist.load_data()


def cylinder_dataset(data_train_path, data_label_path):
    print('Start reading data ...')
    xAll = np.load(data_train_path)  # Change
    yAll = np.load(data_label_path)  # Change
    print('x/y shape:', xAll.shape, yAll.shape)

    x = xAll
    y = yAll

    return x, y


def main(sizeOfVector, data_train_path, data_label_path):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or any {'0', '1', '2'}

    x, y = cylinder_dataset(data_train_path, data_label_path)

    normed_train_data = x
    train_labels = y

    # Change
    model = keras.Sequential([
        layers.Reshape(target_shape=(sizeOfVector * 3,),
                       input_shape=(sizeOfVector, 3)),
        tf.keras.layers.Dropout(0.2),
        layers.Dense(70, activation='elu'),
        tf.keras.layers.Dropout(0.2),
        layers.Dense(70, activation='elu'),
        tf.keras.layers.Dropout(0.4),
        layers.Dense(7)])

    model.summary()
    model.compile(optimizer=optimizers.Adam(),
                  loss='mse')

    #early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

    history = model.fit(normed_train_data, train_labels, epochs=300,
                        validation_split=0.2)
    # callbacks=[early_stop]

    model.save('cylinder_predictions_d108.h5')

    return history


if __name__ == '__main__':
    print(tf.__version__)
    # Data (x, y, z). Here, sizeOfVector = len(x)
    # data_train_path: Path of training data
    # data_label_path: Path of labeled data
    sizeOfVector = 24000
    main(sizeOfVector, data_train_path, data_label_path)
