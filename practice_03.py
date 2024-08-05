# 4장
import keras
from keras import models, layers
import matplotlib.pyplot as plt
from original.keraspp.skeras import plot_acc, plot_loss

# 흑백
class CNN(models.Sequential):
    def __init__(self, input_shape, num_classes):
        super().__init__()

        # convolutionL 특징점 찾기
        self.add(layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
        self.add(layers.Conv2D(64, (3, 3), activation='relu'))
        self.add(layers.MaxPooling2D(pool_size=(2, 2)))  # pool_size: 1개의 값으로 변화할 크기
        self.add(layers.Dropout(0.25))
        self.add(layers.Flatten())

        # dense: 실질적 분류 작업
        self.add(layers.Dense(128, activation='relu'))
        self.add(layers.Dropout(0.5))
        self.add(layers.Dense(num_classes, activation='softmax'))

        self.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(), metrics=['acc'])


from keras.datasets import mnist

class DATA():
    def __init__(self):
        num_classes = 10

        (X_train, y_train), (X_test, y_test) = mnist.load_data()

        # 채널의 위치는 image_data_format에 따름
        print(keras.backend.image_data_format())

        # 1 = 채널의 크기
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
        # X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)
        X_test = X_test.reshape(*X_test.shape, 1)
        input_shape = (X_train.shape[1], X_train.shape[2], 1)

        ## 채널의 위치가 앞 단에 존재할 경우
        # X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1], X_train.shape[2])
        # X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1], X_test.shape[2])
        # input_shape = (1, X_train.shape[1], X_train.shape[2])

        X_train = X_train.astype('float32')
        X_test = X_test.astype('float32')
        X_train /= 255
        X_test /= 255

        y_train = keras.utils.to_categorical(y_train, num_classes)
        y_test = keras.utils.to_categorical(y_test, num_classes)

        self.input_shape = input_shape
        self.num_classes = num_classes
        self.X_train, self.y_train = X_train, y_train
        self.X_test, self.y_test = X_test, y_test


data = DATA()
model = CNN(data.input_shape, data.num_classes)
history = model.fit(data.X_train, data.y_train, batch_size=128, epochs=10, validation_split=0.2)
score = model.evaluate(data.X_test, data.y_test)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

plot_acc(history)
plt.show(block=False)
plt.pause(2)
plt.close()

plot_loss(history)
plt.show(block=False)
plt.pause(2)
plt.close()

# 컬러

import keras.backend
from sklearn import model_selection, metrics
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt
import os
from keras import backend as K
from keras.utils import np_utils
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from original.keraspp import skeras, sfile


class CNN(Model):
    def __init__(model, nb_classes, in_shape=None):
        model.nb_classes = nb_classes
        model.in_shape = in_shape
        model.build_model()
        super().__init__(model.x, model.y)
        model.compile()

    def build_model(model):
        nb_classes = model.nb_classes
        in_shape = model.in_shape

        x = Input(in_shape)
        h = Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=in_shape)(x)
        h = Conv2D(64, (3, 3), activation='relu')(h)
        h = MaxPooling2D(pool_size=(2, 2))(h)
        h = Dropout(0.25)(h)
        h = Flatten()(h)
        z_cl = h

        h = Dense(128, activation='relu')(h)
        h = Dropout(0.5)(h)
        z_fl = h

        y = Dense(nb_classes, activation='softmax', name='preds')(h)
        model.cl_part = Model(x, z_cl)
        model.fl_part = Model(x, z_fl)
        model.x, model.y = x, y


class DataSet():
    def __init__(self, X, y, nb_classes, scaling=True, test_size=0.2, random_state=0):
        self.X = X
        self.add_channels()
        X = self.X

        X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2,
                                                                            random_state=random_state)
        X_train = X_train.astype('float32')
        X_test = X_test.astype('float32')

        if scaling:
            # scaling to have (0, 1) for each feature (each pixel)
            scaler = MinMaxScaler()
            n = X_train.shape[0]
            X_train = scaler.fit_transform(X_train.reshape(n, -1)).reshape(X_train.shape)
            n = X_test.shape[0]
            X_test = scaler.transform(X_test.reshape(n, -1)).reshape(X_test.shape)
            self.scaler = scaler

        y_train = np_utils.to_categorical(y_train, nb_classes)
        y_test = np_utils.to_categorical(y_test, nb_classes)

        self.X_train, self.X_test = X_train, X_test
        self.y_train, self.y_test = y_train, y_test

    def add_channels(self):
        X = self.X

        if len(X.shape) == 3:
            N, img_rows, img_cols = X.shape

            if K.image_data_format() == 'channels_first':
                X = X.reshape(X.shape[0], 1, img_rows, img_cols)
                input_shape = (1, img_rows, img_cols)
            else:
                X = X.reshape(X.shape[0], img_rows, img_cols, 1)
                input_shape = (img_rows, img_cols, 1)
        else:
            input_shape = X.shape[1:]  # channel is already included

        self.X = X
        self.input_shape = input_shape


class Machine():
    def __init__(self, X, y, nb_classes=2, fig=True):
        self.nb_classes = nb_classes
        self.set_data(X, y)
        self.set_model()
        self.fig = fig

    def set_data(self, X, y):
        nb_classes = self.nb_classes
        self.data = DataSet(X, y, nb_classes)

    def set_model(self):
        nb_classes = self.nb_classes
        data = self.data
        self.model = CNN(nb_classes=nb_classes, in_shape=data.input_shape)

    def fit(self, nb_epoch=10, batch_size=128, verbose=1):
        data = self.data
        model = self.model
        history = model.fit(data.X_train, data.y_train, batch_size=batch_size, epochs=nb_epoch, verbose=verbose,
                            validation_data=(data.X_test, data.y_test))
        return history

    def run(self, nb_epoch=10, batch_size=128, verbose=1):
        data = self.data
        model = self.model
        fig = self.fig

        history = self.fit(nb_epoch=nb_epoch, batch_size=batch_size, verbose=verbose)
        score = model.evaluate(data.X_test, data.y_test, verbose=0)

        print('Confusion matrix')
        y_test_pred = model.predict(data.X_test, verbose=0)
        y_test_pred = np.argmax(y_test_pred, axis=1)
        print(metrics.confusion_matrix(data.y_test, y_test_pred))

        print('Test score:', score[0])
        print('Test accuracy:', score[1])

        suffix = sfile.unique_filename('datatime')
        foldname = 'output_' + suffix
        os.makedirs(foldname)
        skeras.save_history_history('history_history.npy', history.history, fold=foldname)
        model.save_weights(os.path.join(foldname, 'dl_model.h5'))
        print('Output results are saved in', foldname)

        if fig:
            plt.figure(figsize=(12, 4))
            plt.subplot(1, 2, 1)
            skeras.plot_acc(history)
            plt.subplot(1, 2, 3)
            skeras.plot_loss(history)
            plt.show(block=False)
            plt.pause(2)
            plt.close()
        self.history = history


from keras.datasets import cifar10
import keras

assert keras.backend.image_data_format() == 'channels_last'
from original.keraspp import aicnn


class Machine(aicnn.Machine):
    def __init__(self):
        (X, y), (x_test, y_test) = cifar10.load_data()
        super().__init__(X, y, nb_classes=10)

def main():
    m = Machine()
    m.run()

if __name__ == '__main__':
    main()