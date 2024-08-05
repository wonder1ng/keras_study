# 3장
from keras import layers, models

Nin = 784
Nh_l = [100, 50]
number_of_class = 10
Nout = number_of_class

class DNN(models.Sequential):
    def __init__(self, Nin, Nh_1, Nout):
        super().__init__()

        self.add(layers.Dense(Nh_1[0], activation='relu', input_shape=(Nin, ), name='Hidden-1'))
        self.add(layers.Dropout(0.2))
        self.add(layers.Dense(Nh_1[1], activation='relu', name='Hidden-2'))
        self.add(layers.Dropout(0.2))
        self.add(layers.Dense(Nout, activation='softmax'))

        self.compile(loss='categorical_crossentropy', optimizer='adam', metrics='accuracy')

import numpy as np
from keras.datasets import mnist
from keras.utils import np_utils # to_categorical

def Data_func():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)

    L, W, H = X_train.shape
    X_train = X_train.reshape(-1, W*H)
    X_test = X_test.reshape(-1, W*H)

    X_train = X_train/255.0
    X_test = X_test/255.0

    return (X_train, y_train), (X_test, y_test)

(X_train, y_train), (X_test, y_test) = Data_func()

model = DNN(Nin, Nh_l, Nout)
history = model.fit(X_train, y_train, epochs=10, batch_size=100, validation_split=0.2)
performance_test = model.evaluate(X_test, y_test, batch_size=100)
print('Test Loss and Accuracy ->', performance_test)

# 컬러 이미지
import numpy as np
from keras.datasets import cifar10
from keras.utils import np_utils

def Data_func():
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)

    L, W, H, C = X_train.shape
    X_train = X_train.reshape(-1, W*H*C)
    X_test = X_test.reshape(-1, W*H*C)

    X_train = X_train/255.0
    X_test = X_test/255.0

    return (X_train, y_train), (X_test, y_test)


# 모델링
from keras import layers, models

class DNN(models.Sequential):
    def __init__(self, Nin, Nh_l, Pd_l, Nout):
        super().__init__()

        self.add(layers.Dense(Nh_l[0], activation='relu', input_shape=(Nin, ), name='Hiiden-1'))
        self.add(layers.Dropout(Pd_l[0]))
        self.add(layers.Dense(Nh_l[1], activation='relu', input_shape=(Nin, ), name='Hiiden-2'))
        self.add(layers.Dropout(Pd_l[1]))
        self.add(layers.Dense(Nout, activation='softmax'))
        self.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

# 학습 및 시각화
from original.keraspp.skeras import plot_loss, plot_acc
import matplotlib.pyplot as plt

Nh_l = [100, 50]
Pd_l = [0.0, 0.0]
number_of_class = 10
Nout = number_of_class

(X_train, y_train), (X_test, y_test) = Data_func()
model = DNN(X_train.shape[1], Nh_l, Pd_l, Nout)
history = model.fit(X_train, y_train, epochs=10, batch_size=100, validation_split=0.2)
performance_test = model.evaluate(X_test, y_test, batch_size=100)
print('Test Loss and Accuracy ->', performance_test)

plot_acc(history)
plt.show(block=False)
plt.pause(2)
plt.close()
plot_loss(history)
plt.show(block=False)
plt.pause(2)
plt.close()