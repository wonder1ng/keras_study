# 1장
import keras

print(keras.backend.backend())

import numpy as np

x = np.array([0, 1, 2, 3, 4])
y = x * 2 + 1

import keras

model = keras.models.Sequential()
model.add(keras.layers.Dense(1, input_shape=(1, )))
model.compile(optimizer='SGD', loss='mse')

model.fit(x[:2], y[:2], epochs=1000, verbose=0)

print('Targets:', y[2:])
print('Predictions:', model.predict(x[2:]).flatten())

# 2장
input_num = 784
hidden_num = 100
number_of_class = 10
output_num = number_of_class

# 분산 방식 모델링, 함수형 구현
from keras import layers, models

x = layers.Input(shape=(input_num, ))
h = layers.Activation('relu')(layers.Dense(hidden_num)(x))
y = layers.Activation('softmax')(layers.Dense(output_num)(h))

model = models.Model(x, y)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 연쇄 방식 모델링, 함수형 구현
model = models.Sequential() # 모델 구조 정의 전 Sequential로 초기화
model.add(layers.Dense(hidden_num, activation='relu', input_shape=(input_num, )))
model.add(layers.Dense(output_num, activation='softmax'))

model = models.Model(x, y)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 분산 방식 모델링, 객체지향형 구현
class ANN(models.Model):
    def __init__(self, input_num, hidden_num, output_num, **kwargs):
        hidden = layers.Dense(hidden_num)
        output = layers.Dense(output_num)
        relu = layers.Activation('relu')
        softmax = layers.Activation('softmax')

        x = layers.Input(shape=input_num)
        h = relu(hidden(x))
        y = softmax(output(h))

        super().__init__(x, y)
        self.compile(loss='categorical_crossentropu', optimizer='adam', metrics=['accuracy'])

# 연쇄 방식 모델링, 객체지향형 구현
class ANN(models.Sequential):
    def __init__(self, input_num, hidden_num, output_num):
        super().__init__()
        self.add(layers.Dense(hidden_num, activation='relu', input_shape=(input_num, )))
        self.add(layers.Dense(output_num, activation='softmax'))
        self.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# mnist 불러오기
import numpy as np
# from keras import datasets  #mnist
from keras.datasets import mnist    # 이렇게 불러와야 실행됨
from keras.utils import np_utils # to_categorical

def Data_func():
    # (X_train, y_train), (X_test, y_test) = datasets.mnist.load_data()
    (X_train, y_train), (X_test, y_test) = mnist.load_data()  # 위의 코드 오류 발생으로 변경
    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)

    L, W, H = X_train.shape
    X_train = X_train.reshape(-1, W*H)
    X_test = X_test.reshape(-1, W*H)

    X_train = X_train/255.0
    X_test = X_test/255.0

    return (X_train, y_train), (X_test, y_test)

# history 시각화
import matplotlib.pyplot as plt

def plot_loss(history):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc=0)

def plot_acc(history):
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc=0)

# 모델 학습
input_num = 784
hidden_num = 100
number_of_class = 10
output_num = number_of_class
model = ANN(input_num, hidden_num, output_num)
(X_train, y_train), (X_test, y_test) = Data_func()

history = model.fit(X_train, y_train, epochs=5, batch_size=100, validation_split=0.2)

performance_test = model.evaluate(X_test, y_test, batch_size=100)
print('Test loss and Accuracy -> {: .2f}, {: .2f}'.format(*performance_test))

# 시각화
plot_loss(history)
plt.show(block=False)
plt.pause(2)
plt.close()

plot_acc(history)
plt.show(block=False)
plt.pause(2)
plt.close()

# 시계열
from keras import layers, models

class ANN(models.Model):
    def __init__(self, Nin, Nh, Nout):
        hidden = layers.Dense(Nh)
        output = layers.Dense(Nout)
        relu = layers.Activation('relu')

        x = layers.Input(shape=(Nin, ))
        h = relu(hidden(x))
        y = output(h)

        super().__init__(x, y)
        self.compile(loss='mse', optimizer='sgd')

# 보스턴 집값 데이터
from keras.datasets import boston_housing
from sklearn import preprocessing

def Data_func():
    (X_train, y_train), (X_test, y_test) = boston_housing.load_data()
    scaler = preprocessing.MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return (X_train, y_train), (X_test, y_test)

# 시각화
from original.keraspp.skeras import plot_loss
import matplotlib.pyplot as plt

def main():
    Nin = 13
    Nh = 5
    Nout = 1

    model = ANN(Nin, Nh, Nout)
    (X_train, y_train), (X_test, y_test) = Data_func()
    history = model.fit(X_train, y_train, epochs=100, batch_size=100, validation_split=0.2, verbose=2)

    performance_test = model.evaluate(X_test, y_test, batch_size=100)
    print('\nTest Loss -> {: .2f}'.format(performance_test))

    plot_loss(history)
    plt.show(block=False)
    plt.pause(2)
    plt.close()

main()

