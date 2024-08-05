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
