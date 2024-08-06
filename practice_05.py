# 6장
from keras import layers, models


class AE(models.Model):
    def __init__(self, x_nodes, z_dim):
        x_shape = (x_nodes,)
        x = layers.Input(shape=x_shape)  # 입력 계층
        z = layers.Dense(z_dim, activation='relu')(x)  # 은닉 계층
        y = layers.Dense(x_nodes, activation='sigmoid')(z)  # 출력 계층
        super().__init__(x, y)
        self.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

        self.x = x
        self.z = z
        self.z_dim = z_dim

    def Encoder(self):  # output을 z로 불러와서 encoder 부분만으로 만든 모델
        return models.Model(self.x, self.z)

    def Decoder(self):  # deoder부분인 y_layer를 가져온 후 input을 맞춘 새로운 z를 입력한 모델
        z_shape = (self.z_dim,)
        z = layers.Input(shape=z_shape)
        y_layer = self.layers[-1]
        y = y_layer(z)

        return models.Model(z, y)


import matplotlib.pyplot as plt


def show_ae(autoencoder):
    (_, _), (x_test, _) = mnist.load_data()
    x_test = x_test.astype('float32') / 255.
    x_test = x_test.reshape((len(x_test), -1))

    encoder = autoencoder.Encoder()
    decoder = autoencoder.Decoder()
    encoded_imgs = encoder.predict(x_test)
    decoded_imgs = decoder.predict(encoded_imgs)

    n = 10
    plt.figure(figsize=(20, 6))
    for i in range(n):
        ax = plt.subplot(3, n, i + 1)
        plt.imshow(x_test[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        ax = plt.subplot(3, n, i + 1 + n)
        plt.stem(encoded_imgs[i].reshape(-1))

        ax = plt.subplot(3, n, i + 1 + n + n)
        plt.imshow(decoded_imgs[i].reshape(28, 28))

    plt.show(block=False)
    plt.pause(2)
    plt.close()


from keras.datasets import mnist
from original.keraspp.skeras import plot_loss, plot_acc


def main():
    (x_train, _), (x_test, _) = mnist.load_data()
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    x_train = x_train.reshape((len(x_train), -1))
    x_test = x_test.reshape((len(x_test), -1))

    x_nodes = 784
    z_dim = 36

    autoencoder = AE(x_nodes, z_dim)
    history = autoencoder.fit(x_train, x_train, epochs=10, batch_size=1024, shuffle=True,
                              validation_data=(x_test, x_test))
    plot_acc(history)
    plt.show(block=False)
    plt.pause(2)
    plt.close()

    plot_loss(history)
    plt.show(block=False)
    plt.pause(2)
    plt.close()

    show_ae(autoencoder)


if __name__ == '__main__':
    main()


# 합성곱 AE 모델링
from keras import layers, models

def Conv2D(filters, kernel_size, padding='same', activation='relu'):
    return layers.Conv2D(filters, kernel_size, padding=padding, activation=activation)

class AE(models.Model):
    def __init__(self, org_shape):
        # Input
        original = layers.Input(shape=org_shape)
        x = Conv2D(4, (3, 3))(original)
        x = layers.MaxPooling2D((2, 2), padding='same')(x)
        x = Conv2D(8, (3, 3))(x)
        x = layers.MaxPooling2D((2, 2), padding='same')(x)

        z = Conv2D(1, (7, 7))(x)  # encoding의 output이자 decoding의 input

        y = Conv2D(16, (3, 3))(z)
        y = layers.UpSampling2D((2, 2))(y)
        y = Conv2D(8, (3, 3))(y)
        y = layers.UpSampling2D((2, 2))(y)
        y = Conv2D(4, (3, 3))(y)

        decoded = Conv2D(1, (3, 3), activation='sigmoid')(y)

        super().__init__(original, decoded)
        self.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])


from practice_03 import DATA
from original.keraspp.skeras import plot_loss, plot_acc
import matplotlib.pyplot as plt
from keras import backend

def show_ae(autoencoder, data):
    x_test = data.X_test
    decoded_imgs = autoencoder.predict(x_test)
    print(decoded_imgs.shape, data.X_test.shape)

    if backend.image_data_format() == 'channels_first':
        N, n_ch, n_i, n_j = x_test.shape
    else:
        N, n_i, n_j, n_ch = x_test.shape

    x_test = x_test.reshape(N, n_i, n_j)
    decoded_imgs = decoded_imgs.reshape(decoded_imgs.shape[0], n_i, n_j)

    n = 10
    plt.figure(figsize=(20, 4))
    for i in range(n):
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(x_test[i])
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(decoded_imgs[i])
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

def main(epochs=20, batch_size=128):
    data = DATA()
    autoencoder = AE(data.input_shape)
    # history = autoencoder.fit(data.X_train, data.X_train, epochs=epochs, batch_size=batch_size, shuffle=True, validation_split=0.2)
    history = autoencoder.fit(data.X_train, data.X_train, epochs=epochs, batch_size=batch_size, shuffle=True,
                              validation_data=(data.X_test, data.X_test))

    plot_acc(history)
    plt.show(block=False)
    plt.pause(2)
    plt.close()

    plot_loss(history)
    plt.show(block=False)
    plt.pause(2)
    plt.close()

    show_ae(autoencoder, data)
    plt.show(block=False)
    plt.pause(2)
    plt.close()


if __name__ == '__main__':
    main(batch_size=512)