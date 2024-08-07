# 7장
import numpy as np
import matplotlib.pyplot as plt
from keras import models
from keras.layers import Dense, Conv1D, Reshape, Flatten, Lambda
from keras.optimizers import Adam
from keras import backend as K


def main():
    machine = Machine(n_batch=1, ni_D=100)
    machine.run(n_repeat=200, n_show=200, n_test=100)


class Data:
    def __init__(self, mu, sigma, ni_D):
        self.real_sample = lambda n_batch: np.random.normal(mu, sigma, (n_batch, ni_D))  # 허구 데이터
        self.in_sample = lambda n_batch: np.random.rand(n_batch, ni_D)  # 노이즈


class Machine:
    def __init__(self, n_batch=10, ni_D=100):
        data_mean = 4
        data_stddev = 1.25
        self.data = Data(data_mean, data_stddev, ni_D)

        self.gan = GAN(ni_D=ni_D, nh_D=50, nh_G=50)
        self.n_batch = n_batch
        self.n_iter_D = 1
        self.n_iter_G = 5

    def run_epochs(self, epochs, n_test):
        self.train((epochs))
        self.test_and_show(n_test)

    def run(self, n_repeat=30000 // 200, n_show=200, n_test=100):
        for ii in range(n_repeat):
            print('Stage', ii, '(Epoch: {}'.format(ii * n_show))
            self.run_epochs(n_show, n_test)
            plt.show(block=False)
            plt.pause(2)
            plt.close()

    def train(self, epochs):
        for epoch in range(epochs):
            self.train_each()

    def train_each(self):
        for it in range(self.n_iter_D):
            self.train_D()
        for it in range(self.n_iter_G):
            self.train_GD()

    def test(self, n_test):
        gan = self.gan
        data = self.data
        Z = data.in_sample(n_test)
        Gen = gan.G.predict(Z)

        return Gen, Z

    def train_D(self):
        gan = self.gan
        n_batch = self.n_batch
        data = self.data

        Real = data.real_sample(n_batch)  # (n_batch, ni_D)
        Z = data.in_sample(n_batch)  # (n_batch, ni_D)
        Gen = gan.G.predict(Z)  # (n_batch, ni_D)
        gan.D.trainable = True
        gan.D_train_on_batch(Real, Gen)

    def train_GD(self):
        gan = self.gan
        n_batch = self.n_batch
        data = self.data
        Z = data.in_sample(n_batch)

        gan.D.trainable = False
        gan.GD_train_on_batch(Z)

    def test_and_show(self, n_test):
        data = self.data
        Gen, Z = self.test(n_test)
        Real = data.real_sample(n_test)
        self.show_hist(Real, Gen, Z)
        Machine.print_stat(Real, Gen)

    def show_hist(self, Real, Gen, Z):
        plt.hist(Real.reshape(-1), histtype='step', label='Real')
        plt.hist(Gen.reshape(-1), histtype='step', label='Generated')
        plt.hist(Z.reshape(-1), histtype='step', label='Input')
        plt.legend(loc=0)

    @staticmethod  # 인스턴스에 접근하지 않고 바로 함수로써 사용할 수 있게 하는 데코레이터
    def print_stat(Real, Gen):
        def stat(d):
            return (np.mean(d), np.std(d))

        print('Mean and Std of Real:', stat(Real))
        print('Mean and Std of Gen:', stat(Gen))


def add_decorate(x):
    m = K.mean(x, axis=-1, keepdims=True)
    d = K.square(x - m)

    return K.concatenate([x, d], axis=-1)


def add_decorate_shape(input_shape):
    shape = list(input_shape)
    assert len(shape) == 2
    shape[1] *= 2
    return tuple(shape)


lr = 2e-4
adam = Adam(learning_rate=lr, beta_1=0.9, beta_2=0.999)


def model_compile(model):
    return model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['acc'])


class GAN:
    def __init__(self, ni_D, nh_D, nh_G):
        self.ni_D = ni_D
        self.nh_D = nh_D
        self.nh_G = nh_G

        self.D = self.gen_D()
        self.G = self.gen_G()
        self.GD = self.make_GD()

    def gen_D(self):
        ni_D = self.ni_D
        nh_D = self.nh_D

        D = models.Sequential()
        D.add(Lambda(add_decorate, output_shape=add_decorate_shape, input_shape=(ni_D,)))
        D.add(Dense(nh_D, activation='relu'))
        D.add(Dense(nh_D, activation='relu'))
        D.add(Dense(1, activation='sigmoid'))
        model_compile(D)

        return D

    def gen_G(self):
        ni_D = self.ni_D
        nh_G = self.nh_G
        G = models.Sequential()
        G.add(Reshape((ni_D, 1), input_shape=(ni_D,)))
        G.add(Conv1D(nh_G, 1, activation='relu'))
        G.add(Conv1D(nh_G, 1, activation='sigmoid'))
        G.add(Conv1D(1, 1))
        G.add(Flatten())
        model_compile(G)

        return G

    def make_GD(self):
        G, D = self.G, self.D
        GD = models.Sequential()
        GD.add(G)
        GD.add(D)
        D.trainable = False
        model_compile(GD)
        D.trainable = True

        return GD

    def D_train_on_batch(self, Real, Gen):
        D = self.D
        X = np.concatenate([Real, Gen], axis=0)
        y = np.array([1] * Real.shape[0] + [0] * Gen.shape[0])
        D.train_on_batch(X, y)

    def GD_train_on_batch(self, Z):
        GD = self.GD
        y = np.array([1] * Z.shape[0])
        GD.train_on_batch(Z, y)


# if __name__ == '__main__':
#     main()


# 합성곱 계층 GAN
from keras.datasets import mnist
from PIL import Image
import numpy as np
import math, os
import keras.backend as K
import tensorflow as tf

K.set_image_data_format('channels_first')
print(K.image_data_format)


def mse_4d(y_true, y_pred):
    return K.mean(K.square(y_pred - y_true), axis=(1, 2, 3))

def mse_4d_tf(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_pred - y_true), axis=(1, 2, 3))


import argparse


def main():
    parser = argparse.ArgumentParser()  # 인자값을 처리하는 클래스인데 용처를 잘 모르겠음

    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for the networks')
    parser.add_argument('--epochs', type=int, default=1000, help='Epochs for the networks')
    parser.add_argument('--output_fold', type=str, default='GAN_OUT', help='Output fold to save the results')
    parser.add_argument('--input_dim', type=int, default=10, help='Input dimension for the generator')
    parser.add_argument('--n_train', type=int, default=32, help='The number of training data')

    args = parser.parse_args()

    train(args)


from keras import models, layers, optimizers


class GAN(models.Sequential):
    def __init__(self, input_dim=64):
        super().__init__()
        self.input_dim = input_dim

        self.generator = self.GENERATOR()
        self.discriminator = self.DISCRIMINATOR()

        self.add(self.generator)
        self.discriminator.trainable = False
        self.add(self.discriminator)
        self.compile_all()

    def compile_all(self):
        d_optim = optimizers.SGD(learning_rate=0.0005, momentum=0.9, nesterov=True)
        g_optim = optimizers.SGD(learning_rate=0.0005, momentum=0.9, nesterov=True)

        self.generator.compile(loss=mse_4d_tf, optimizer='SGD')
        self.compile(loss='binary_crossentropy', optimizer=g_optim)
        self.discriminator.trainable = True
        self.discriminator.compile(loss='binary_crossentropy', optimizer=d_optim)

    def GENERATOR(self):
        input_dim = self.input_dim
        model = models.Sequential()
        model.add(layers.Dense(1024, activation='tanh', input_dim=input_dim))
        model.add(layers.Dense(128 * 7 * 7, activation='tanh'))
        model.add(layers.BatchNormalization())
        model.add(layers.Reshape((128, 7, 7), input_shape=(128 * 7 * 7,)))
        model.add(layers.UpSampling2D(size=(2, 2)))
        model.add(layers.Conv2D(64, (5, 5), padding='same', activation='tanh'))
        model.add(layers.UpSampling2D(size=(2, 2)))
        model.add(layers.Conv2D(1, (5, 5), padding='same', activation='tanh'))

        return model

    def DISCRIMINATOR(self):
        model = models.Sequential()
        model.add(layers.Conv2D(64, (5, 5), padding='same', activation='tanh', input_shape=(1, 28, 28)))
        model.add(layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(layers.Conv2D(128, (5, 5), activation='tanh'))
        model.add(layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(layers.Flatten())
        model.add(layers.Dense(1024, activation='tanh'))
        model.add(layers.Dense(1, activation='sigmoid'))

        return model

    def get_z(self, ln):
        input_dim = self.input_dim
        return np.random.uniform(-1, 1, (ln, input_dim))

    def train_both(self, x):
        ln = x.shape[0]
        # First trial for training discriminator
        z = self.get_z(ln)
        w = self.generator.predict(z, verbose=0)
        xw = np.concatenate((x, w))
        y2 = [1] * ln + [0] * ln

        # train_on_batch에서 list를 넣으면 아래의 오류가 발생하여 array로 변환해야 함
        # AttributeError: 'int' object has no attribute 'shape'

        d_loss = self.discriminator.train_on_batch(xw, np.array(y2))
        z = self.get_z(ln)
        self.discriminator.trainable = False

        # 아래 오류는 해결 못함
        #     ValueError: Dimensions must be equal, but are 28 and 16 for '{{node mse_4d_tf/sub}} = Sub[T=DT_FLOAT](sequential/conv2d_1/Tanh, IteratorGetNext:1)' with input shapes: [16,1,28,28], [16].
        # g_loss = self.generator.train_on_batch(z, np.array([1] * ln, dtype=np.float32))
        g_loss = 0
        self.discriminator.trainable = True

        return d_loss, g_loss


def combine_images(generated_images):
    num = generated_images.shape[0]
    width = int(math.sqrt(num))
    height = int(math.ceil(float(num) / width))
    shape = generated_images.shape[2:]
    image = np.zeros((height * shape[0], width * shape[1]),
                     dtype=generated_images.dtype)
    for index, img in enumerate(generated_images):
        i = int(index / width)
        j = index % width
        image[i * shape[0]:(i + 1) * shape[0],
        j * shape[1]:(j + 1) * shape[1]] = img[0, :, :]
    return image


def get_x(X_train, index, BATCH_SIZE):
    return X_train[index * BATCH_SIZE:(index + 1) * BATCH_SIZE]


def save_images(generated_images, output_fold, epoch, index):
    image = combine_images(generated_images)
    image = image * 127.5 + 127.5
    Image.fromarray(image.astype(np.uint8)).save(
        output_fold + '/' +
        str(epoch) + "_" + str(index) + ".png")


def load_data(n_train):
    (X_train, y_train), (_, _) = mnist.load_data()

    return X_train[:n_train]


def train(args):
    BATCH_SIZE = args.batch_size
    epochs = args.epochs
    output_fold = args.output_fold
    input_dim = args.input_dim
    n_train = args.n_train

    os.makedirs(output_fold, exist_ok=True)
    print('Output_fold is', output_fold)

    X_train = load_data(n_train)
    X_train = (X_train.astype(np.float32) - 127.5) / 127.5
    X_train = X_train.reshape((X_train.shape[0], 1) + X_train.shape[1:])

    gan = GAN(input_dim)

    d_loss_ll = []
    g_loss_ll = []
    for epoch in range(epochs):
        print("Epoch is", epoch)
        print("Number of batches", int(X_train.shape[0] / BATCH_SIZE))

        d_loss_l = []
        g_loss_l = []

        for index in range(int(X_train.shape[0] / BATCH_SIZE)):
            x = get_x(X_train, index, BATCH_SIZE)

            d_loss, g_loss = gan.train_both(x)
            d_loss_l.append(d_loss)
            g_loss_l.append(g_loss)

        if epoch % 10 == 0 or epoch == epochs - 1:
            z = gan.get_z(x.shape[0])
            w = gan.generator.predict(z, verbose=0)
            save_images(w, output_fold, epoch, index)

        d_loss_ll.append(d_loss_l)
        g_loss_ll.append(g_loss_l)

    gan.generator.save_weights(output_fold + '/' + 'generator', True)
    gan.discriminator.save_weights(output_fold + '/' + 'discriminator', True)

    np.savetxt(output_fold + '/' + 'd_loss', d_loss_ll)
    np.savetxt(output_fold + '/' + 'g_loss', g_loss_ll)


if __name__ == '__main__':
    main()