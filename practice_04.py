# 5장
from __future__ import print_function  # 파이썬 2와 3 간의 호환성 위한 패키지
# 파이썬 2에선 print를 괄호 없이 사용하지만 이 함수를 호출함으로써 함수로써(괄호와 함꼐) 사용하게 함

# 자언여
from keras.preprocessing import sequence
from keras.utils import pad_sequences
from keras.datasets import imdb
from keras import layers, models


class Data:
    def __init__(self, max_features=20000, maxlen=80):
        (x_train, self.y_train), (x_test, self.y_test) = imdb.load_data(num_words=max_features)

        # x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
        # x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
        # pad_sequences의 경로가 바뀜
        self.x_train = pad_sequences(x_train, maxlen=maxlen)
        self.x_test = pad_sequences(x_test, maxlen=maxlen)


class RNN_LSTM(models.Model):
    def __init__(self, max_features, maxlen):
        # model = models.Sequential()
        # model.add(layers.Embedding(max_features, 128))
        # model.add(layers.LSTM(128, dropout=0.2, recurrent_dropout=0.2))
        # model.add(layers.Dense(1, activation='sigmoid'))
        # model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

        x = layers.Input((maxlen,))
        h = layers.Embedding(max_features, 128)(x)
        h = layers.LSTM(128, dropout=0.2, recurrent_dropout=0.2)(h)
        y = layers.Dense(1, activation='sigmoid')(h)
        super().__init__(x, y)
        self.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])


class Machine:
    def __init__(self, max_features=20000, maxlen=80):
        self.data = Data(max_features, maxlen)
        self.model = RNN_LSTM(max_features, maxlen)

    def run(self, epochs=3, batch_size=32):
        data = self.data
        model = self.model

        print('Training stage')
        print('====================')
        model.fit(data.x_train, data.y_train, batch_size=batch_size, epochs=epochs,
                  validation_data=(data.x_test, data.y_test))
        score, acc = model.evaluate(data.x_test, data.y_test, batch_size=batch_size)
        print('Test performance: accuracy{0}, loss={1}'.format(acc, score))


def main():
    m = Machine()
    m.run(batch_size=512)


if __name__ == '__main__':
    main()

# 시계열
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import model_selection
from keras import models, layers

from original.keraspp import skeras


def main():
    machine = Machine()
    machine.run(epochs=400)


class Machine():
    def __init__(self):
        self.data = Dataset()
        shape = self.data.X.shape[1:]
        self.model = rnn_model(shape)

    def run(self, epochs=400):
        d = self.data
        X_train, X_test = d.X_train, d.X_test
        y_train, y_test = d.y_train, d.y_test
        X, y = d.X, d.y

        m = self.model
        h = m.fit(X_train, y_train, epochs=epochs, validation_data=[X_test, y_test], verbose=0)
        skeras.plot_loss(h)
        plt.title('History of training')
        plt.show(block=False)
        plt.pause(2)
        plt.close()

        yp = m.predict(X_test)
        print('Loss:', m.evaluate(X_test, y_test))
        plt.plot(yp, label='Original')
        plt.plot(y_test, label='Prediction')
        plt.legend(loc=0)
        plt.title('Validation Results')
        plt.show(block=False)
        plt.pause(2)
        plt.close()

        yp = m.predict(X_test).reshape(-1)
        print('Loss:', m.evaluate(X_test, y_test))
        print(yp.shape, y_test.shape)
        df = pd.DataFrame()
        df['Sample'] = list(range(len(y_test))) * 2
        df['Normalized #Passengers'] = np.concatenate([y_test, yp], axis=0)
        df['Type'] = ['Original'] * len(y_test) + ['Prediction'] * len(yp)

        plt.figure(figsize=(7, 5))
        sns.barplot(x='Sample', y='Normalized #Passengers', hue='Type', data=df)
        plt.ylabel('Normalized #Passengers')
        plt.show(block=False)
        plt.pause(2)
        plt.close()

        yp = m.predict(X)
        plt.plot(yp, label='Original')
        plt.plot(y, label='Prediction')
        plt.legend(loc=0)
        plt.title('All Results')
        plt.show(block=False)
        plt.pause(2)
        plt.close()


def rnn_model(shape):
    m_x = layers.Input(shape=shape)
    m_h = layers.LSTM(10)(m_x)
    m_y = layers.Dense(1)(m_h)
    m = models.Model(m_x, m_y)
    m.compile('adam', 'mean_squared_error')
    m.summary()

    return m


class Dataset:
    def __init__(self, fname='original/international-airline-passengers.csv', D=12):
        data_dn = load_data(fname=fname)
        X, y = get_Xy(data_dn, D=D)
        X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2, random_state=42)
        self.X, self.y = X, y
        self.X_train, self.X_test = X_train, X_test
        self.y_train, self.y_test = y_train, y_test


def load_data(fname='original/international-airline-passengers.csv'):
    dataset = pd.read_csv(fname, usecols=[1], engine='python', skipfooter=3)
    data = dataset.values.reshape(-1)

    plt.plot(data)
    plt.xlabel('Time')
    plt.ylabel('#Passengers')
    plt.title('Original Data')
    plt.show(block=False)
    plt.pause(2)
    plt.close()

    data_dn = (data - np.mean(data)) / np.std(data) / 5
    plt.plot(data_dn)
    plt.xlabel('Time')
    plt.ylabel('Normalized #Passengers')
    plt.title('Normalized data by $E[]$ and $5\sigma$')
    plt.show(block=False)
    plt.pause(2)
    plt.close()

    return data_dn


def get_Xy(data, D=12):
    # make X and y
    X_l = []
    y_l = []
    N = len(data)
    assert N > D, 'N should be larger than D, where N is len(data)'
    for ii in range(N - D - 1):
        X_l.append(data[ii:ii + D])
        y_l.append(data[ii + D])
    X = np.array(X_l)
    X = X.reshape(X.shape[0], X.shape[1], 1)
    y = np.array(y_l)
    print(X.shape, y.shape)

    return X, y


if __name__ == '__main__':
    main()