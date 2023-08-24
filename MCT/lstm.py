
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from pandas import read_csv
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM

dir = r'D:\Nut_Store\Meachine Learning\Tensorflow tutorial\RNN\Volumetric_swelling'
csv_path = dir+'\wpvolumetric-swelling-221.csv'

data = pd.read_csv(csv_path)
data = data.iloc[:,0:3].values

def get_pic(data):
    data_x = data[:, 0:1]
    data_y = data[:,2:]
    plt.plot(data_x,data_y)
    plt.figure(figsize=(100, 50))
    plt.show()

get_pic(data)
print(data.shape)

#
#

def processing(set, seg_length,predict_length):
    x = []
    y = []
    step = 1
    for i in range(len(set)-seq_length-predict_len):
        _x = set[i*step:(i*step+seq_length), :]
        _y = set[i*step+seq_length+predict_len, :]
        x.append(_x)
        y.append(_y)
    return np.array(x), np.array(y)

sc = MinMaxScaler()
data = sc.fit_transform(data[0:221, :])

#print('data',data)

seq_length = 10
predict_len = 2
x, y = processing(data, seq_length, predict_len)
n_output = y.shape[1]
print('n_output',n_output)
print(x.shape)
print(y.shape)

train_size = int(len(y) * 0.7)
val_size = int(len(y) * 0.3)

x_train = x[:train_size]
x_test = x[train_size:]
y_train = y[:train_size]
y_test = y[train_size:]

print(np.shape(x_train))
print(np.shape(y_train))

x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 3))
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 3))

print(x_train)
#define lstm
model = Sequential()
model.add(LSTM(60, input_shape=(x_train.shape[1], 3), return_sequences=True))
model.add(LSTM(100))
model.add(Dense(n_output, activation='linear'))
model.summary()

model.compile(loss='mse', optimizer='rmsprop')
print('Train...')
model.fit(x_train, y_train, batch_size=x_test.shape[0], epochs=300, validation_split=0.3, validation_data=(x_test, y_test))

#does the model work well as we anticipated?
predict_train = model.predict(x_train)
predict_test = model.predict(x_test)

y_test = sc.inverse_transform(np.reshape(y_test, (len(y_test), n_output)))
predict_test= sc.inverse_transform(predict_test)

y_train = sc.inverse_transform(np.reshape(y_train, (len(y_train), n_output)))
predict_train = sc.inverse_transform(predict_train)
#print(type(predict))
#print(predict)

#plot
plt.plot(predict_test[:,1:2], 'g:')
plt.plot(y_test[:,1:2], 'r-')

plt.plot(predict_train[:,1:2],'g:')
plt.plot(y_train[:,1:2])

plt.show()

print(predict_test.shape)
