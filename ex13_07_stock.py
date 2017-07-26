import tensorflow as tf
import numpy as np

def MinMaxScaler(data):
    numerator = data - np.min(data, 0)
    denominator = np.max(data,0) - np.min(data,0)
    #noise term prevents the zero division
    return numerator / (denominator + 1e-7)

timesteps = seq_length = 7
data_dim = 5
output_dim = 1

xy = np.loadtxt('data-02-stock_daily.csv', delimiter=',')
xy = xy[::-1]
xy = MinMaxScaler(xy)
x = xy
y = xy[:, [-1]]

dataX = []
dataY = []
for i in range(0, len(y) - seq_length):
    _x = x[i:i + seq_length]
    _y = y[i+seq_length]
    print(_x, "->", _y)
    dataX.append(_x)
    dataY.append(_y)