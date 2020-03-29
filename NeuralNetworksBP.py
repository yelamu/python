# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import scipy.optimize as opt
from sklearn.metrics import classification_report  # 这个包是评价报告
from sklearn.preprocessing import OneHotEncoder

def load_mat(path):
    '''读取数据'''
    data = loadmat(path)  # return a dict
    X = data['X']
    y = data['y'].flatten()
    
    return X, y  


def plot_100_images(X):
    """随机画100个数字"""
    index = np.random.choice(range(5000), 100)
    images = X[index]
    fig, ax_array = plt.subplots(10, 10, sharey=True, sharex=True, figsize=(8, 8))
    for r in range(10):
        for c in range(10):
            ax_array[r, c].matshow(images[r*10 + c].reshape(20,20), cmap='gray_r')
    plt.xticks([])
    plt.yticks([])
    plt.show()

def expand_y(y):
    print('y: {}'.format(y))
    result = []
    # 把y中每个类别转化为一个向量，对应的lable值在向量对应位置上置为1
    for i in y:
        print('i: {}'.format(i))
        y_array = np.zeros(10)
        y_array[i-1] = 1
        print('y_array: {}'.format(y_array))
        result.append(y_array)
    '''
    # 或者用sklearn中OneHotEncoder函数
    encoder =  OneHotEncoder(sparse=False)  # return a array instead of matrix
    y_onehot = encoder.fit_transform(y.reshape(-1,1))
    return y_onehot
    ''' 
    return np.array(result)

X,y = load_mat('./data/ex4data1.mat')
#plot_100_images(X)

print('y:', y.shape)
print(y[1:6])
print(np.unique(y))

X = np.insert(X, 0, 1, axis=1)
y = expand_y(y)

print('X: {}'.format(X.shape))
print('y: {}'.format(y.shape))
