#%matplotlib inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import seaborn as sb
from scipy.io import loadmat
from sklearn import svm

def plotData(X, y):
    plt.figure(figsize=(8,5))
    plt.scatter(X[:,0], X[:,1], c=y.flatten(), cmap='rainbow')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.legend() 


def plotBoundary(clf, X):
    '''plot decision bondary'''
    x_min, x_max = X[:,0].min()*1.2, X[:,0].max()*1.1
    y_min, y_max = X[:,1].min()*1.1,X[:,1].max()*1.1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500),
                         np.linspace(y_min, y_max, 500))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contour(xx, yy, Z)


mat = loadmat('./data/ex6data1.mat')
print(mat.keys())
# dict_keys(['__header__', '__version__', '__globals__', 'X', 'y'])
X = mat['X']
y = mat['y']
plotData(X, y)
plt.show()
