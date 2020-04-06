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



def gaussKernel(x1, x2, sigma):
    return np.exp(- ((x1 - x2) ** 2).sum() / (2 * sigma ** 2))

mat3 = loadmat('data/ex6data3.mat')
X3, y3 = mat3['X'], mat3['y']
Xval, yval = mat3['Xval'], mat3['yval']
#plotData(X3, y3)
#plt.show()

Cvalues = (0.01, 0.03, 0.1, 0.3, 1., 3., 10., 30.)
sigmavalues = Cvalues
best_pair, best_score = (0, 0), 0

for C in Cvalues:
    for sigma in sigmavalues:
        gamma = np.power(sigma,-2.)/2
        model = svm.SVC(C=C,kernel='rbf',gamma=gamma)
        model.fit(X3, y3.flatten())
        this_score = model.score(Xval, yval)
        print('this_pair={}, this_score={}'.format((C, sigma), this_score))
        if this_score > best_score:
            best_score = this_score
            best_pair = (C, sigma)
print('best_pair={}, best_score={}'.format(best_pair, best_score))

#model = svm.SVC(C=best_pair[0], kernel='rbf', gamma = np.power(best_pair[1], -2.)/2)
model = svm.SVC(C=1.0, kernel='rbf', gamma = np.power(3.0, -2.)/2)
model.fit(X3, y3.flatten())
plotData(X3, y3)
plotBoundary(model, X3)
plt.show()
