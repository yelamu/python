# -*- coding: cp936 -*-
# 这我的一个练习画图的，和作业无关，给个画图的参考。
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from scipy.io import loadmat

# we create 40 separable points
np.random.seed(0)


mat = loadmat('./data/ex6data1.mat')

X = mat['X']
Y = mat['y']

# fit the model
clf = svm.SVC(kernel='linear')
clf.fit(X, Y)

# get the separating hyperplane
w = clf.coef_[0]
a = -w[0] / w[1]
xx = np.linspace(0, 5)
yy = a * xx - (clf.intercept_[0]) / w[1]

# plot the parallels to the separating hyperplane that pass through the
# support vectors
b = clf.support_vectors_[0]
yy_down = a * xx + (b[1] - a * b[0])
b = clf.support_vectors_[10]
yy_up = a * xx + (b[1] - a * b[0])

# plot the line, the points, and the nearest vectors to the plane
plt.figure(figsize=(8,5))
plt.plot(xx, yy, 'k-')
plt.plot(xx, yy_down, 'k--')
plt.plot(xx, yy_up, 'k--')
# 圈出支持向量
plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1],
            s=150, facecolors='none', edgecolors='k', linewidths=1.5)
plt.scatter(X[:, 0], X[:, 1], c=Y.flatten(), cmap=plt.cm.rainbow)

plt.axis('tight')
plt.show()

print("clf:{}".format(clf))
print("clf.coef_:{}".format(clf.coef_))
print("clf.support_vectors_:{}".format(clf.support_vectors_))
print("clf.intercept_:{}".format(clf.intercept_))

