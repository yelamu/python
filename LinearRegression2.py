# -*- coding: cp936 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from mpl_toolkits.mplot3d import axes3d

def warmUpExercise():
    return(np.identity(5))

def computeCost(X, y, theta=[[0],[0]]):
    m = y.size
    J = 0
    #print(theta)
    
    h = X.dot(theta)
    #print(h[0])
    #print(h.shape)
    J = 1/(2*m)*np.sum(np.square(h-y))
    
    return(J)

def gradientDescent(X, y, theta=[[0],[0]], alpha=0.01, num_iters=1500):
    m = y.size
    J_history = np.zeros(num_iters)
    
    for iter in np.arange(num_iters):
        h = X.dot(theta)
        theta = theta - alpha*(1/m)*(X.T.dot(h-y))
        J_history[iter] = computeCost(X, y, theta)
    return(theta, J_history)

#print(warmUpExercise())

data = np.loadtxt('data/ex1data1.txt', delimiter=',')
#print(data)
print(data.shape)

# 数组连接成矩阵
#   np.c_  : 按列转换成矩阵
#   np.r_  : 按行转换成矩阵
X = np.c_[np.ones(data.shape[0]),data[:,0]]
print(X.shape)
y = np.c_[data[:,1]]

# theta for minimized cost J
theta , Cost_J = gradientDescent(X, y)

# Create grid coordinates for plotting
B0 = np.linspace(-10, 10, 50)
B1 = np.linspace(-1, 4, 50)
xx, yy = np.meshgrid(B0, B1, indexing='xy')
print(xx.shape)
Z = np.zeros((B0.size,B1.size))

# Calculate Z-values (Cost) based on grid of coefficients
for (i,j),v in np.ndenumerate(Z):
    Z[i,j] = computeCost(X,y, theta=[[xx[i,j]], [yy[i,j]]])

fig = plt.figure(figsize=(15,6))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122, projection='3d')

# Left plot
CS = ax1.contour(xx, yy, Z, np.logspace(-2, 3, 20), cmap=plt.cm.jet)
ax1.scatter(theta[0],theta[1], c='r')

# Right plot
ax2.plot_surface(xx, yy, Z, rstride=1, cstride=1, alpha=0.6, cmap=plt.cm.jet)
ax2.set_zlabel('Cost')
ax2.set_zlim(Z.min(),Z.max())
ax2.view_init(elev=15, azim=230)

# settings common to both plots
for ax in fig.axes:
    ax.set_xlabel(r'$\theta_0$', fontsize=17)
    ax.set_ylabel(r'$\theta_1$', fontsize=17)

plt.show()
