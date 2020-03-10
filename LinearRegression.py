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
        print("theta %d; J_history[iter]:%f" %(iter, J_history[iter]), theta.ravel())
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

#plt.scatter(X[:,1], y, s=10, c='r', marker='x', linewidths=1)
##plt.scatter(data[:,0], data[:,1], s=10, c='r', marker='x', linewidths=1)
#plt.xlim(4,24)
#plt.xlabel('Population of City in 10,000s')
#plt.ylabel('Profit in $10,000s');
#plt.show()

cost = computeCost(X,y)
print("cost:", cost)

h2 = np.zeros(data.shape[0])
#print(h2[0])
#print(h2.shape)
h2 = h2.reshape(h2.size, 1)
#print(h2[0])
#print(h2.shape)
#print(y.shape)
cost2 = 1/(2*y.size)*np.sum(np.square(h2-y))
print("cost2:", cost2)

# theta for minimized cost J
theta , Cost_J = gradientDescent(X, y)
print('theta: ',theta.ravel())
print('Cost_J: ',Cost_J.ravel())

exit()

#plt.plot(Cost_J)
#plt.ylabel('Cost J')
#plt.xlabel('Iterations');
#plt.show()

xx = np.arange(5,23)
yy = theta[0]+theta[1]*xx

# Plot gradient descent
#print(X[:,1].reshape(-1,1))
plt.scatter(X[:,1], y, s=30, c='r', marker='x', linewidths=1)
plt.plot(xx,yy, label='Linear regression (Gradient descent)')

# Compare with Scikit-learn Linear regression 
regr = LinearRegression()
regr.fit(X[:,1].reshape(-1,1), y.ravel())
plt.plot(xx, regr.intercept_+regr.coef_*xx, label='Linear regression (Scikit-learn GLM)')

plt.xlim(4,24)
plt.xlabel('Population of City in 10,000s')
plt.ylabel('Profit in $10,000s')
plt.legend(loc=4);
plt.show()
