# -*- coding: cp936 -*-
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from scipy.optimize import minimize

from sklearn.preprocessing import PolynomialFeatures

def loaddata(file, delimeter):
    data = np.loadtxt(file, delimiter=delimeter)
    print('Dimensions: ',data.shape)
    print(data[0:6,:])
    return(data)

def plotData(data, label_x, label_y, label_pos, label_neg, axes=None):
    # Get indexes for class 0 and class 1
    neg = data[:,2] == 0
    pos = data[:,2] == 1
    # ���� data[pos][:,0] ��˵����
    # ��ʹ�ò�������b��Ϊ�±��ȡ����x�е�Ԫ��ʱ�����ռ�����x������������b�ж�Ӧ�±�ΪTrue��Ԫ�ء�
    # ʹ�ò���������Ϊ�±��õ����鲻��ԭʼ���鹲�����ݿռ䣬ע�����ַ�ʽֻ��Ӧ�ڲ�������(array)��
    # ����ʹ�ò����б�list��
    
    # If no specific axes object has been passed, get the current axes.
    if axes == None:
        axes = plt.gca()
    axes.scatter(data[pos][:,0], data[pos][:,1], marker='+', c='k', s=60, linewidth=2, label=label_pos)
    axes.scatter(data[neg][:,0], data[neg][:,1], c='y', s=60, label=label_neg)
    axes.set_xlabel(label_x)
    axes.set_ylabel(label_y)
    axes.legend(frameon= True, fancybox = True);
    

data = loaddata('data/ex2data1.txt', ',')

# np.ones ��һ������ָ����������Ľṹ��[2,3]��2��3�У��ڶ���������ָ������Ԫ�����ͣ�Ĭ���Ǹ�����
# X = np.c_[np.ones(data.shape[0]), data[:,0:2]]
X = np.c_[np.ones([data.shape[0],1]), data[:,0:2]]
y = np.c_[data[:,2]]

#plotData(data, 'Exam 1 score', 'Exam 2 score', 'Admitted', 'Not admitted')
#plt.show()


def sigmoid(z):
    return(1 / (1 + np.exp(-z)))

def costFunction(theta, X, y):
    m = y.size
    h = sigmoid(X.dot(theta))
    
    J = -1*(1/m)*(np.log(h).T.dot(y)+np.log(1-h).T.dot(1-y))

    print('J:', J)
              
    if np.isnan(J[0]):
        return(np.inf)
    return(J[0])

def gradient(theta, X, y):
    m = y.size
    h = sigmoid(X.dot(theta.reshape(-1,1)))
    
    grad =(1/m)*X.T.dot(h-y)

    return(grad.flatten())

def predict(theta, X, threshold=0.5):
    p = sigmoid(X.dot(theta.T)) >= threshold
    return(p.astype('int'))

initial_theta = np.zeros(X.shape[1])
cost = costFunction(initial_theta, X, y)
grad = gradient(initial_theta, X, y)
print('Cost: \n', cost)
print('Grad: \n', grad)


res = minimize(costFunction, initial_theta, args=(X,y), method=None, jac=gradient, options={'maxiter':400})
print(res)

# Student with Exam 1 score 45 and Exam 2 score 85
# Predict using the optimized Theta values from above (res.x)
sigmoid(np.array([1, 45, 85]).dot(res.x.T))

p = predict(res.x, X)
print('p:', p)
print('Train accuracy {}%'.format(100*sum(p == y.ravel())/p.size))

# linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None)
# ��ָ���Ĵ����ڣ����ع̶���������ݡ��������ء�num�����ȼ���������������[start, stop]�С�
# ���У�����Ľ����˵���Ա��ų����⡣

# �﷨��X,Y = numpy.meshgrid(x, y)
# �����x��y�����������ĺ����������������Ǿ���
# �����X��Y�������������
# ������Ľ�����������ַ������ɵ��������һëһ����
# x = np.array([0, 1, 2])
# y = np.array([0, 1])

# X, Y = np.meshgrid(x, y)
# [[0 1 2]
# [0 1 2]]
# [[0 0 0]
# [1 1 1]]

plt.scatter(45, 85, s=60, c='r', marker='v', label='(45, 85)')
plotData(data, 'Exam 1 score', 'Exam 2 score', 'Admitted', 'Not admitted')
x1_min, x1_max = X[:,1].min(), X[:,1].max(),
x2_min, x2_max = X[:,2].min(), X[:,2].max(),
xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max), np.linspace(x2_min, x2_max))
print('xx1:', xx1.shape)
print('xx2:', xx2.shape)

h = sigmoid(np.c_[np.ones((xx1.ravel().shape[0],1)), xx1.ravel(), xx2.ravel()].dot(res.x))
h = h.reshape(xx1.shape)
plt.contour(xx1, xx2, h, [0.5], linewidths=1, colors='b');
plt.show()
