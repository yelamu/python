# %load ../../standard_import.txt
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# load MATLAB files
from scipy.io import loadmat
from scipy.optimize import minimize

from sklearn.linear_model import LogisticRegression

data = loadmat('data/ex3data1.mat')
print(data.keys())
print('__header__:', data['__header__'])
print('__version__:', data['__version__'])
print('__globals__:', data['__globals__'][1:6])
print('X:', data['X'].shape)
print(data['X'][1:6])
print('y:', data['y'].shape)
print(data['y'][1:6])
print(np.unique(data['y']))




weights = loadmat('data/ex3weights.mat')
print(weights.keys())
print('__header__:', weights['__header__'])
print('__version__:', weights['__version__'])
print('__globals__:', weights['__globals__'][1:6])
print('Theta1:', weights['Theta1'].shape)
print(weights['Theta1'][1:6])
print('Theta2:', weights['Theta2'].shape)
print(weights['Theta2'][1:6])

y = data['y']
# Add constant for intercept
X = np.c_[np.ones((data['X'].shape[0],1)), data['X']]

print('X: {} (with intercept)'.format(X.shape))
print('y: {}'.format(y.shape))

theta1, theta2 = weights['Theta1'], weights['Theta2']

print('theta1: {}'.format(theta1.shape))
print('theta2: {}'.format(theta2.shape))

sample = np.random.choice(X.shape[0], 20)
plt.imshow(X[sample,1:].reshape(-1,20).T)
plt.axis('off');
#plt.show()



def sigmoid(z):
    return(1 / (1 + np.exp(-z)))

def lrcostFunctionReg(theta, reg, X, y):
    m = y.size
    h = sigmoid(X.dot(theta))
    
    J = -1*(1/m)*(np.log(h).T.dot(y)+np.log(1-h).T.dot(1-y)) + (reg/(2*m))*np.sum(np.square(theta[1:]))
    
    if np.isnan(J[0]):
        return(np.inf)
    return(J[0])

def lrgradientReg(theta, reg, X,y):
    m = y.size
    h = sigmoid(X.dot(theta.reshape(-1,1)))
      
    grad = (1/m)*X.T.dot(h-y) + (reg/m)*np.r_[[[0]],theta[1:].reshape(-1,1)]
        
    return(grad.flatten())

def oneVsAll(features, classes, n_labels, reg):
    initial_theta = np.zeros((X.shape[1],1))  # 401x1
    all_theta = np.zeros((n_labels, X.shape[1])) #10x401

    for c in np.arange(1, n_labels+1):
        res = minimize(lrcostFunctionReg, initial_theta, args=(reg, features, (classes == c)*1), method=None,
                       jac=lrgradientReg, options={'maxiter':50})
        all_theta[c-1] = res.x
    return(all_theta)

def predictOneVsAll(all_theta, features):
    probs = sigmoid(X.dot(all_theta.T))
        
    # Adding one because Python uses zero based indexing for the 10 columns (0-9),
    # while the 10 classes are numbered from 1 to 10.
    return(np.argmax(probs, axis=1)+1)

# One-vs-all Prediction
theta = oneVsAll(X, y, 10, 0.1)
print('theta:', theta.shape)
pred = predictOneVsAll(theta, X)
print('Training set accuracy: {} %'.format(np.mean(pred == y.ravel())*100))

# Multiclass Logistic Regression with scikit-learn
clf = LogisticRegression(C=10, penalty='l2', solver='liblinear', multi_class='auto')
# Scikit-learn fits intercept automatically, so we exclude first column with 'ones' from X when fitting.
clf.fit(X[:,1:],y.ravel())

pred2 = clf.predict(X[:,1:])
print('clf:', clf)
print('Training set accuracy: {} %'.format(np.mean(pred2 == y.ravel())*100))


# Neural Networks
def predict(theta_1, theta_2, features):
    z2 = theta_1.dot(features.T)
    a2 = np.c_[np.ones((data['X'].shape[0],1)), sigmoid(z2).T]
    
    z3 = a2.dot(theta_2.T)
    a3 = sigmoid(z3)
        
    return(np.argmax(a3, axis=1)+1)

pred = predict(theta1, theta2, X)
print('Training set accuracy: {} %'.format(np.mean(pred == y.ravel())*100))



