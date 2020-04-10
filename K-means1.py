# -*- coding: cp936 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat

def findClosestCentroids(X, centroids):
    """
    output a one-dimensional array idx that holds the 
    index of the closest centroid to every training example.
    """
    idx = []
    max_dist = 1000000  # ����һ��������
    for i in range(len(X)):
        minus = X[i] - centroids  # here use numpy's broadcasting
        dist = minus[:,0]**2 + minus[:,1]**2
        #print('i={}; X[i]:{}; minus:{}; dist:{}'.format(i, X[i], minus, dist))
        if dist.min() < max_dist:
            ci = np.argmin(dist)
            idx.append(ci)
    return np.array(idx)

mat = loadmat('data/ex7data2.mat')
# print(mat.keys())
X = mat['X']
print('X shape:{}\nsample data:{}'.format(X.shape, X[0:6]))

init_centroids = np.array([[3, 3], [6, 2], [8, 5]])
idx = findClosestCentroids(X, init_centroids)

print('idx shape:{}\nsample idx:{}'.format(idx.shape, idx[0:6]))

def computeCentroids(X, idx):
    centroids = []
    for i in range(len(np.unique(idx))):  # np.unique() means K
        u_k = X[idx==i].mean(axis=0)  # ��ÿ�е�ƽ��ֵ
        centroids.append(u_k)
        
    return np.array(centroids)

tmpCentroid = computeCentroids(X, idx)
print('tmpCentroid shape:{}\ntmpCentroid data:{}'.format(tmpCentroid.shape, tmpCentroid))


def plotData(X, centroids, idx=None):
    """
    ���ӻ����ݣ����Զ��ֿ���ɫ��
    idx: ���һ�ε������ɵ�idx�������洢ÿ����������Ĵ����ĵ��ֵ
    centroids: ����ÿ�����ĵ���ʷ��¼
    """
    colors = ['b','g','gold','darkorange','salmon','olivedrab', 
              'maroon', 'navy', 'sienna', 'tomato', 'lightgray', 'gainsboro'
             'coral', 'aliceblue', 'dimgray', 'mintcream', 'mintcream']
    
    assert len(centroids[0]) <= len(colors), 'colors not enough '
      
    subX = []  # �ֺ����������
    if idx is not None:
        for i in range(centroids[0].shape[0]):
            x_i = X[idx == i]
            subX.append(x_i)

    else:
        subX = [X]  # ��Xת��Ϊһ��Ԫ�ص��б�ÿ��Ԫ��Ϊÿ���ص��������������·���ͼ
    
    # �ֱ𻭳�ÿ���صĵ㣬���Ų�ͬ����ɫ
    plt.figure(figsize=(8,5))    
    for i in range(len(subX)):
        xx = subX[i]
        plt.scatter(xx[:,0], xx[:,1], c=colors[i], label='Cluster %d'%i)
    plt.legend()
    plt.grid(True)
    plt.xlabel('x1',fontsize=14)
    plt.ylabel('x2',fontsize=14)
    plt.title('Plot of X Points',fontsize=16)
    
    # ���������ĵ���ƶ��켣
    xx, yy = [], []
    for centroid in centroids:
        xx.append(centroid[:,0])
        yy.append(centroid[:,1])
    
    plt.plot(xx, yy, 'rx--', markersize=8)
                         
#plotData(X, [init_centroids])
#plt.show()

def runKmeans(X, centroids, max_iters):
    K = len(centroids)
    
    centroids_all = []
    centroids_all.append(centroids)
    centroid_i = centroids
    for i in range(max_iters):
        idx = findClosestCentroids(X, centroid_i)
        centroid_i = computeCentroids(X, idx)
        centroids_all.append(centroid_i)
    
    return idx, centroids_all

#idx, centroids_all = runKmeans(X, init_centroids, 20)
#plotData(X, centroids_all, idx)
#plt.show()

#centroids_all = np.array(centroids_all)
#print('centroids_all shape:{}\ncentroids_all data:{}'.format(centroids_all.shape, centroids_all))

def initCentroids(X, K):
    """�����ʼ��"""
    m, n = X.shape
    idx = np.random.choice(m, K)
    centroids = X[idx]
    
    return centroids 

for i in range(3):
    centroids = initCentroids(X, 3)
    idx, centroids_all = runKmeans(X, centroids, 10)
    plotData(X, centroids_all, idx)
    plt.show()

