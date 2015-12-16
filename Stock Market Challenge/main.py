# coding:utf-8
__author__ = 'liangz14'
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from xgboost.sklearn import XGBRegressor
import numpy as np

df_train = pd.read_csv("train.csv")
df_test = pd.read_csv("test.csv")

df_train = df_train.fillna(0)
df_train_feature = df_train[df_train.columns[1:26]]

#print df_train_feature
#print df_train_feature.columns
#print df_train_feature.size

c = df_train_feature.values

def KMeans_classify():
    random_state = 170
    '''
        这里使用了简单的KMeans来进行分类，但是数据,各个维度的权重差异太大
        ，所以分类结果几乎会由单一维度决定。
    '''
    y_pred = KMeans(n_clusters=3, random_state=random_state).fit_predict(c)

    pca = PCA(n_components=2)
    c_2D = pca.fit_transform(c).T

    mask = y_pred==0
    print mask
    print list(c_2D[0][mask])
    plt.scatter(list(c_2D[0][mask]),list(c_2D[1][mask]),c='r')

    mask = y_pred==1
    print mask
    print list(c_2D[0][mask])
    plt.scatter(list(c_2D[0][mask]),list(c_2D[1][mask]),c='g')

    mask = y_pred==2
    print mask
    print list(c_2D[0][mask])
    plt.scatter(list(c_2D[0][mask]),list(c_2D[1][mask]),c='b')

    plt.show()
'''
    这里声明了XBG树，但是突然想到，这是无监督分类所以不能用！
'''
def xbgclassify():
    xgb = XGBRegressor(max_depth=6, learning_rate=0.3, n_estimators=25, subsample=0.5, colsample_bytree=0.5, seed=0)