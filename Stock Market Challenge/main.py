# coding:utf-8
__author__ = 'liangz14'
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn import linear_model
import matplotlib.pyplot as plt
from xgboost.sklearn import XGBRegressor
import numpy as np



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


def xbgclassify():
    '''
    这里声明了XBG树，但是突然想到，这是无监督分类所以不能用！
    '''
    xgb = XGBRegressor(max_depth=6, learning_rate=0.3, n_estimators=25, subsample=0.5, colsample_bytree=0.5, seed=0)


def regression_lasso(train_list,target_list,test_list,mean):
    """
    直接使用数据进行预测
    :param train_list:
    :param test_list:
    :return:
    """
    clf = linear_model.LassoLars(alpha=.01)
    clf.fit(train_list, target_list)
    result_list = clf.predict(test_list)+mean
    return result_list


def data_filter(train_list, target_list, test_list):
    train_list = np.asarray(train_list, dtype=float)
    target_list = np.asarray(target_list, dtype=float)
    test_list = np.asarray(test_list, dtype=float)

    target_mean = np.mean(target_list, axis=0)
    train_mean = np.mean(train_list, axis=0)

    target_list = target_list - target_mean
    train_list = train_list - train_mean

    train_var = np.var(train_list, axis=0)

    train_list = train_list/train_var

    test_list = (test_list - train_mean)/train_var

    print train_list.shape
    print target_list.shape
    print test_list.shape

    return train_list, target_list, test_list, target_mean

def run_result():
    df_train = pd.read_csv("train.csv")
    df_test = pd.read_csv("test_2.csv")

    df_train = df_train.fillna(0)
    df_test = df_test.fillna(0)
    print len(df_train.columns)
    print len(df_test.columns)
    df_train_feature = df_train[df_train.columns[26:]]
    df_test_feature = df_test[df_test.columns[26:]]

    print len(df_train_feature.columns)
    print len(df_test_feature.columns)
    train_list = df_train[df_train_feature.columns[:121]].values.astype(np.float32)
    target_list = df_train[df_train_feature.columns[121:-2]].values.astype(np.float32)
    test_list = df_test[df_test_feature.columns[:]].values.astype(np.float32)

    train_list, target_list, test_list,mean = data_filter(train_list, target_list, test_list)

    result_list = regression_lasso(train_list, target_list, test_list, mean)

    result_list = result_list.reshape((-1,))
    print result_list.shape
    f = open('result.csv','w')
    f.write("id,Predicted\n")
    for i,v in enumerate(result_list):
        f.write("%d_%d,%f\n" % (i/62+1,i%62+1,v))

    f.close()

def show_state():
    df_train = pd.read_csv("train.csv")
    #df_test = pd.read_csv("test_2.csv")

    df_train = df_train.fillna(0)
    #df_test = df_test.fillna(0)
    #print len(df_train.columns)
    # print len(df_test.columns)

    df_train_feature = df_train[df_train.columns[28:-4]]
    print df_train_feature.columns
    data = df_train_feature.values

    data_mean = np.mean(data, axis=0)

    data_list = data - data_mean

    data_var = np.var(data_list, axis=0)

    data = data_list/data_var

    for i,d in enumerate(data):
        for j in range(1,len(d)):
            d[j] = d[j] + d[j-1]
        data[i] = d
    c = data[1:30000,:].T
    print c.shape

    plt.plot(c)
    plt.show()


show_state()
