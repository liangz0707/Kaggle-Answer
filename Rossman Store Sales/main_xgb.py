# coding:utf-8
__author__ = 'liangz14'
import xgboost as xgb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from xgboost.sklearn import XGBRegressor

np.random.seed(0)

#Loading data
df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')

Sales = df_train.Sales.apply(lambda m : float(m)).values
df_train = df_train.drop(['Sales'], axis=1)

ids = df_test["Id"].values
df_test = df_test.drop(["Id"], axis=1)

piv_train = df_train.shape[0]
df_all = pd.concat((df_train, df_test), axis=0, ignore_index=True)

print df_train.columns


#处理特征
#将账户日期拆解成三个
dac = np.vstack(
      df_all.Date\
            .astype(str)\
            .apply(lambda x: list(map(int, x.split('-'))))\
            .values
    )
df_all = df_all.drop(['Date'], axis=1)
df_all['year'] = dac[:,0]
df_all['month'] = dac[:,1]
df_all['day'] = dac[:,2]

#处理store
#为整形，不需要处理

#处理DayOfWeek
#为整形，不需要处理

#StateHoliday
def c(x):
    if x=='0':
        x=0
    elif x == 'a':
        x=3
    elif x =='b':
        x=2
    elif x =='c':
        x=1

StateHoliday = df_all.StateHoliday.astype(str).apply(c).values
df_all.drop(["StateHoliday"], axis=1)
df_all["StateHoliday"] = StateHoliday


vals = df_all.values

print df_all.columns

X = vals[:piv_train]
y = Sales
X_test = vals[piv_train:]


xgb = XGBRegressor(max_depth=6, learning_rate=0.3, n_estimators=25, subsample=0.5, colsample_bytree=0.5, seed=0)
xgb.fit(X, y)

y_pred = xgb.predict(X_test)
sub = pd.DataFrame(np.column_stack((ids, y_pred)), columns=['Id', 'Sales'])
sub.to_csv('sub.csv',index=False)
