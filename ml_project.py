import time
import json
import numpy as np
import pandas as pd
from pandas import json_normalize

import datetime

import streamlit as st
import lightgbm as lgb

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error

import seaborn as sns
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')
st.set_option('deprecation.showPyplotGlobalUse', False)

st.title('Customer Revenue Prediction')
st.header('1. Data Prepration')
st.write('a. Loading libraries')
def load_df(csv_path, nrows = None):
    json_cols = ['device', 'geoNetwork', 'totals', 'trafficSource']
    df = pd.read_csv(csv_path,
                     #converters are dict of functions for converting values in certain columns. Keys can either be integers or column labels.
                     #json.loads() method can be used to parse a valid JSON string and convert it into a Python Dictionary.
                     #It is mainly used for deserializing native string, byte, or byte array which consists of JSON data into Python Dictionary.
                     converters = {col: json.loads for col in json_cols},                                                                         
                         dtype = {'fullVisitorId': 'str'}, # Important!!
                         nrows = nrows)
    for col in json_cols:
        # for each column, flatten data frame such that the values of a single col are spread in different cols
        # This will use subcol as names of flat_col.columns
        flat_col = json_normalize(df[col])
        # Name the columns in this flatten data frame as col.subcol for tracability
        flat_col.columns = [f"{col}.{subcol}" for subcol in flat_col.columns]
        # Drop the json_col and instead add the new flat_col
        df = df.drop(col, axis = 1).merge(flat_col, right_index = True, left_index = True)
    return df

st.write('b. Loading the dataset')

csv_train_path = 'train_v2.csv'
csv_test_path = 'test_v2.csv'


train = load_df(csv_train_path, nrows = 20000)
test = load_df(csv_test_path, nrows = 200)
cols = train.columns

train['totals.transactionRevenue'] = train['totals.transactionRevenue'].astype('float')
#To have it as a data frame with fullVisitorId and totals.transactionRevenue we need to reset_index(). Otherwise, it would return a serie
target = train.groupby('fullVisitorId')['totals.transactionRevenue'].sum().reset_index()

st.subheader('Dataset')
st.dataframe(train.head())
st.dataframe(test.head())

st.header('2. Data Summary')
st.subheader('a. Visualization for users generating revenue (target = 1)')
ax = sns.countplot(pd.cut(target['totals.transactionRevenue'],[-1,0,1e9]))
ax.set_xticklabels(['Users with 0$ transaction','Users with positive transacton'])
ax.set_xlabel('Revenue', fontsize = 14)
st.pyplot()
st.write("A very low percentage of users contribute in generating revenue. Let's have a closer look in them. To see how many they are and how do they contribute to the revenue generation.")

temp_rev = target[target['totals.transactionRevenue'] != 0]
st.write('number of users with transaction:',len(temp_rev))
st.write('number of total users:',len(target))
st.write('number of total transactions:', len(train))
st.write(np.round(np.sum(target['totals.transactionRevenue'] != 0)*100 / len(target),2), 'percent of users generate revenue!!')

st.subheader('b. Total sum of transaction from users contributing to revenue generation per count')
ax = sns.countplot(pd.cut(temp_rev['totals.transactionRevenue'],[0,1e7,0.5e8,1e9]))
ax.set_xticklabels(['0 - 10M','10M - 50M','50M - 1B'])
ax.set_xlabel('Total sum of transaction from users contributing to revenue generation')
st.pyplot()
st.write('16 out of 24 users have had a transaction between 10-50M dollars')

st.subheader('c. Visualizing missing values')
missing_val = pd.DataFrame()
for col in cols:
    na_count = train[col].isna().sum()
    if na_count != 0:
        missing_val.loc[col, 'NaN_val(%)'] = na_count/len(train)*100
missing_val.sort_values('NaN_val(%)', inplace = True)
st.write('Number of columns with missing values in train set:', len(missing_val))


fig, ax = plt.subplots(figsize = (12, 8))
sns.barplot(data = missing_val, x = 'NaN_val(%)', y = missing_val.index , ax = ax)
ax.set_xlabel('Missing Data (%)')
st.pyplot()

st.write('We see some columns with more than 50% missing values. We will later remove them in data cleaning.')

const_cols = []
for col in cols:
     if train[col].nunique() == 1: const_cols.append(col)


st.write('Number of columns with constant values in train set:', len(const_cols))

y_train_ = train['totals.transactionRevenue']
y_train_.fillna(0, inplace = True)
y_train_.astype('float')

for col in missing_val.index:
    if missing_val.loc[col, 'NaN_val(%)'] > 48: 
        train.drop(col, axis = 1, inplace = True)
    else: 
        train[col].fillna('0', inplace = True)
        test[col].fillna('0', inplace = True)

for col in const_cols:
    if col not in missing_val.index:
        train.drop(col, axis = 1, inplace = True)

irrelavant = ['fullVisitorId', 'visitId', 'trafficSource.campaign']
for col in irrelavant:
    train.drop(col, axis = 1, inplace = True)

for df in [train, test]:
    df['visitStartTime'] = pd.to_datetime(df['visitStartTime'], unit='s')
    df['rec_dayofweek'] = df['visitStartTime'].dt.dayofweek
    df['rec_hours'] = df['visitStartTime'].dt.hour
    df['rec_dayofmonth'] = df['visitStartTime'].dt.day
    df.drop('visitStartTime', axis = 1, inplace = True)

le = LabelEncoder()
for col in train.columns:
    if train[col].dtype == 'O':
        train.loc[:, col] = le.fit_transform(train.loc[:, col])
        test.loc[:, col] = le.fit_transform(test.loc[:, col])

for col in train.columns:
    train[col] = train[col].astype('float')

fullvisitorid = []
for col in test.columns:
    if col == 'fullVisitorId':
        fullvisitorid = test[col] 
    if col not in train.columns:
        test.drop(col, axis = 1, inplace = True)

for col in test.columns:
    test[col] = test[col].astype('float')



model = lgb.LGBMRegressor(
        num_leaves = 31,  #(default = 31) – Maximum tree leaves for base learners.
        learning_rate = 0.03, #(default = 0.1) – Boosting learning rate. You can use callbacks parameter of fit method to shrink/adapt learning rate in training using 
                              #reset_parameter callback. Note, that this will ignore the learning_rate argument in training.
        n_estimators = 1000, #(default = 100) – Number of boosted trees to fit.
        subsample = .9, #(default = 1.) – Subsample ratio of the training instance.
        colsample_bytree = .9, #(default = 1.) – Subsample ratio of columns when constructing each tree
        random_state = 34
)

st.header("3. Building The Model")
st.code("model = lgb.LGBMRegressor(num_leaves = 31,  learning_rate = 0.03,n_estimators = 1000, subsample = .9, colsample_bytree = .9,random_state = 34)")

x_train = train[ train['date'] < 20171101 ]
x_valid = train[ train['date'] >= 20171101 ]

y_train_len = len(x_train)
y_train = y_train_[:y_train_len]
y_valid = y_train_[y_train_len:]

x_train.drop('date', axis = 1, inplace = True)
x_valid.drop('date', axis = 1, inplace = True)

model.fit(
        x_train, np.log1p(y_train),
        eval_set = [(x_valid, np.log1p(y_valid))],
        early_stopping_rounds = 50,
        verbose = 100,
        eval_metric = 'rmse'
    )

st.header("4. Important Features")
feat_impr = pd.DataFrame()
feat_impr['feature'] = x_train.columns
feat_impr['importance'] = model.booster_.feature_importance(importance_type = 'gain')
feat_impr.sort_values(by = 'importance', ascending = False)[:10]

plt.figure(figsize = (8,5))
sns.barplot(x = 'importance', y = 'feature', data = feat_impr.sort_values('importance', ascending = False)[:15])
st.pyplot()

valid_preds = model.predict(x_valid, num_iteration = model.best_iteration_)
valid_preds[valid_preds < 0] = 0

test_preds = model.predict(test[x_train.columns], num_iteration = model.best_iteration_)
test_preds[test_preds < 0] = 0

rec_submit = pd.concat([fullvisitorid, pd.Series(test_preds)], axis = 1)
rec_submit.columns = ['fullVisitorId','PredictedLogRevenue']

user_submit = rec_submit.groupby('fullVisitorId').sum().reset_index()
st.header("5. Final Prediction")
st.dataframe(user_submit)