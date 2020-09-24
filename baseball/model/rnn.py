# import libararies
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt

import statsmodels.api as sm


import warnings
warnings.filterwarnings("ignore")

# data preparation
# Data Source : http://www.statiz.co.kr/
pitcher_file_path = './baseball/data/pitcher_stats_2017.csv'
batter_file_path = './baseball/data//batter_stats_2017.csv'

pitcher = pd.read_csv(pitcher_file_path)
pbatter = pd.read_csv(batter_file_path)


# step1 : explore
print(pitcher.columns)

print("pitcher shape :", pitcher.shape)

print(pitcher['연봉(2018)'].describe())

pitcher['연봉(2018)'].hist(bins=100)
# plt.show()

# font setting
set(sorted([f.name for f in mpl.font_manager.fontManager.ttflist]))

# mpl.rc('font', family='HYGothic-Medium')

# pitcher.boxplot(column=['연봉(2018)'])
# plt.show()

# check the features
pitcher_features_df = pitcher[['승', '패', '세', '홀드', '블론', '경기', '선발', '이닝', '삼진/9', '볼넷/9', '홈런/9', 
                               'BABIP', 'LOB%', 'ERA', 'RA9-WAR', 'FIP', 'kFIP', 'WAR', '연봉(2018)', '연봉(2017)']]

# print histograms of each column
def plot_hist_each_column(df):
    plt.rcParams['figure.figsize'] = [20, 16]
    fig = plt.figure(1)
    
    # print subplot (the number of columns)
    for i in range(len(df.columns)):
        ax = fig.add_subplot(5, 5, i+1)
        plt.hist(df[df.columns[i]], bins=50)
        ax.set_title(df.columns[i], fontdict={'fontsize': 15, 'fontweight' : 'medium'})
    fig.tight_layout()
    # plt.show()

plot_hist_each_column(pitcher_features_df)

# plt.show()

# step2 : predict
pd.options.mode.chained_assignment = None

# for feature scaling
def standard_scaling(df, scale_columns):
    for col in scale_columns:
        series_mean = df[col].mean()
        series_std = df[col].std()
        df[col] = df[col].apply(lambda x: (x-series_mean)/series_std)
        
    return df

scale_columns = ['승', '패', '세', '홀드', '블론', '경기', '선발', '이닝', '삼진/9', '볼넷/9', 
                 '홈런/9', 'BABIP', 'LOB%', 'ERA', 'RA9-WAR', 'FIP', 'kFIP', 'WAR', '연봉(2017)']

pitcher_df = standard_scaling(pitcher, scale_columns)
pitcher_df = pitcher_df.rename(columns = {'연봉(2018)': 'y'})
print(pitcher_df.head(5))
print("-------------------------------------------------------------------------------------------------------------------------------")



# one-hot encoding (team name)
team_encoding = pd.get_dummies(pitcher_df['팀명'])
print(team_encoding.head(5))
print()
print("-------------------------------------------------------------------------------------------------------------------------------")
print()
pitcher_df = pitcher_df.drop('팀명', axis = 1) # axis = 1 : column
print(pitcher_df)
pitcher_df = pitcher_df.join(team_encoding)
print()
print("-------------------------------------------------------------------------------------------------------------------------------")
print()
print(pitcher_df)
print("-------------------------------------------------------------------------------------------------------------------------------")


# split data
x = pitcher_df[pitcher_df.columns.difference(['선수명', 'y'])]
y = pitcher_df['y']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 19)

print("x_train shape :", x_train.shape) # x_train shape : (121, 28)
print("x_test shape :", x_test.shape) # x_test shape : (31, 28)
print("y_train shape :", y_train.shape) # y_train shape : (121,)
print("y_test shape :", y_test.shape) # y_test shape : (31,)


x = pitcher_df[['FIP', 'WAR', '볼넷/9', '삼진/9', '연봉(2017)']]
y = pitcher_df['y']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=19)\

print("x_train.shaep :", x_train.shape)
print("y_train.shaep :", y_train.shape)
print("x_test.shaep :", x_test.shape)
print("y_test.shaep :", y_test.shape)





from keras.models import Model
from keras.layers import Dense, Dropout, Input



from sklearn.metrics import r2_score
from sklearn.utils.testing import all_estimators 

allAlgorithms = all_estimators(type_filter = 'regressor') 


for (name, algorithm) in allAlgorithms : 
    model = algorithm()

    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    print(name, "의 정답률 : ", r2_score(y_test, y_pred))

import sklearn
print(sklearn.__version__)


from sklearn.metrics import accuracy_score
y_pred = model.predict(x_test)

model.fit(x_train, y_train)
r2 = model.score(x_test, y_test)

print("r2score : ", r2_score(y_test, y_pred))
print("r2       : ", r2)


'''
inputs = Input(shape = (5, ))

x = Dense(64)(inputs)
x = Dense(128)(x)
x = Dense(256)(x)
x = Dense(512)(x)
x = Dense(1024)(x)
x = Dense(512)(x)
x = Dense(256)(x)
x = Dense(128)(x)
x = Dropout(0.4)(x)
x = Dense(16, activation = 'relu')(x)

outputs = Dense(1)(x)

model = Model(inputs = inputs, outputs = outputs)

model.summary()

model.compile(loss = 'mse', optimizer = 'adam', metrics = ['mse'])
model.fit(x_train, y_train, epochs = 128, batch_size = 8, validation_split = 0.3, verbose = 1)

# 4. evaluate, predict
loss, mse = model.evaluate(x_test, y_test, batch_size = 1)

print("LOSS :", loss)
print("MSE : ", mse)
'''