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

print("x_train shape :", x_train.shape)
print("x_test shape :", x_test.shape)
print("y_train shape :", y_train.shape)
print("y_test shape :", y_test.shape)


x_train = sm.add_constant(x_train)
model = sm.OLS(y_train, x_train).fit()
model.summary()

# feature_coef_graph

# mpl.rc('font', family='HYGothic-Medium')
plt.rcParams['figure.figsize'] = [20, 16]

# return to list
coefs = model.params.tolist()
coefs_series = pd.Series(coefs)

x_labels = model.params.index.tolist()

ax = coefs_series.plot(kind='bar')
ax.set_title('feature_coef_graph')
ax.set_xlabel('x_features')
ax.set_ylabel('coef')
ax.set_xticklabels(x_labels)

# evaluate 'R2'

x = pitcher_df[pitcher_df.columns.difference(['선수명', 'y'])]
y = pitcher_df['y']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 19)

lr = linear_model.LinearRegression()
model = lr.fit(x_train, y_train)

print("R2 score(train) :", model.score(x_train, y_train))
print()
print("R2 score(test) :", model.score(x_test, y_test))
print("-------------------------------------------------------------------------------------------------------------------------------")


# evaluate 'RMSE
y_prediction = lr.predict(x_train)
print("RMSE score(train) :",sqrt(mean_squared_error(y_train, y_prediction)))
print()
y_prediction = lr.predict(x_test)
print("RMSE score(test) :",sqrt(mean_squared_error(y_test, y_prediction)))



import seaborn as sns

corr = pitcher_df[scale_columns].corr(method='pearson')
show_cols = ['win', 'lose', 'save', 'hold', 'blon', 'match', 'start', 'inning', 'strike3', 
             'ball4', 'homerun', 'BABIP', 'LOB', 'ERA', 'RA9-WAR', 'FIP', 'kFIP', 'WAR', '2017']

# visualize heatmap
# plt.rc('font', family='HYGothic-Medium')
sns.set(font_scale = 1.5)

hm = sns.heatmap(corr.values, cbar = True, annot = True, square = True, fmt = '.2f', 
                 annot_kws = {'size':15}, yticklabels=show_cols, xticklabels=show_cols)

plt.tight_layout()
# plt.show()


# check Multicollinearity with vif(variance inflation factor)
from statsmodels.stats.outliers_influence import variance_inflation_factor

vif = pd.DataFrame()
vif['VIF Factor'] = [variance_inflation_factor(x.values, i) for i in range(x.shape[1])]
vif['features'] = x.columns
vif.round(1)

# feature selection, train again

x = pitcher_df[['FIP', 'WAR', '볼넷/9', '삼진/9', '연봉(2017)']]
y = pitcher_df['y']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=19)


# R2
lr = linear_model.LinearRegression()
model = lr.fit(x_train, y_train)

print("R2 score(train) :", model.score(x_train, y_train))
print("R2 score(test) :", model.score(x_test, y_test))
print()

# RMSE
y_prediction = lr.predict(x_train)
print("RMSE score(train) :",sqrt(mean_squared_error(y_train, y_prediction)))
y_prediction = lr.predict(x_test)
print("RMSE score(test) :",sqrt(mean_squared_error(y_test, y_prediction)))


# VIF

x = pitcher_df[['FIP', 'WAR', '볼넷/9', '삼진/9', '연봉(2017)']]
vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(x.values, i) for i in range(x.shape[1])]
vif["features"] = x.columns
vif.round(1)


# visualize result ( predict vs real )
# player - real 2018 - predict 2018 - 2017

x = pitcher_df[['FIP', 'WAR', '볼넷/9', '삼진/9', '연봉(2017)']]
predict_2018_salary = lr.predict(x)
pitcher_df['예측연봉(2018)'] = pd.Series(predict_2018_salary)


# read the original data
pitcher = pd.read_csv(pitcher_file_path)
pitcher = pitcher[['선수명', '연봉(2017)']]

# join
result_df = pitcher_df.sort_values(by = ['y'], ascending=False)
result_df.drop(['연봉(2017)'], axis = 1, inplace=True, errors='ignore')
result_df = result_df.merge(pitcher, on=['선수명'], how='left')
result_df = result_df[['선수명', 'y', '예측연봉(2018)', '연봉(2017)']]
result_df.columns = ['선수명', '실제연봉(2018)', '예측연봉(2018)', '작년연봉(2017)']

result_df = result_df[result_df['작년연봉(2017)'] != result_df['실제연봉(2018)']]
result_df = result_df.reset_index()
result_df = result_df.iloc[:10, :]
print(result_df.head(10))

# graph
# mpl.rc('font', family = 'HYGothic-Medium')
result_df.plot(x='선수명', y=['작년연봉(2017)', '예측연봉(2018)', '실제연봉(2018)'], kind = 'bar')
