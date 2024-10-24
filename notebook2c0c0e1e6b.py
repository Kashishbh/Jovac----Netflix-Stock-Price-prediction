
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PowerTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import warnings

df = pd.read_csv('NFLX.csv')

df.shape

df.head()

df.isnull().sum().any()

df.info()

df.describe()

df.columns = df.columns.str.lower()

df[['year', 'month', 'day']] = df['date'].apply(lambda x: pd.Series(map(int, x.split('-'))))

df.drop(['date'], axis = 1, inplace = True)
df.head()

df.hist(figsize = (15,9), bins = 50)
plt.show()

sns.heatmap(df.corr(), annot = True)
plt.show()

power = PowerTransformer('yeo-johnson')

transformed_df = power.fit_transform(df)
transformed_df = pd.DataFrame(transformed_df, columns = df.columns)

transformed_df.hist(figsize = (15,9), bins = 50)
plt.show()

transformed_df.columns

model_r = RandomForestRegressor(n_estimators=150, random_state = 42)
sfs = SFS(estimator=model_r, k_features='best', forward= True, floating = True, scoring = 'neg_mean_squared_error', cv = 12)

xtrain, xtest, ytrain, ytest = train_test_split(transformed_df.drop(['volume'], axis = 1), transformed_df.volume, test_size = 0.25, random_state = 42, shuffle = True)

sfs.fit(xtrain, ytrain)

list(sfs.k_feature_names_)

x = transformed_df[list(sfs.k_feature_names_)]
y = transformed_df.volume

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size = 0.2, random_state = 42, shuffle = True)

model_lin = LinearRegression()

model_lin.fit(xtrain, ytrain)

model_rand = RandomForestRegressor(n_estimators=200, random_state = 42)

model_rand.fit(xtrain, ytrain)

pred_lin = model_lin.predict(xtest)
pred_rand = model_rand.predict(xtest)

print(f'LINEAR REGRESSION r2_score : {r2_score(ytest,pred_lin)}')
print(f'RANDOM FOREST r2_score : {r2_score(ytest,pred_rand)}')
print()
print(f'LINEAR REGRESSION mean_absolute_error : {mean_absolute_error(ytest,pred_lin)}')
print(f'RANDOM FOREST mean_absolute_error : {mean_absolute_error(ytest,pred_rand)}')

