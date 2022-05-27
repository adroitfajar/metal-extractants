# -*- coding: utf-8 -*-
"""
Created on 2/22/2022

Authors:

    (1) Adroit T.N. Fajar, Ph.D.
        JSPS Postdoctoral Fellow | 日本学術振興会外国人特別研究員
        Department of Applied Chemistry, Graduate School of Engineering, Kyushu University
        744 Motooka, Nishi-ku, Fukuoka 819-0395, Japan
        Email: fajar.toriq.nur.adroit.009@m.kyushu-u.ac.jp / adroit.fajar@gmail.com
        Scopus Author ID: 57192386143
        Google Scholar: https://scholar.google.com/citations?user=o6jQEEMAAAAJ&hl=en&oi=ao
        ResearchGate: https://www.researchgate.net/profile/Adroit-Fajar
        
    (2) Aditya Dewanto Hartono, Ph.D.
        Postdoctoral Fellow
        Mathematical Modeling Laboratory
        Center for Promotion of International Education and Research
        Department of Agro-environmental Sciences, Faculty of Agriculture, Kyushu University
        744 Motooka, Nishi-ku, Fukuoka 819-0395, Japan
        Email: adityadewanto@gmail.com
        ResearchGate: https://www.researchgate.net/profile/Aditya-Hartono

    
Eco-toxicity of ILs --> tested againts Aliivibrio fischeri
Learning and prediction by Random Forest Regressor
EC50 = half maximum efective concentration
"""


### Configure the number of available CPU
import os as os
cpu_number = os.cpu_count()
n_jobs = cpu_number - 2

### Import some standard libraries
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

### Load and define dataframe for the learning dataset
Learning = pd.read_csv("learn_regression.csv") # This contains 110 ILs and the toxicity values

print('\t')
print('Learning dataset (original): \n')
print(f'Filetype: {type(Learning)}, Shape: {Learning.shape}')
print(Learning)
print(Learning.describe())

### Define X and Y out of the learning data (X: features, Y: label)
X = Learning.drop('LogEC50', axis=1)
Y = Learning['LogEC50']

print('\n')
print('Features (X): \n')
print(f'Filetype: {type(X)}, Shape: {X.shape}')
print(X)

print('\n')
print('Label (Y): \n')
print(f'Filetype: {type(Y)}, Shape: {Y.shape}')
print(Y)

### Split the learning data into training set and test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.12, random_state=1)

### Train and evaluate the model
from sklearn.ensemble import RandomForestRegressor
RFrgs = RandomForestRegressor(random_state=1)
from sklearn.model_selection import cross_val_score
scores = cross_val_score(RFrgs, X_train, Y_train, scoring="r2", cv=10)
def display_score(scores):
    print('\n')
    print('Preliminary run: \n')
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())
display_score(scores)

### Fine tune the model using GridSearchCV
from sklearn.model_selection import GridSearchCV
param_grid = [
    {'n_estimators': [100, 200, 300, 400, 500],
      'max_depth': [100, 200, 300, 400, 500],
      'max_features': [2, 4, 6, 8, 10, 12]}
    ]
grid_search = GridSearchCV(RFrgs, param_grid, scoring="r2", cv=10, n_jobs=n_jobs)
grid_search.fit(X_train, Y_train)
grid_search.best_params_

cvres = grid_search.cv_results_

print('\n')
print('Hyperparameter tuning: \n')
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(mean_score, params)
    
grid_search.best_estimator_

### Re-train the model with the best hyperparameters and the whole training set
RFrgs_opt = grid_search.best_estimator_
model = RFrgs_opt.fit(X_train, Y_train)

### Analyze and visualize the optimized model performance on TRAINING SET using CROSS-VALIDATION
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_predict

cv_pred = cross_val_predict(RFrgs_opt, X_train, Y_train, cv=10)
R2_cv = r2_score(Y_train, cv_pred)
RMSE_cv = np.sqrt(mean_squared_error(Y_train, cv_pred))

print('\n')
print('Quality assessment with cross-validation (employ model with the best hyperparameters): \n')
print('R2 score: ', R2_cv)
print('RMSE score', RMSE_cv)

### Plot the figure
fig = plt.figure(figsize=(5,5))
ax = fig.add_subplot(111)
for m in ['top', 'bottom', 'left', 'right']:
    ax.spines[m].set_linewidth(1)
    ax.spines[m].set_color('black')
ax.scatter(Y_train, cv_pred, 50, 'tab:blue')
ax.plot([Y_train.min(), Y_train.max()], [Y_train.min(), Y_train.max()], "k--", lw=2)
import matplotlib.font_manager as fm
fonts = fm.FontProperties(family='arial', size=20, weight='normal', style='normal')
plt.text(0.03, 0.9, '$R^2$ = {}'.format(str(round(R2_cv, 2))), transform=ax.transAxes, fontproperties=fonts)
plt.text(0.03, 0.82, '$RMSE$ = {}'.format(str(round(RMSE_cv, 2))), transform=ax.transAxes, fontproperties=fonts)
plt.xlabel('Measured logEC$_5$$_0$', labelpad=10, fontproperties=fonts)
plt.ylabel('Predicted logEC$_5$$_0$', labelpad=10, fontproperties=fonts)
import matplotlib.ticker as mticker
ticker_arg = [0.5, 1, 0.5, 1]
tickers = [mticker.MultipleLocator(ticker_arg[i]) for i in range(len(ticker_arg))]
ax.xaxis.set_minor_locator(tickers[0])
ax.xaxis.set_major_locator(tickers[1])
ax.yaxis.set_minor_locator(tickers[2])
ax.yaxis.set_major_locator(tickers[3])
xcoord = ax.xaxis.get_major_ticks()
ycoord = ax.yaxis.get_major_ticks()
[(i.label.set_fontproperties('arial'), i.label.set_fontsize(20)) for i in xcoord]
[(j.label.set_fontproperties('arial'), j.label.set_fontsize(20)) for j in ycoord]
dpi_assign = 300
plt.savefig('fig1c.jpg', dpi=dpi_assign, bbox_inches='tight')

### Analyze and visualize the optimized model performance on TRAINING SET via a SINGLE RUN
train_pred = model.predict(X_train)
R2_train = r2_score(Y_train, train_pred)
RMSE_train = np.sqrt(mean_squared_error(Y_train, train_pred))

print('\n')
print('Learning results for training set (employ model with the best hyperparameters): \n')
print('R2 score: ', R2_train)
print('RMSE score', RMSE_train)

### Plot the figure
fig = plt.figure(figsize=(5,5))
ax = fig.add_subplot(111)
for m in ['top', 'bottom', 'left', 'right']:
    ax.spines[m].set_linewidth(1)
    ax.spines[m].set_color('black')
ax.scatter(Y_train, train_pred, 50, 'tab:grey')
ax.plot([Y_train.min(), Y_train.max()], [Y_train.min(), Y_train.max()], "k--", lw=2)
plt.text(0.03, 0.9, '$R^2$ = {}'.format(str(round(R2_train, 2))), transform=ax.transAxes, fontproperties=fonts)
plt.text(0.03, 0.82, '$RMSE$ = {}'.format(str(round(RMSE_train, 2))), transform=ax.transAxes, fontproperties=fonts)
plt.xlabel('Measured logEC$_5$$_0$', labelpad=10, fontproperties=fonts)
plt.ylabel('Predicted logEC$_5$$_0$', labelpad=10, fontproperties=fonts)
ticker_arg = [0.5, 1, 0.5, 1]
tickers = [mticker.MultipleLocator(ticker_arg[i]) for i in range(len(ticker_arg))]
ax.xaxis.set_minor_locator(tickers[0])
ax.xaxis.set_major_locator(tickers[1])
ax.yaxis.set_minor_locator(tickers[2])
ax.yaxis.set_major_locator(tickers[3])
xcoord = ax.xaxis.get_major_ticks()
ycoord = ax.yaxis.get_major_ticks()
[(i.label.set_fontproperties('arial'), i.label.set_fontsize(20)) for i in xcoord]
[(j.label.set_fontproperties('arial'), j.label.set_fontsize(20)) for j in ycoord]
dpi_assign = 300
plt.savefig('figs1b.jpg', dpi=dpi_assign, bbox_inches='tight')

### Analyze and visualize the optimized model performance on TEST SET via a SINGLE RUN
test_pred = model.predict(X_test)
R2_test = r2_score(Y_test, test_pred)
RMSE_test = np.sqrt(mean_squared_error(Y_test, test_pred))

print('\n')
print('Learning results for test set (employ model with the best hyperparameters): \n')
print('R2 score: ', R2_test)
print('RMSE score: ', RMSE_test)

### Plot the figure
fig = plt.figure(figsize=(5,5))
ax = fig.add_subplot(111)
for m in ['top', 'bottom', 'left', 'right']:
    ax.spines[m].set_linewidth(1)
    ax.spines[m].set_color('black')
ax.scatter(Y_test, test_pred, 50, 'tab:red')
ax.plot([Y_test.min(), Y_test.max()], [Y_test.min(), Y_test.max()], "k--", lw=2)
plt.text(0.03, 0.9, '$R^2$ = {}'.format(str(round(R2_test, 2))), transform=ax.transAxes, fontproperties=fonts)
plt.text(0.03, 0.82, '$RMSE$ = {}'.format(str(round(RMSE_test, 2))), transform=ax.transAxes, fontproperties=fonts)
plt.xlabel('Measured logEC$_5$$_0$', labelpad=10, fontproperties=fonts)
plt.ylabel('Predicted logEC$_5$$_0$', labelpad=10, fontproperties=fonts)
ticker_arg = [0.5, 1, 0.5, 1]
tickers = [mticker.MultipleLocator(ticker_arg[i]) for i in range(len(ticker_arg))]
ax.xaxis.set_minor_locator(tickers[0])
ax.xaxis.set_major_locator(tickers[1])
ax.yaxis.set_minor_locator(tickers[2])
ax.yaxis.set_major_locator(tickers[3])
xcoord = ax.xaxis.get_major_ticks()
ycoord = ax.yaxis.get_major_ticks()
[(i.label.set_fontproperties('arial'), i.label.set_fontsize(20)) for i in xcoord]
[(j.label.set_fontproperties('arial'), j.label.set_fontsize(20)) for j in ycoord]
dpi_assign = 300
plt.savefig('fig1d.jpg', dpi=dpi_assign, bbox_inches='tight')

#Extract and visualize feature importances
feature_importances = pd.DataFrame([X_train.columns, model.feature_importances_]).T
feature_importances.columns = ['features', 'importance']

print('\n')
print(feature_importances)

fig = plt.figure(figsize=(12,5))
ax = sns.barplot(x=feature_importances['features'], y=feature_importances['importance'], palette='vlag') 
plt.xlabel('Feature', labelpad=10, fontproperties=fonts)
plt.ylabel('Importance', labelpad=10, fontproperties=fonts)
import matplotlib.font_manager as fm
fonts = fm.FontProperties(family='arial', size=20, weight='normal', style='normal')
import matplotlib.ticker as mticker
ticker_arg = [0.025, 0.05, 0.025, 0.05]
tickers = [mticker.MultipleLocator(ticker_arg[i]) for i in range(len(ticker_arg))]
# ax.yaxis.set_minor_locator(tickers[0])
ax.yaxis.set_major_locator(tickers[1])
xcoord = ax.xaxis.get_major_ticks()
ycoord = ax.yaxis.get_major_ticks()
[(i.label.set_fontproperties('arial'), i.label.set_fontsize(20)) for i in xcoord]
[(j.label.set_fontproperties('arial'), j.label.set_fontsize(20)) for j in ycoord]
dpi_assign = 300
plt.savefig('fig2b.jpg', dpi=dpi_assign, bbox_inches='tight')

### Predict EC50 values of previous reports -- the classification data
prev_rep = pd.read_csv("learn_classification.csv")
prev_rep_des = prev_rep.drop('Metal', axis=1)
prev_pred = model.predict(prev_rep_des)

print('\n')
print('Prediction of classification data: ')
print(prev_pred)

### Load descriptors for actual predictions
descriptors = pd.read_csv("predict_descriptors.csv") # This contains descriptors (features) for 150 chemicals

print('\n')
print('Descriptor data: \n')
print(f'Filetype: {type(descriptors)}, Shape: {descriptors.shape}')
print(descriptors.head())
print(descriptors.describe())

### Predict the LogEC50 value of each descriptor i.e. eco-toxicity 
value_pred = model.predict(descriptors)

print('\n')
print('Prediction of descriptor data: ')
print(value_pred)
