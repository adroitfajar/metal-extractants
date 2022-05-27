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


Selectivity of ILs toward particular metals
Learning and prediction by Random Forest Classifier
Data abbreviations:
    IL  = ionic liquid
    Pt  = platinum
    Li  = lithium
    Nd  = neodymium
    Nap = not applicable
"""


### Configure the number of available CPU
import os as os
cpu_number = os.cpu_count()
n_jobs = cpu_number - 2

### Import some standard libraries
import pandas as pd
import seaborn as sns
# import numpy as np
import matplotlib.pyplot as plt
pd.options.mode.chained_assignment = None

### Load and define dataframe for the learning dataset
Learning = pd.read_csv("learn_classification.csv") # This contains 76 ILs with non-numeric labels

print('\t')
print('Learning dataset (original): \n')
print(f'Filetype: {type(Learning)}, Shape: {Learning.shape}')
print(Learning)
print(Learning.describe())

### Convert non-numeric data to numeric
Learning.Metal[Learning.Metal == 'Pt'] = 1
Learning.Metal[Learning.Metal == 'Li'] = 2
Learning.Metal[Learning.Metal == 'Nd'] = 3
Learning.Metal[Learning.Metal == 'Nap'] = 0

print('\n')
print('Learning dataset (converted): \n')
print(f'Filetype: {type(Learning)}, Shape: {Learning.shape}')
print(Learning)

### Define X and Y out of the learning data (X: features, Y: label)
X = Learning.drop('Metal', axis=1)
Y = Learning['Metal'].values
Y = Y.astype('int')

### Split the learning data into training set and test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.15, random_state=1, stratify=Y)

### Train and evaluate the model
from sklearn.ensemble import RandomForestClassifier
RFclf = RandomForestClassifier(random_state=1)
from sklearn.model_selection import cross_val_score
scores = cross_val_score(RFclf, X_train, Y_train, scoring="accuracy", cv=9)
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
grid_search = GridSearchCV(RFclf, param_grid, scoring="accuracy", cv=9, n_jobs=n_jobs)
grid_search.fit(X_train, Y_train)
grid_search.best_params_

cvres = grid_search.cv_results_

print('\n')
print('Hyperparameter tuning: \n')
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(mean_score, params)
    
grid_search.best_estimator_

### Re-train the model with the best hyperparameters and the whole training set
RFclf_opt = grid_search.best_estimator_
model = RFclf_opt.fit(X_train, Y_train)

### Analyze and visualize the optimized model performance on TRAINING SET using CROSS-VALIDATION
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import cross_val_predict

cv_pred = cross_val_predict(RFclf_opt, X_train, Y_train, cv=9)

print('\n')
print('Quality assessment with cross-validation (employ model with the best hyperparameters): \n')
print('Accuracy score: ', accuracy_score(Y_train, cv_pred))
print('Classification report: \n', classification_report(Y_train, cv_pred))
print('Confusion matrix: \n', confusion_matrix(Y_train, cv_pred))

### Plot using ConfusionMatrixDisplay
import matplotlib.font_manager as fm
fonts = fm.FontProperties(family='arial', size=20, weight='normal', style='normal')
categories = 'NA', 'Pt', 'Li', 'Nd'
cm_cv = confusion_matrix(Y_train, cv_pred, labels=RFclf_opt.classes_)
disp_cv = ConfusionMatrixDisplay(confusion_matrix=cm_cv, display_labels=categories)
fig = plt.figure(figsize=(5, 5))
ax = fig.add_subplot(111)
disp_cv.plot(ax=ax, cmap='Blues')
plt.xlabel('Predicted Label', labelpad=10, fontproperties=fonts)
plt.ylabel('True Label', labelpad=10, fontproperties=fonts)
dpi_assign = 300
plt.savefig('fig1a.jpg', dpi=dpi_assign, bbox_inches='tight')

#### Analyze and visualize the optimized model performance on TRAINING SET via a SINGLE RUN
train_pred = model.predict(X_train)

print('\n')
print('Learning results for training set (employ model with the best hyperparameters): \n')
print('Accuracy score: ', accuracy_score(Y_train, train_pred))
print('Classification report: \n', classification_report(Y_train, train_pred))
print('Confusion matrix: \n', confusion_matrix(Y_train, train_pred))

### Plot using ConfusionMatrixDisplay
cm_train = confusion_matrix(Y_train, train_pred, labels=RFclf_opt.classes_)
disp_train = ConfusionMatrixDisplay(confusion_matrix=cm_train, display_labels=categories)
fig = plt.figure(figsize=(5, 5))
ax = fig.add_subplot(111)
disp_train.plot(ax=ax, cmap='Greys')
plt.xlabel('Predicted Label', labelpad=10, fontproperties=fonts)
plt.ylabel('True Label', labelpad=10, fontproperties=fonts)
dpi_assign = 300
plt.savefig('figs1a.jpg', dpi=dpi_assign, bbox_inches='tight')

### Analyze and visualize the optimized model performance on TEST SET via a SINGLE RUN
test_pred = model.predict(X_test)

print('\n')
print('Learning results for test set (employ model with the best hyperparameters): \n')
print('Accuracy score: ', accuracy_score(Y_test, test_pred))
print('Classification report: \n', classification_report(Y_test, test_pred))
print('Confusion matrix: \n', confusion_matrix(Y_test, test_pred))

### Plot using ConfusionMatrixDisplay
cm_test = confusion_matrix(Y_test, test_pred, labels=RFclf_opt.classes_)
disp_test = ConfusionMatrixDisplay(confusion_matrix=cm_test, display_labels=categories)
fig = plt.figure(figsize=(5, 5))
ax = fig.add_subplot(111)
disp_test.plot(ax=ax, cmap='Reds')
plt.xlabel('Predicted Label', labelpad=10, fontproperties=fonts)
plt.ylabel('True Label', labelpad=10, fontproperties=fonts)
dpi_assign = 300
plt.savefig('fig1b.jpg', dpi=dpi_assign, bbox_inches='tight')

### Extract and visualize feature importances
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
ax.yaxis.set_minor_locator(tickers[0])
ax.yaxis.set_major_locator(tickers[1])
xcoord = ax.xaxis.get_major_ticks()
ycoord = ax.yaxis.get_major_ticks()
[(i.label.set_fontproperties('arial'), i.label.set_fontsize(20)) for i in xcoord]
[(j.label.set_fontproperties('arial'), j.label.set_fontsize(20)) for j in ycoord]
dpi_assign = 300
plt.savefig('fig2a.jpg', dpi=dpi_assign, bbox_inches='tight')

### Predict labels of previous reports -- the regression data
prev_rep = pd.read_csv("learn_regression.csv")
prev_rep_des = prev_rep.drop('LogEC50', axis=1)
prev_pred = model.predict(prev_rep_des)

print('\n')
print('Prediction of regression data: ')
print(prev_pred)

Previous = pd.DataFrame(prev_pred, columns = ['Class']) # Covert numpy to pandas
Previous.Class[Previous.Class == 1] = 'Pt'
Previous.Class[Previous.Class == 2] = 'Li'
Previous.Class[Previous.Class == 3] = 'Nd'
Previous.Class[Previous.Class == 0] = 'Nap'

print('\n')
print('Prediction of regression data (converted): ')
print(Previous)
print(Previous.value_counts())

### Load descriptors for actual predictions
descriptors = pd.read_csv("predict_descriptors.csv") # This contains descriptors (features) for 150 chemicals

print('\n')
print('Descriptor data: ')
print(f'Filetype: {type(descriptors)}, Shape: {descriptors.shape}')
print(descriptors)
print(descriptors.describe())

### Predict the class of each descriptor i.e. metal selectivity
label_pred = model.predict(descriptors)

print('\n')
print('Prediction of descriptor data: ')
print(label_pred)

Prediction = pd.DataFrame(label_pred, columns = ['Selectivity']) # Covert numpy to pandas
Prediction.Selectivity[Prediction.Selectivity == 1] = 'Pt'
Prediction.Selectivity[Prediction.Selectivity == 2] = 'Li'
Prediction.Selectivity[Prediction.Selectivity == 3] = 'Nd'
Prediction.Selectivity[Prediction.Selectivity == 0] = 'Nap'

print('\n')
print('Prediction of descriptor data (converted): ')
print(Prediction)
print(Prediction.value_counts())
