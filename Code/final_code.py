#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 17 15:32:15 2021

@author: tarun
"""

#Data Manipulation
import numpy as np
import pandas as pd

#Visualisation
import matplotlib.pyplot as plt
from matplotlib import pyplot
import seaborn as sns

#Scikit learn
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, auc

#Keras
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import SGD
from keras.wrappers.scikit_learn import KerasClassifier

#Tensorflow
from tensorflow.random import set_seed

#Other
from random import randint, random
from statistics import mean

BASE_DIR = '/home/tarun/MATH5836/Assessment_2/'

credit_data = pd.read_table(BASE_DIR + 'crx.data', delimiter=",", header=None)
credit_data.columns = ['A'+str(i+1) for i in range(16)]

credit_data.shape
credit_data.describe()
credit_data.dtypes




#########################
#### DATA PROCESSING ####
#########################




#Get rid of rows with ? in A1, A4, A5, A6, A7
credit_data.shape
credit_data = credit_data.drop(credit_data[(credit_data.A1 == '?') | (credit_data.A4 == '?') | (credit_data.A5 == '?') | (credit_data.A6 == '?') | (credit_data.A7 == '?')].index)
credit_data.shape

#A2 and A14 set to column average
a2_mean = round(credit_data[credit_data['A2']!='?']['A2'].astype(float, errors='ignore').mean(), 2)
credit_data.loc[credit_data['A2']=='?', 'A2'] = a2_mean

a14_mean = round(credit_data[credit_data['A14']!='?']['A14'].astype(float, errors='ignore').mean(), 0)
credit_data.loc[credit_data['A14']=='?', 'A14'] = a14_mean

#Put to floats
credit_data['A2'] = credit_data['A2'].astype(float)
credit_data['A3'] = credit_data['A3'].astype(float)
credit_data['A8'] = credit_data['A8'].astype(float)

#Put to integers
credit_data['A11'] = credit_data['A11'].astype(int)
credit_data['A14'] = credit_data['A14'].astype(int)
credit_data['A15'] = credit_data['A15'].astype(int)

credit_data.describe()
credit_data.dtypes


#Encode nominal data type deature columns
credit_data.loc[:,'A1'] = credit_data.loc[:,'A1'].map({'a':0, 'b':1})
credit_data.loc[:,'A4'] = credit_data.loc[:,'A4'].map({'u':0, 'y':1, 'l':2})
credit_data.loc[:,'A5'] = credit_data.loc[:,'A5'].map({'g':0, 'p':1, 'gg':2})
credit_data.loc[:,'A6'] = credit_data.loc[:,'A6'].map({'c':0, 'd':1, 'cc':2, 'i':3, 'j':4, 'k':5, 'm':6, 'r':7, 'q':8, 'w':9, 'x':10, 'e':11, 'aa':12, 'ff':13})
credit_data.loc[:,'A7'] = credit_data.loc[:,'A7'].map({'v':0, 'h':1, 'bb':2, 'j':3, 'n':4, 'z':5, 'dd':6, 'ff':7, 'o':8})
credit_data.loc[:,'A9'] = credit_data.loc[:,'A9'].map({'t':0, 'f':1})
credit_data.loc[:,'A10'] = credit_data.loc[:,'A10'].map({'t':0, 'f':1})
credit_data.loc[:,'A12'] = credit_data.loc[:,'A12'].map({'t':0, 'f':1})
credit_data.loc[:,'A13'] = credit_data.loc[:,'A13'].map({'g':0, 'p':1, 's':2})
credit_data.loc[:,'A16'] = credit_data.loc[:,'A16'].map({'-':0, '+':1})

#Normalise data - Min-Max Normalisation
credit_data_processed = (credit_data-credit_data.min())/(credit_data.max()-credit_data.min())
credit_data_processed.to_csv(BASE_DIR + 'data.csv', index=False)




#######################
#### VISUALISATION ####
#######################




plt.style.use('ggplot')

#HISTOGRAMS for A2, A3, A8, A11, A14, A15
def plot_hist(data_df = credit_data, feature_col_name = [], bins = 10):
    
    for name in feature_col_name:
        plt.hist(data_df[name], bins = bins, edgecolor = 'black', color = 'blue', alpha = 0.6)
        plt.xlabel(name)
        plt.ylabel('Frequency')
        plt.title(f'Histogram of {name} feature')
        plt.savefig(f'/home/tarun/MATH5836/Assessment_2/{name}_hist.png')
        plt.close()

plot_hist(data_df = credit_data, feature_col_name = ['A2', 'A3', 'A8', 'A11', 'A14'], bins = 20)
plot_hist(data_df = credit_data, feature_col_name = ['A15'], bins = 30)


#BOXPLOTS for A2, A3, A8, A11, A14, A15
def plot_box_plot(data_df = credit_data, feature_col_name = []):
    
    for name in feature_col_name:
                
        plt.title(f'Histogram of {name} feature')
        
        credit_data.boxplot(column = name, grid=True)
        plt.title(f'Boxplot of {name} feature')
        plt.savefig(f'/home/tarun/MATH5836/Assessment_2/{name}_boxplot.png')
        plt.close()
        
plot_box_plot(data_df = credit_data, feature_col_name = ['A2', 'A3', 'A8', 'A11', 'A14', 'A15'])


#BARPLOTS for A1, A4, A5, A6, A7, A9, A10, A12, A13
def bar_plot(data_df = credit_data, feature_col_name = []):
    
    for name in feature_col_name:
        
        bars = []
        height = []
        for k,v in sorted(dict(credit_data[name].value_counts()).items()):
            bars.append(str(k))
            height.append(v)
        
        plt.bar(bars, height, color ='green', width = 0.3, alpha=0.7)
        plt.xlabel('Category')
        plt.ylabel('Frequency')
        plt.title(f'Bar plot of {name} feature')
        plt.savefig(f'/home/tarun/MATH5836/Assessment_2/{name}_barplot.png')
        plt.close()
        
bar_plot(data_df = credit_data, feature_col_name = ['A1', 'A4', 'A5', 'A6', 'A7', 'A9', 'A10', 'A12', 'A13'])
bar_plot(data_df = credit_data, feature_col_name = ['A16'])



# CORRELATION HEATMAP
def corr_map(credit_data_array):

    corr_matrix = np.corrcoef(credit_data_array.T)
    plt.subplots(figsize=(15,10))
    sns.set(font_scale=1)
    x_axis_labels = list(credit_data_processed.columns) # labels for x-axis
    y_axis_labels = list(credit_data_processed.columns)
    sns.heatmap(corr_matrix, cmap='mako', annot=True, fmt='.2f', xticklabels=x_axis_labels, yticklabels=y_axis_labels, linewidths=.5)
    plt.title('Correlation Heatmap for Credit Approval Data')
    plt.savefig('/home/tarun/MATH5836/Assessment_2/feature_correlations.png')
    plt.close()

corr_map(np.array(credit_data_processed))




###################
#### MODELLING ####
###################




X = credit_data_processed[list(credit_data_processed.columns)[:-1]]
y = credit_data_processed[[list(credit_data_processed.columns)[-1]]]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.5, random_state=42, shuffle=True, stratify=y)


#Training time (number of epochs) selection

#50 Epochs
set_seed(200)
model = Sequential()
model.add(Dense(10, input_dim=15, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50, verbose=1)
pd.DataFrame(history.history).plot(figsize=(8,5))
plt.grid(True)
plt.gca().set_ylim(0,1)
plt.xlabel('Number of Epochs')
plt.title('Learning Curves over 50 Epochs')
plt.savefig('/home/tarun/MATH5836/Assessment_2/Learning_curves_50_epochs.png')


#75 Epochs
set_seed(200)
model = Sequential()
model.add(Dense(10, input_dim=15, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=75, verbose=1)
pd.DataFrame(history.history).plot(figsize=(8,5))
plt.grid(True)
plt.gca().set_ylim(0,1)
plt.xlabel('Number of Epochs')
plt.title('Learning Curves over 75 Epochs')
plt.savefig('/home/tarun/MATH5836/Assessment_2/Learning_curves_75_epochs.png')


#100 Epochs
set_seed(200)
model = Sequential()
model.add(Dense(10, input_dim=15, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, verbose=1)
pd.DataFrame(history.history).plot(figsize=(8,5))
plt.grid(True)
plt.gca().set_ylim(0,1)
plt.xlabel('Number of Epochs')
plt.title('Learning Curves over 100 Epochs')
plt.savefig('/home/tarun/MATH5836/Assessment_2/Learning_curves_100_epochs.png')


#150 Epochs
set_seed(200)
model = Sequential()
model.add(Dense(10, input_dim=15, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=150, verbose=1)
pd.DataFrame(history.history).plot(figsize=(8,5))
plt.grid(True)
plt.gca().set_ylim(0,1)
plt.xlabel('Number of Epochs')
plt.title('Learning Curves over 150 Epochs')
plt.savefig('/home/tarun/MATH5836/Assessment_2/Learning_curves_150_epochs.png')


#200 Epochs
set_seed(200)
model = Sequential()
model.add(Dense(10, input_dim=15, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=200, verbose=1)
pd.DataFrame(history.history).plot(figsize=(8,5))
plt.grid(True)
plt.gca().set_ylim(0,1)
plt.xlabel('Number of Epochs')
plt.title('Learning Curves over 200 Epochs')
plt.savefig('/home/tarun/MATH5836/Assessment_2/Learning_curves_200_epochs.png')


#500 Epochs
set_seed(200)
model = Sequential()
model.add(Dense(10, input_dim=15, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=500, verbose=1)
pd.DataFrame(history.history).plot(figsize=(8,5))
plt.grid(True)
plt.gca().set_ylim(0,1)
plt.xlabel('Number of Epochs')
plt.title('Learning Curves over 500 Epochs')
plt.savefig('/home/tarun/MATH5836/Assessment_2/Learning_curves_500_epochs.png')

#100 epochs should suffice


#ADAM vs SGD

adam_train_acc = []
adam_test_acc = []
sgd_train_acc = []
sgd_test_acc = []

for experiment in range(20):
    
    set_seed(200+experiment)
    state = randint(0,200)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.5, random_state=state, shuffle=True, stratify=y)
    
    #ADAM
    model_adam = Sequential()
    model_adam.add(Dense(10, input_dim=15, activation='relu'))
    model_adam.add(Dense(1, activation='sigmoid'))
    model_adam.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    history_adam = model_adam.fit(X_train, y_train, epochs=100, verbose=0) #validation_data=(testX, testy)
    
    # evaluate the adam model
    _, train_acc_adam = model_adam.evaluate(X_train, y_train, verbose=0)
    _, test_acc_adam = model_adam.evaluate(X_test, y_test, verbose=0)
    print('ADAM Train: %.3f, Test: %.3f' % (train_acc_adam, test_acc_adam))
    
    adam_train_acc.append(train_acc_adam)
    adam_test_acc.append(test_acc_adam)
    
    
    #SGD
    model_sgd = Sequential()
    model_sgd.add(Dense(10, input_dim=15, activation='relu'))
    model_sgd.add(Dense(1, activation='sigmoid'))
    model_sgd.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])
    
    history_sgd = model_sgd.fit(X_train, y_train, epochs=100, verbose=0) #validation_data=(testX, testy)
        
    # evaluate the sgd model
    _, train_acc_sgd = model_sgd.evaluate(X_train, y_train, verbose=0)
    _, test_acc_sgd = model_sgd.evaluate(X_test, y_test, verbose=0)
    print('SGD Train: %.3f, Test: %.3f' % (train_acc_sgd, test_acc_sgd))
    
    sgd_train_acc.append(train_acc_sgd)
    sgd_test_acc.append(test_acc_sgd)
    

adam_vs_sgd_dict = {'ADAM Accuracy on Train Set':adam_train_acc,
                    'ADAM Accuracy on Test Set':adam_test_acc,
                    'SGD Accuracy on Train Set':sgd_train_acc,
                    'SGD Accuracy on Test Set':sgd_test_acc,
                    }

adam_vs_sgd = pd.DataFrame(adam_vs_sgd_dict)
adam_vs_sgd.to_csv('/home/tarun/MATH5836/Assessment_2/adam_vs_sgd.csv', index = False)



#SGD: Learning rate comparison
opt1 = SGD(lr=0.001)
opt2 = SGD(lr=0.01)
opt3 = SGD(lr=0.1)

opt1_train_acc = []
opt1_test_acc = []

opt2_train_acc = []
opt2_test_acc = []

opt3_train_acc = []
opt3_test_acc = []


for experiment in range(10):
    
    set_seed(200 + experiment)
    state = randint(0,200)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.5, random_state=state, shuffle=True, stratify=y)
    
    model_sgd_1 = Sequential()
    model_sgd_1.add(Dense(10, input_dim=15, activation='relu'))
    model_sgd_1.add(Dense(1, activation='sigmoid'))
    model_sgd_1.compile(loss='binary_crossentropy', optimizer = opt1, metrics=['accuracy'])
    history_sgd_1 = model_sgd_1.fit(X_train, y_train, epochs=100, verbose=0)
    
    _, train_acc_1 = model_sgd_1.evaluate(X_train, y_train, verbose=0)
    _, test_acc_1 = model_sgd_1.evaluate(X_test, y_test, verbose=0)
    
    opt1_train_acc.append(train_acc_1)
    opt1_test_acc.append(test_acc_1)
    
    
    
    model_sgd_2 = Sequential()
    model_sgd_2.add(Dense(10, input_dim=15, activation='relu'))
    model_sgd_2.add(Dense(1, activation='sigmoid'))
    model_sgd_2.compile(loss='binary_crossentropy', optimizer = opt2, metrics=['accuracy'])
    history_sgd_2 = model_sgd_2.fit(X_train, y_train, epochs=100, verbose=0)
    
    _, train_acc_2 = model_sgd_2.evaluate(X_train, y_train, verbose=0)
    _, test_acc_2 = model_sgd_2.evaluate(X_test, y_test, verbose=0)
    
    opt2_train_acc.append(train_acc_2)
    opt2_test_acc.append(test_acc_2)
    
    
    
    model_sgd_3 = Sequential()
    model_sgd_3.add(Dense(10, input_dim=15, activation='relu'))
    model_sgd_3.add(Dense(1, activation='sigmoid'))
    model_sgd_3.compile(loss='binary_crossentropy', optimizer = opt3, metrics=['accuracy'])
    history_sgd_3 = model_sgd_3.fit(X_train, y_train, epochs=100, verbose=0)
    
    _, train_acc_3 = model_sgd_3.evaluate(X_train, y_train, verbose=0)
    _, test_acc_3 = model_sgd_3.evaluate(X_test, y_test, verbose=0)
    
    opt3_train_acc.append(train_acc_3)
    opt3_test_acc.append(test_acc_3)
    
    #print('done')
    

sgd_lr_comparison = {'Training Accuracy LR: 0.001':opt1_train_acc,
                     'Test Accuracy LR: 0.001':opt1_test_acc,
                     'Training Accuracy LR: 0.01':opt2_train_acc,
                     'Test Accuracy LR: 0.01':opt2_test_acc,
                     'Training Accuracy LR: 0.1':opt3_train_acc,
                     'Test Accuracy LR: 0.1':opt3_test_acc}
                     
                     
sgd_lr_comp_df = pd.DataFrame(sgd_lr_comparison)
sgd_lr_comp_df.to_csv('/home/tarun/MATH5836/Assessment_2/sgd_lr_comparison.csv', index = False)


#SGD with learning rate = 0.1 ROC/AUC and Confusion matrix

#https://machinelearningmastery.com/binary-classification-tutorial-with-the-keras-deep-learning-library/
#https://www.dlology.com/blog/simple-guide-on-how-to-generate-roc-plot-for-keras-classifier/

def build_model():

    opt3 = SGD(lr=0.1)
    
    model_sgd_3 = Sequential()
    model_sgd_3.add(Dense(10, input_dim=15, activation='relu'))
    model_sgd_3.add(Dense(1, activation='sigmoid'))
    model_sgd_3.compile(loss='binary_crossentropy', optimizer = opt3, metrics=['accuracy'])
    
    return model_sgd_3

keras_model = build_model()
keras_model.fit(X_train, y_train, epochs=100, verbose=0)

y_pred_keras = keras_model.predict(X_test).ravel()

#Confusion Matrix
confusion_matrix(y_true, y_pred_keras)

#ROC/AUC
fpr_keras, tpr_keras, thresholds_keras = roc_curve(y_test, y_pred_keras)

plt.plot(fpr_keras, tpr_keras, linestyle='solid')
x_ln = np.linspace(0,1,100)
y_ln = x_ln
plt.plot(x_ln, y_ln, linestyle='dotted')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.savefig('/home/tarun/MATH5836/Assessment_2/roc.png')
auc_keras = auc(fpr_keras, tpr_keras)
