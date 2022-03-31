
import PySimpleGUI as sg
from scipy.io.wavfile import read
import os

import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import pyACA

from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from itertools import combinations
from sklearn.base import clone
from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs



fList = ['SpectralCentroid', 'SpectralCrestFactor', 'SpectralDecrease', 'SpectralFlatness', 'SpectralFlux',
         'SpectralKurtosis', 'SpectralMfccs', 'SpectralPitchChroma', 'SpectralRolloff', 'SpectralSkewness',
         'SpectralSlope', 'SpectralSpread', 'SpectralTonalPowerRatio', 'TimeAcfCoeff', 'TimeMaxAcf', 'TimePeakEnvelope',
         'TimePredictivityRatio', 'TimeRms', 'TimeStd', 'TimeZeroCrossingRate']


def features(feature, file, f_s):
    [vsf, t] = pyACA.computeFeature(feature, file, f_s, iBlockLength=chunkSize, iHopLength=hopSize)
    return vsf


model1 = LogisticRegression(solver='lbfgs', max_iter=200)  # binary logistic regression
model2 = svm.SVC()
model3 = KNeighborsClassifier(n_neighbors=3)
model4 = RandomForestClassifier(max_depth=2, random_state=0)


# main landing page layout
sg.theme('Black')
layout = \
    [[sg.Text("Path to Folder:"), sg.Input(key="-IN-", change_submits=True), sg.FolderBrowse(key="-IN-")],
     [sg.Text('__'*46)],
     [sg.Text('Enter a Chunk Size (samples):    '), sg.Slider(range=(256, 8192), key="-CHUNK-", default_value=2048,
                                                              size=(40, 20), orientation='horizontal')],
     [sg.Text('Enter an Overlap Ratio (percent):'), sg.Slider(range=(0, 99), key="-HOP-", default_value=50,
                                                              size=(40, 20), orientation='horizontal')],
     [sg.Text('Feature Extraction: Audio features are computed and averaged across chunks')],
     [sg.Text('How many features would you like to test?'), sg.Slider(range=(1,18), key="-FEATURESNUM-",
                                                                      default_value=18, size=(40,28),
                                                                      orientation='horizontal')],
     [sg.Text('    (Step 1 of 2) Launch Feature Extraction:'), sg.Button('LAUNCH', key="-FEAT-"),
                                                               sg.Button('Cancel', key="-FEATX-")],
     [sg.Text('__'*46)],
     [sg.Text('Model Training: multiple machine learning models which each have their own "flavor" of prediction')],
     [sg.Text('How many models would you like to train?'), sg.Slider(range=(1, 4),
                                                                     key="-MODELSNUM-", default_value=4,
                                                                     size=(40,28),orientation='horizontal')],
     [sg.Text('    (Step 2 of 2) Launch Model Training:     '), sg.Button('LAUNCH', key="-MODEL-"),
                                                                sg.Button('Cancel', key="-MODELX-")],
     [sg.Text('__'*46)],
     [sg.Text('View Results: '), sg.Button('GO', key="-RESULTS-")]]


# results page layout
layout2 = \
    [[sg.Text("Optimization Curve:")],
     [sg.Text('__'*30)]]
     #[sg.Image(filename='/Users/tuckeralexander/Desktop/Classes/Capstone/Results/example_1', key="-IMAGE-")]]


window = sg.Window('Setup and Training', layout, element_justification='l')
window2 = sg.Window('Results and Code', layout2, modal=True)

modelBool = False       # deactivating buttons that aren't supposed to be used yet
resultsBool = False     # initializing results window in the background
count = 0               # initializing tracker for window 2 launch event


# launching landing page:
while True:
    event, values = window.read()
    if event == "-FEAT-":       # upon hitting step 1 'Launch' button
        modelBool = True
        chunkSize = values["-CHUNK-"]
        hopSize = (1 - (values["-HOP-"])/100) * chunkSize
        classes = os.listdir(values["-IN-"])
        entries1 = os.listdir(values["-IN-"] + '/' + classes[1])
        entries2 = os.listdir(values["-IN-"] + '/' + classes[2])
        total = len(entries1) + len(entries2)
        featuresA = []
        featuresB = []
        feature_names = []
        for i in range(total):  # step 1 progress bar
            if not sg.one_line_progress_meter('Feature Extraction Progress', i+1, total, 'step 1 of 2: feature extraction'):
                break
            if i >= len(entries1):  # processing class B
                [fs, x2] = read(values["-IN-"] + '/' + classes[2] + '/' + entries2[i-len(entries1)])
                x2 = x2[:, 0]  # grabbing only channel 1
            else:   # processing class A
                [fs, x] = read(values["-IN-"] + '/' + classes[1] + '/' + entries1[i])
                x = x[:, 0]  # grabbing only channel 1
        for f in range(int(values["-FEATURESNUM-"])):
            feature_names.append(fList[f])
            current_featureA = features(fList[f], x2, fs)
            current_featureA[:] = current_featureA[:] / max(current_featureA)
            featuresA.append(current_featureA)
            current_featureB = features(fList[f], x, fs)
            current_featureB[:] = current_featureB[:] / max(current_featureB)
            featuresB.append(current_featureB)
    elif event == "-MODEL-" and modelBool:     # upon hitting step 2 'Launch' button
        data1 = np.ones(len(featuresA[0]))
        data2 = np.zeros(len(featuresB[0]))
        data3 = np.concatenate((data1, data2))
        data5 = []
        for d in range(int(values["-FEATURESNUM-"])):
            data4 = np.concatenate([featuresA[d], featuresB[d]])
            data5.append(data4)
        # predictors = model_data.iloc[:, 1:]
        # categories = model_data.iloc[:, 0]
        categories = pd.DataFrame(data3)
        predictors = pd.DataFrame(data5).transpose()
        predictors.columns = feature_names
        #cat_train, cat_test, pred_train, pred_test = train_test_split(categories, predictors, test_size=.2,
        #                                                              random_state=25)
        for i in range(0, int(values["-MODELSNUM-"])):
            count = count + 1
            if not sg.one_line_progress_meter('Model Evaluation Progress', i+1, int(values["-MODELSNUM-"]),
                                              'step 2 of 2: testing model performance'):
                break
            if i == 0:
                sfs1 = SFS(model1, k_features=int(values["-FEATURESNUM-"]), forward=True,
                                                 floating=False, verbose=2, scoring='accuracy', cv=0)
                pipe1 = make_pipeline(StandardScaler(), sfs1)
                pipe1.fit(predictors, categories)
                plot_sfs(sfs1.get_metric_dict(), kind='std_err');
            elif i == 1:
                sfs2 = SFS(model2, k_features=int(values["-FEATURESNUM-"]), forward=True,
                                                 floating=False, verbose=2, scoring='accuracy', cv=0)
                pipe2 = make_pipeline(StandardScaler(), sfs1)
                pipe2.fit(pred_train, cat_train)
            elif i == 2:
                sfs3 = SFS(model3, k_features=int(values["-FEATURESNUM-"]), forward=True,
                                                 floating=False, verbose=2, scoring='accuracy', cv=0)
                pipe3 = make_pipeline(StandardScaler(), sfs1)
                pipe3.fit(pred_train, cat_train)
            elif i == 3:
                sfs4 = SFS(model4, k_features=int(values["-FEATURESNUM-"]), forward=True,
                                                 floating=False, verbose=2, scoring='accuracy', cv=0)
                pipe4 = make_pipeline(StandardScaler(), sfs1)
                pipe4.fit(pred_train, cat_train)
            if count == int(values["-MODELSNUM-"]):
                resultsBool = True
    elif event == "-RESULTS-" and resultsBool:      # launch results window
        window.close()
        k_features = [len(k) for k in sfs1.subsets_]
        plt.plot(k_features, sfs1.scores_, label='Logistic Regression', marker='o')
        plt.plot(k_features, sfs2.scores_, label='Support Vector Machine', marker='o')
        plt.plot(k_features, sfs3.scores_, label='K Nearest Neighbor', marker='o')
        plt.plot(k_features, sfs4.scores_, marker='o', label='Random Forest')

        plt.ylim([.33, 1.02])
        plt.ylabel('Accuracy')
        plt.xlabel('Number of features')
        plt.grid()
        plt.tight_layout()
        plt.legend()
        while True:
            event2, values2 = window2.read()
            if event2 == "Exit" or event2 == sg.WIN_CLOSED:
                break
    elif event == sg.WIN_CLOSED or event == "Exit" or event == '-FEATX-' or event == '-MODELX-':
        break


window.close()
