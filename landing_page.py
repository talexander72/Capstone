
import PySimpleGUI as sg
from scipy.io.wavfile import read
import os

import pandas as pd
import numpy as np
import math
import pyACA

from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

from itertools import combinations
from sklearn.base import clone
from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score


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
        for i in range(total):  # step 1 progress bar
            if not sg.one_line_progress_meter('Feature Extraction Progress', i+1, total, 'step 1 of 2: feature extraction'):
                break
            if i >= len(entries1):  # processing class B
                [fs, x2] = read(values["-IN-"] + '/' + classes[2] + '/' + entries2[i-len(entries1)])
                for f in range(int(values["-FEATURESNUM-"])):
                    current_feature = features(fList[f], x2, fs)
                    current_feature[:] = current_feature[:] / max(current_feature)
                    for f in range(len(current_feature)-1):
                        featuresA.append(current_feature[f])
                i = i + 1
            else:   # processing class A
                [fs, x] = read(values["-IN-"] + '/' + classes[1] + '/' + entries1[i])
                for f in range(int(values["-FEATURESNUM-"])):
                    current_feature = features(fList[f], x, fs)
                    current_feature[:] = current_feature[:] / max(current_feature)
                    for f in range(len(current_feature) - 1):
                        featuresB.append(current_feature[f])
                i = i + 1
    elif event == "-MODEL-" and modelBool:     # upon hitting step 2 'Launch' button
        data1 = np.ones(len(featuresA))
        data2 = np.zeros(len(featuresB))
        data3 = np.concatenate((data1, data2))
        data4 = np.concatenate((featuresA, featuresB))
        data5 = np.array([data3, data4])
        model_data = pd.DataFrame(data5).transpose()
        predictors = model_data.iloc[:, 1:]
        categories = model_data.iloc[:, 0]
        cat_train, cat_test, pred_train, pred_test = train_test_split(categories, predictors, test_size=.2,
                                                                      random_state=25)
        model1.fit(pred_train, cat_train)
        for i in range(0, int(values["-MODELSNUM-"])):
            count = count + 1
            if not sg.one_line_progress_meter('Model Training Progress', i+1, int(values["-MODELSNUM-"]),
                                              'step 2 of 2: training models'):
                break
            if i == 0:
                model1.fit(pred_train, cat_train)
                predictions1 = model1.predict(pred_test)
            elif i == 1:
                model2.fit(pred_train, cat_train)
                predictions2 = model2.predict(pred_test)
            elif i == 2:
                model3.fit(pred_train, cat_train)
                predictions3 = model3.predict(pred_test)
            elif i == 3:
                model4.fit(pred_train, cat_train)
                predictions4 = model4.predict(pred_test)
            #if count == int(values["-MODELSNUM-"]):
            #    resultsBool = True
    elif event == "-RESULTS-" and resultsBool:      # launch results window
        window.close()
        while True:
            event2, values2 = window2.read()
            if event2 == "Exit" or event2 == sg.WIN_CLOSED:
                break
    elif event == sg.WIN_CLOSED or event == "Exit" or event == '-FEATX-' or event == '-MODELX-':
        break


window.close()
