
import PySimpleGUI as sg
from scipy.io.wavfile import read
import os
import pandas as pd
import numpy as np
import math
import pyACA
from sklearn.model_selection import StratifiedKFold, KFold


def chunk_fit(in1, chunk_size, hop_size):     # in1 is current audio file
    # adds zero padding if necessary to current audio file
    fit = ((len(in1)-(chunk_size + 1))/hop_size) + 1
    if fit - math.floor(fit) != 0:
        add = ((math.ceil(fit)-1)*hop_size + chunk_size + 1) - len(in1)
        add1 = add//2
        add2 = add-add1     # add1 = number of zeros before audio, add2 = number of zeros after audio
        add1 = int(add1)
        add2 = int(add2)
        add1 = np.zeros(add1)
        add2 = np.zeros(add2)
        fitted = [*add1, *in1, *add2]
    else:
        fitted = [*in1]
    return fitted


def chunker(fitted, chunk_size, hop_size):
    # divides current audio file into equal sized segments
    frames = []
    fitted = chunk_fit(fitted, chunk_size, hop_size)
    num_chunks = ((len(fitted)-(chunk_size + 1))/hop_size) + 1
    for c in range(int(num_chunks)):
        chunk = fitted[int(hop_size)*c:(int(hop_size)*c)+int(chunk_size)]
        frames.append(chunk)
    return frames


fList = ['SpectralCentroid', 'SpectralCrestFactor', 'SpectralDecrease', 'SpectralFlatness', 'SpectralFlux',
         'SpectralKurtosis', 'SpectralMfccs', 'SpectralPitchChroma', 'SpectralRolloff', 'SpectralSkewness',
         'SpectralSlope', 'SpectralSpread', 'SpectralTonalPowerRatio', 'TimeAcfCoeff', 'TimeMaxAcf', 'TimePeakEnvelope',
         'TimePredictivityRatio', 'TimeRms', 'TimeStd', 'TimeZeroCrossingRate']


def features(feature, file, f_s):
    [vsf, t] = pyACA.computeFeature(feature, file, f_s, iBlockLength=chunkSize, iHopLength=hopSize)
    return vsf


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
     [sg.Text('How many models would you like to train?'), sg.Slider(range=(1, 5),
                                                                     key="-MODELSNUM-", default_value=5,
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
                    featuresA.append(current_feature)
                i = i + 1
            else:   # processing class A
                [fs, x] = read(values["-IN-"] + '/' + classes[1] + '/' + entries1[i])
                for f in range(int(values["-FEATURESNUM-"])):
                    current_feature = features(fList[f], x, fs)
                    current_feature[:] = current_feature[:] / max(current_feature)
                    featuresB.append(current_feature)
                i = i + 1
    elif event == "-MODEL-" and modelBool:     # upon hitting step 2 'Launch' button
        for i in range(total):
            count = count + 1
            if not sg.one_line_progress_meter('Model Training Progress', i+1, total, 'step 2 of 2: training machine learning models'):
                break

            elif count == total:
                resultsBool = True
    elif event == "-RESULTS-" and resultsBool:      # launch results window
        window.close()
        while True:
            event2, values2 = window2.read()
            if event2 == "Exit" or event2 == sg.WIN_CLOSED:
                break
    elif event == sg.WIN_CLOSED or event == "Exit" or event == '-READX-' or event == '-FEATX-':
        break


window.close()
