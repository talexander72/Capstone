
import PySimpleGUI as sg
from scipy.io.wavfile import read
import os
import pandas as pd
import numpy as np
import math


def chunk_fit(in1, chunksize, hopsize):      #in1 is list of each audio file
    fit = ((len(in1)-(chunksize + 1))/hopsize) + 1             #fit determines if chunk size divides evenly into the audio sample length
    if fit - math.floor(fit) != 0: #if fit value isn't zero, zeros need to be added to the audio so all chunks are same size
        add = ((math.ceil(fit)-1)*hopsize + chunksize + 1) - len(in1)
        add1 = (add)//2         #chunksize-fit equates number of zeros that need to be added
        add2 = (add)-add1       #add1 value is number of zeros before audio samples, add2 value is after
        add1 = np.zeros(add1, dtype=int)  #add1 and add2 values will either be equal or differ by one
        add2 = np.zeros(add2, dtype=int)
        fitted = [*add1, *in1, *add2]     #compiles and returns the fitted samples
    else:
        fitted = [*in1]
    return fitted


def chunker(fit_list, chunksize, hopsize):
    frames = {}
    for j in range(0, len(fit_list)):
        df = pd.DataFrame()
        fitted = fit_list[j]
        fitted = chunk_fit(fitted, chunksize, hopsize)
        numChunks = ((len(fitted)-(chunksize + 1))/hopsize) + 1
        for i in range(0, int(numChunks)):
            window = (hopsize/chunksize)
            chunk = fitted[hopsize*i:(hopsize*i)+chunksize]
            print(len(chunk))
            df[str(i)] = chunk
        frames[str(j)] = df
    return frames

#def features(feature, files):
#    specList = []
#    for i in files:
#        fs = i[0]
#        x = i[1]
#        [vsf, t] = computeFeature(feature, x, fs)
#        specList.append([vsf, t])
#    return specList


fList = ['SpectralCentroid', 'SpectralCrestFactor', 'SpectralDecrease', 'SpectralFlatness', 'SpectralFlux',
         'SpectralKurtosis', 'SpectralMfccs', 'SpectralPitchChroma', 'SpectralRolloff', 'SpectralSkewness',
         'SpectralSlope', 'SpectralSpread', 'SpectralTonalPowerRatio', 'TimeAcfCoeff', 'TimeMaxAcf',
         'TimePeakEnvelope', 'TimePredictivityRatio', 'TimeRms', 'TimeStd', 'TimeZeroCrossingRate']

BAR_MAX = 500
sg.theme('Black')
layout = \
    [[sg.Text("Path to Folder:"), sg.Input(key="-IN-", change_submits=True), sg.FolderBrowse(key="-IN-")],
    [sg.Text('Preprocessing: before feature extraction, audio must be "chunked" into equal size segments')],
    [sg.Text('Enter a Chunk Size (samples)'), sg.Slider(range=(100, 500), default_value=222, size=(40, 20), orientation='horizontal')],
    [sg.Text('Enter an Overlap Ratio (percent)'), sg.Slider(range=(0, 99), default_value=0, size=(40, 20), orientation='horizontal')],
    [sg.Text('Smaller datasets may benefit from cross validation. Implement K-Fold Cross Validation?)'), sg.Radio('Yes', "RADIO1"), sg.Radio('No', "RADIO1", default=True)],
    [sg.Button('Ok')], [sg.Button('Cancel')]]

window = sg.Window('Enhanced Feature Selection', layout, element_justification='c')


while True:
    event, values = window.read()
    print(values["-IN-"])
    if event == "Ok":
        print("You chose a chunk size of %s samples, with cross validation = %s" % (values[0], values[2]))
        classes = os.listdir(values["-IN-"])
        entries1 = os.listdir(values["-IN-"] + '/' + classes[1])
        entries2 = os.listdir(values["-IN-"] + '/' + classes[2])
        total = len(entries1) + len(entries2)
        BAR_MAX = total
        classA = []
        classB = []
        chunksA = []
        chunksB = []
        for i in range(total):
            if not sg.one_line_progress_meter('File Read Progress', i+1, total, 'File Read Progress'):
                break
            if i >= len(entries1):
                [fs, x2] = read(values["-IN-"] + '/' + classes[2] + '/' + entries2[i-len(entries1)])
                framesB = chunker(x2[:, 2], values[0], values[1])
                classB.append([fs, x2[:, 2]])
                chunksB.append(framesB[str(i-len(entries1))])
                i = i + 1
            else:
                [fs, x] = read(values["-IN-"] + '/' + classes[1] + '/' + entries1[i])
                framesA = chunker(x[:,2], values[0], values[1])
                classA.append([fs, x[:, 2]])
                chunksA.append(framesA[str(i)])
                i = i + 1
        #for j in range(total):
        #    if not sg.one_line_progress_meter('Chunking Progress', j+1, total, 'Chunking Progress'):
        #        break
        #    if j >= len(entries1):
        #        framesB = chunker(classB[j-len(entries1)][1], values[0], values[1])
        #        j = j + 1
        #    else:
        #        framesA = chunker(classA[j][1], values[0], values[1])
        #        j = j + 1
    elif event == sg.WIN_CLOSED or event == "Exit" or event == 'Cancel':
        break

window.close()
