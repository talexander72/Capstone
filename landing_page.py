
import PySimpleGUI as sg
from scipy.io.wavfile import read
import os
import pandas as pd
import numpy as np
import math
import pyAudioAnalysis
#from sklearn.model_selection import StratifiedKFold, KFold


def chunk_fit(in1, chunksize, hopsize):     # in1 is current audio file
    # adds zero padding if necessary to current audio file
    fit = ((len(in1)-(chunksize + 1))/hopsize) + 1
    if fit - math.floor(fit) != 0:
        add = ((math.ceil(fit)-1)*hopsize + chunksize + 1) - len(in1)
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


def chunker(fitted, chunksize, hopsize):
    # divides current audio file into equal sized segments
    frames = []
    fitted = chunk_fit(fitted, chunksize, hopsize)
    numchunks = ((len(fitted)-(chunksize + 1))/hopsize) + 1
    for c in range(int(numchunks)):
        chunk = fitted[int(hopsize)*c:(int(hopsize)*c)+int(chunksize)]
        frames.append(chunk)
    return frames


def open_window(format):
    window = sg.Window("Results and Training", format, modal=True)
    choice = None
    while True:
        event, values = window.read()
        if event == "Exit" or event == sg.WIN_CLOSED:
            break

    window.close()

# main landing page layout
sg.theme('Black')
layout = \
    [[sg.Text("Path to Folder:"), sg.Input(key="-IN-", change_submits=True), sg.FolderBrowse(key="-IN-")],
     [sg.Text('__'*30)],
     [sg.Text('Preprocessing Pipeline: before feature extraction, audio must be "chunked" into equal size segments')],
     [sg.Text('Enter a Chunk Size (samples):    '), sg.Slider(range=(256, 8192), key="-CHUNK-", default_value=2048, size=(40, 20), orientation='horizontal')],
     [sg.Text('Enter an Overlap Ratio (percent):'), sg.Slider(range=(0, 99), key="-HOP-", default_value=50, size=(40, 20), orientation='horizontal')],
     [sg.Text('Smaller datasets may benefit from cross validation. Implement K-Fold Cross Validation?)'), sg.Radio('Yes', "RADIO1"), sg.Radio('No', "RADIO1", default=True)],
     [sg.Text('    (Step 1 of 3) Launch Preprocessing:    '), sg.Button('LAUNCH', key="-READ-"), sg.Button('Cancel', key="-READX-")],
     [sg.Text('__'*30)],
     [sg.Text('Feature Extraction: Audio features are computed and averaged across chunks')],
     [sg.Text('How many features would you like to test?'), sg.Slider(range=(1,10), default_value=10, size=(40,28), orientation='horizontal')],
     [sg.Text('    (Step 2 of 3) Launch Feature Extraction:'), sg.Button('LAUNCH', key="-FEAT-"), sg.Button('Cancel', key="-FEATX-")],
     [sg.Text('__'*30)],
     [sg.Text('Model Training: multiple machine learning models which each have their own "flavor" of prediction')],
     [sg.Text('How many models would you like to train?'), sg.Slider(range(1,5), default_value=5, size=(40,28), orientation='horizontal')],
     [sg.Text('    (Step 3 of 3) Launch Model Training:     '), sg.Button('LAUNCH', key="-MODEL-"), sg.Button('Cancel', key="-MODELX-")]]

window = sg.Window('Enhanced Feature Selection', layout, element_justification='l')
featuresbool = False
modelbool = False   # deactivating buttons that aren't supposed to be used yet

# launching landing page
while True:
    event, values = window.read()
    if event == "-READ-":   # upon hitting step 1 'OK' button
        featuresbool = True     # activates step 2 'OK' button
        chunksize = values["-CHUNK-"]
        hopsize = (1 - (values["-HOP-"])/100) * chunksize
        classes = os.listdir(values["-IN-"])
        entries1 = os.listdir(values["-IN-"] + '/' + classes[1])
        entries2 = os.listdir(values["-IN-"] + '/' + classes[2])
        total = len(entries1) + len(entries2)
        classA = []
        classB = []
        chunksA = []
        chunksB = []
        for i in range(total):  # step 1 progress bar
            if not sg.one_line_progress_meter('File Read Progress', i+1, total, 'step 1 of 3: parsing data'):
                break
            if i >= len(entries1):  # processing class B
                [fs, x2] = read(values["-IN-"] + '/' + classes[2] + '/' + entries2[i-len(entries1)])
                framesB = chunker(x2[:, 2], chunksize, hopsize)
                classB.append([fs, x2[:, 2]])
                chunksB.append(framesB)
                i = i + 1
            else:   # processing class A
                [fs, x] = read(values["-IN-"] + '/' + classes[1] + '/' + entries1[i])
                framesA = chunker(x[:, 2], chunksize, hopsize)
                classA.append([fs, x[:, 2]])
                chunksA.append(framesA)
                i = i + 1
    elif event == "-FEAT-" and featuresbool:    # upon hitting step 2 'OK' button
        for i in range(total):
            if not sg.one_line_progress_meter('File Read Progress', i+1, total, 'step 2 of 3: extracting features'):
                break
            #for f in range(10):
    elif event == sg.WIN_CLOSED or event == "Exit" or event == '-READX-' or event == '-FEATX-':
        break


window.close()
