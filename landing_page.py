
import PySimpleGUI as sg
from scipy.io.wavfile import read
import os
import pandas as pd
import numpy as np
import math


def chunk_fit(in1, chunksize, hopsize):      # in1 is list of each audio file
    fit = ((len(in1)-(chunksize + 1))/hopsize) + 1             # fit determines if chunk size divides evenly into the audio sample length
    if fit - math.floor(fit) != 0:    # if fit value isn't zero, zeros need to be added to the audio so all chunks are same size
        add = ((math.ceil(fit)-1)*hopsize + chunksize + 1) - len(in1)
        add1 = add//2         # chunk size-fit equates number of zeros that need to be added
        add2 = add-add1       # add1 value is number of zeros before audio samples, add2 value is after
        add1 = int(add1)
        add2 = int(add2)
        add1 = np.zeros(add1)  # add1 and add2 values will either be equal or differ by one
        add2 = np.zeros(add2)
        fitted = [*add1, *in1, *add2]     # compiles and returns the fitted samples
    else:
        fitted = [*in1]
    return fitted


def chunker(fitted, chunksize, hopsize):
    # frames = pd.DataFrame()
    frames = []
    fitted = chunk_fit(fitted, chunksize, hopsize)
    numchunks = ((len(fitted)-(chunksize + 1))/hopsize) + 1
    for c in range(int(numchunks)):
        chunk = fitted[int(hopsize)*c:(int(hopsize)*c)+int(chunksize)]
        frames.append(chunk)
    return frames


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
                chunksB.append(framesB)
                i = i + 1
            else:
                [fs, x] = read(values["-IN-"] + '/' + classes[1] + '/' + entries1[i])
                framesA = chunker(x[:, 2], values[0], values[1])
                classA.append([fs, x[:, 2]])
                chunksA.append(framesA)
                i = i + 1
    elif event == sg.WIN_CLOSED or event == "Exit" or event == 'Cancel':
        break

window.close()
