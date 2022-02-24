import PySimpleGUI as sg
from scipy.io.wavfile import read
import os
#import pyACA


#def features(feature, files):
#    specList = []
#    for i in files:
#        fs = i[0]
#        x = i[1]
#        [vsf, t] = pyACA.computeFeature(feature, x, fs)
#        specList.append([vsf, t])
#    return specList


#fList = ['SpectralCentroid', 'SpectralCrestFactor', 'SpectralDecrease', 'SpectralFlatness', 'SpectralFlux',
#         'SpectralKurtosis', 'SpectralMfccs', 'SpectralPitchChroma', 'SpectralRolloff', 'SpectralSkewness',
#         'SpectralSlope', 'SpectralSpread', 'SpectralTonalPowerRatio', 'TimeAcfCoeff', 'TimeMaxAcf',
#         'TimePeakEnvelope', 'TimePredictivityRatio', 'TimeRms', 'TimeStd', 'TimeZeroCrossingRate']
BAR_MAX = 1000
sg.theme('Black')
layout = \
    [[sg.Text("Path to Folder:"), sg.Input(key="-IN-", change_submits=True), sg.FolderBrowse(key="-IN-")],
    [sg.Text('Preprocessing: before feature extraction, audio must be "chunked" into equal size segments')],
    [sg.Text('Enter a Chunk Size (samples)'), sg.Slider(range=(100, 500), default_value=222, size=(40, 20), orientation='horizontal')],
    [sg.Text('Enter a Hop Size (samples)'), sg.Slider(range=(100, 500), default_value=500, size=(40, 20), orientation='horizontal')],
    [sg.Text('Preprocessing: Would You Like to Cross Validate?)'), sg.Radio('Yes', "RADIO1"), sg.Radio('No', "RADIO1", default=True)],
    [sg.Button('Ok'), sg.Button('Cancel')],
    [[sg.Text('file read progress:')],
    [sg.ProgressBar(BAR_MAX, orientation='h', size=(20, 20), key='-PROG-')]]]

window = sg.Window('Enhanced Feature Selection', layout)

while True:
    event, values = window.read()
    print(values["-IN-"])
    if event == sg.WIN_CLOSED or event == "Exit" or event == 'Cancel':
        break
    elif event == "Ok":
        print("You chose a chunk size of %s samples, with cross validation = %s" % (values[0], values[2]))
        classes = os.listdir(values["-IN-"])
        entries1 = os.listdir(values["-IN-"] + '/' + classes[1])
        entries2 = os.listdir(values["-IN-"] + '/' + classes[2])
        total = len(entries1) + len(entries2)
        BAR_MAX = total
        classA = []
        classB = []
        for k in range(total):
            if k >= len(entries1):
                [fs, x2] = read(values["-IN-"] + '/' + classes[2] +'/' + entries2[k-len(entries1)])
                classB.append([fs, x[:, 2]])
                window['-PROG-'].update(k+1)
            else:
                [fs, x] = read(values["-IN-"] + '/' + classes[1] +'/' + entries1[k])
                classA.append([fs, x[:, 2]])
                window['-PROG-'].update(k+1)


window.close()
