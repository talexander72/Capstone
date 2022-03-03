
#from pyACA import computeFeature
import PySimpleGUI as sg
from scipy.io.wavfile import read
import os
#import scipy
#import sklearn




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
        for i in range(total):
            if not sg.one_line_progress_meter('File Read Progress', i+1, total, 'File Read Progress', ):
                break
            if i >= len(entries1):
                [fs, x2] = read(values["-IN-"] + '/' + classes[2] + '/' + entries2[i-len(entries1)])
                classB.append([fs, x[:, 2]])
                i = i + 1
            else:
                [fs, x] = read(values["-IN-"] + '/' + classes[1] + '/' + entries1[i])
                classA.append([fs, x[:, 2]])
                i = i + 1
    elif event == sg.WIN_CLOSED or event == "Exit" or event == 'Cancel':
        break


window.close()
