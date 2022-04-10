
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
from sklearn.metrics import accuracy_score

from itertools import combinations
from sklearn.base import clone
from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs


def main():

    feature_list = ['SpectralCentroid', 'SpectralCrestFactor', 'SpectralDecrease', 'SpectralFlatness', 'SpectralFlux',
                    'SpectralRolloff', 'SpectralSkewness', 'SpectralSpread', 'SpectralTonalPowerRatio']

    def extractFeatures(feature, file, f_s):
        [vsf, t] = pyACA.computeFeature(feature, file, f_s, iBlockLength=chunk_size, iHopLength=hop_size)
        return vsf

    model1 = LogisticRegression(solver='lbfgs', max_iter=200)  # binary logistic regression
    model2 = svm.SVC()
    model3 = KNeighborsClassifier(n_neighbors=3)
    model4 = RandomForestClassifier(max_depth=2, random_state=0)

    # main landing page layout
    sg.theme('Black')
    layout = \
        [[sg.Text('Path to Folder:'), sg.Input(key='-IN-', change_submits=True), sg.FolderBrowse(key='-IN-')],

         [sg.Text('__'*46)],

         [sg.Text('Enter a Chunk Size (samples):    '), sg.Slider(range=(256, 8192), key='-CHUNK-', default_value=2048,
                                                                  size=(40, 20), orientation='horizontal')],

         [sg.Text('Enter an Overlap Ratio (percent):'), sg.Slider(range=(0, 99), key='-HOP-', default_value=50,
                                                                  size=(40, 20), orientation='horizontal')],

         [sg.Text('Feature Extraction: Audio features are computed and averaged across chunks')],

         [sg.Text('How many features would you like to extract?'), sg.Slider(range=(1, 18), key='-FEATURESNUM-',
                                                                             default_value=9, size=(40,28),
                                                                             orientation='horizontal')],

         [sg.Text('    (Step 1 of 2) Parse Data:'), sg.Button('LAUNCH', key='-FEAT-'),
          sg.Button('HELP', key='-HELP1-')],

         [sg.Text('__'*46)],

         [sg.Text('Model Training: multiple machine learning models which each have their own "flavor" of prediction')],

         [sg.Text('How many features would you like to test?'), sg.Slider(range=(1, 18),
                                                                          key='-TESTNUM-', default_value=9,
                                                                          size=(40, 28), orientation='horizontal')],

         [sg.Text('Model Selection:')],
         [sg.Checkbox('Logistic Regression', default=True, key='-LR-')],
         [sg.Checkbox('KNN', default=True, key='-KNN-')],
         [sg.Checkbox('Random Forest', default=True, key='-RF-')],
         [sg.Checkbox('SVM', default=True, key='-SVM-')],
         [sg.Checkbox('MLP', default=True, key='-MLP-')],

         [sg.Text('    (Step 2 of 2) Train Models:'), sg.Button('LAUNCH', key='-MODEL-'),
          sg.Button('HELP', key='-HELP2-')],

         [sg.Text('__'*46)],

         [sg.Text('View Results: '), sg.Button('GO', key='-RESULTS-')]]

    # results page layout
    layout2 = \
        [[sg.Text('Optimization Curve:')],
         [sg.Text('__'*30)]]

    setup_window = sg.Window('Setup and Training', layout, element_justification='l')
    results_window = sg.Window('Results and Code Generation', layout2, modal=True)
    help_window1 = None
    help_window2 = None

    model_bool = False       # deactivating buttons that aren't supposed to be used yet
    results_bool = False     # initializing results window in the background
    count = 0               # initializing tracker for window 2 launch event

    # launching landing page:
    while True:

        event, values = setup_window.read()

        if event == '-HELP1-':        # launching first stage help button

            help_layout1 = \
                [[sg.Text(
                    'Stage 1 encompasses data parsing and feature extraction:')],
                 [sg.Text(
                     'First, files are read in from a user-selected folder. Each file is saved as a list of numbers')],
                 [sg.Text(
                     'representing air pressure over time, with an accompanying sample rate for reconstruction')],
                 [sg.Text(
                     'Machine learning models perform best when trained on consistently formatted data, so we split '
                     'each')],
                 [sg.Text(
                     'variable-length file into many "chunks" of the same length prior to feature extraction.')],
                 [sg.Text('__' * 46)],
                 [sg.Text(
                     'Audio feature extraction refers to the computation of meaningful information from a raw')],
                 [sg.Text(
                     'audio file. Things such as spectral energy, zero-crossing rate, center frequency, etc...')],
                 [sg.Text(
                     'are more meaningful than the raw audio, and thus allows for the training of a much more')],
                 [sg.Text(
                     'robust and accurate model')],
                 [sg.Text('__'*46)]
                 ]
            help_window1 = sg.Window("Data Parsing Help", help_layout1, finalize=True)
            if event == sg.WIN_CLOSED:
                help_window1.close()

        elif event == '-HELP2-':

            help_layout2 = \
                [[sg.Text(
                    'Stage 2 encompasses model training and sequential forward selection of features:')],
                 [sg.Text(
                    'Using the formatted and normalized data from stage 1, multiple machine learning models'
                    ' are trained.')],
                 [sg.Text(
                    'This program supports 5 types of ML models: Logistic Regression, K Nearest Neighbor, '
                    'Random Forest,')],
                 [sg.Text(
                    'Support Vector Machine, and Multilayer Perceptron. Each of these models interpret'
                    ' training data')],
                 [sg.Text(
                    'differently, and make their predictions accordingly.')],
                 [sg.Text('__' * 46)],
                 [sg.Text(
                    'Sequential Forward Selection is used to determine which features are the best descriptors'
                    ' of the ')],
                 [sg.Text(
                    'dataset being analyzed. Up to a user-selected number, this stage computes the accuracy')],
                 [sg.Text(
                    'of each model with increasing amounts of training features (beginning with the most'
                    ' "important" one')],
                 [sg.Text(
                    'This optimization is important because having more data is not always better. This'
                    ' phenomena is')],
                 [sg.Text(
                    'called over-fitting. Limiting the number of features used to train your model will improve'
                    ' processing')],
                 [sg.Text(
                    'speed as well as accuracy in many cases')],
                 [sg.Text('__' * 46)]]
            help_window2 = sg.Window('Model Evaluation Help', help_layout2, finalize=True)
            if event == sg.WIN_CLOSED:
                help_window2.close()

        elif event == '-FEAT-':       # Launch Data Parsing

            model_bool = True
            chunk_size = values['-CHUNK-']
            hop_size = (1 - (values['-HOP-'])/100) * chunk_size

            class_names = os.listdir(values['-IN-'])
            filenames_1 = os.listdir(values['-IN-'] + '/' + class_names[1])
            filenames_2 = os.listdir(values['-IN-'] + '/' + class_names[2])
            num_files = len(filenames_1) + len(filenames_2)

            features_a = []
            features_b = []
            feature_names = []

            for i in range(num_files):  # step 1 progress bar

                if not sg.one_line_progress_meter('Parsing Progress', i+1, num_files, 'step 1 of 2: data parsing'):
                    break

                if i < len(filenames_2):  # processing class A
                    [fs, x] = read(values['-IN-'] + '/' + class_names[1] + '/' + filenames_1[i])
                    x = x[:, 0]  # grabbing only channel 1

                else:   # processing class B
                    [fs, x2] = read(values['-IN-'] + '/' + class_names[2] + '/' + filenames_2[i - len(filenames_1)])
                    x2 = x2[:, 0]  # grabbing only channel 1

            for f in range(int(values['-FEATURESNUM-'])):
                feature_names.append(feature_list[f])
                current_feature_a = extractFeatures(feature_list[f], x, fs)
                features_a.append(current_feature_a)
                current_feature_b = extractFeatures(feature_list[f], x2, fs)
                features_b.append(current_feature_b)

        elif event == '-MODEL-' and model_bool:          # Launch Model Training / Testing

            data1 = np.ones(len(features_a[0]))
            data2 = np.zeros(len(features_b[0]))
            data3 = np.concatenate((data1, data2))
            data5 = []
            for d in range(int(values['-FEATURESNUM-'])):
                data4 = np.concatenate([features_a[d], features_b[d]])
                data5.append(data4)

            categories = pd.DataFrame(data3)
            predictors = pd.DataFrame(data5).transpose()
            predictors.columns = feature_names

            predictors_scaled = predictors.copy()  # normalization of audio features
            for i in feature_names:
                predictors_scaled[i] = (predictors_scaled[i] - predictors_scaled[i].min()) / \
                                            (predictors_scaled[i].max() - predictors_scaled[i].min())

            cat_train, cat_test, pred_train, pred_test = train_test_split(categories, predictors_scaled, test_size=.2,
                                                                          random_state=25)

            for i in range(0, int(values['-MODELSNUM-'])):  # THIS LOOP NEEDS TO BE REDONE **********************
                count = count + 1

                if not sg.one_line_progress_meter('Model Evaluation Progress', i+1, int(values["-MODELSNUM-"]),
                                                  'step 2 of 2: testing model performance'):
                    break

                if i == 0:
                    sfs1 = SFS(model1, k_features=int(values["-TESTNUM-"]), forward=True,
                                                     floating=False, scoring='accuracy', cv=0)
                    pipe1 = make_pipeline(StandardScaler(), sfs1)
                    pipe1.fit(pred_train, cat_train)
                    pred_train_sfs1 = sfs1.transform(pred_train)
                    pred_test_sfs1 = sfs1.transform(pred_test)
                    model1.fit(pred_train_sfs1, cat_train)
                    predictions1 = model1.predict(pred_test_sfs1)
                    score = accuracy_score(cat_test, predictions1)
                elif i == 1:
                    sfs2 = SFS(model2, k_features=int(values["-TESTNUM-"]), forward=True,
                                                     floating=False, verbose=2, scoring='accuracy', cv=5)
                    pipe2 = make_pipeline(StandardScaler(), sfs2)
                    pipe2.fit(pred_train, cat_train)
                elif i == 2:
                    sfs3 = SFS(model3, k_features=int(values["-TESTNUM-"]), forward=True,
                                                     floating=False, verbose=2, scoring='accuracy', cv=5)
                    pipe3 = make_pipeline(StandardScaler(), sfs3)
                    pipe3.fit(pred_train, cat_train)
                elif i == 3:
                    sfs4 = SFS(model4, k_features=int(values["-TESTNUM-"]), forward=True,
                                                     floating=False, verbose=2, scoring='accuracy', cv=5)
                    pipe4 = make_pipeline(StandardScaler(), sfs4)
                    pipe4.fit(pred_train, cat_train)
                if count == int(values["-MODELSNUM-"]):
                    results_bool = True

        elif event == "-RESULTS-" and results_bool:          # launch results window
            setup_window.close()
            plot_sfs(sfs1.get_metric_dict(), kind='std_err');
            plt.ylim([0.8, 1])
            plt.title('Sequential Forward Selection (w. StdDev)')
            plt.grid()
            plt.show()
            while True:
                event2, values2 = results_window.read()
                if event2 == "Exit" or event2 == sg.WIN_CLOSED:
                    break
        elif event == sg.WIN_CLOSED or event == "Exit":
            break

    setup_window.close()


if __name__ == "__main__":
    main()
