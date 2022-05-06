
import PySimpleGUI as sg
from scipy.io.wavfile import read
import os

import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
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
    feature_names = feature_list

    def readData():
        chunk = values['-CHUNK-']
        hop = (1 - (values['-HOP-']) / 100) * chunk

        class_names = os.listdir(values['-PATH-'])
        filenames_1 = os.listdir(values['-PATH-'] + '/' + class_names[1])
        filenames_2 = os.listdir(values['-PATH-'] + '/' + class_names[2])
        num_files = len(filenames_1) + len(filenames_2)

        time_series_a = []  # initializing variables
        time_series_b = []
        bool_check = True

        for i in range(num_files):  # step 1 progress bar
            if not sg.one_line_progress_meter('File Reading Progress', i + 1, num_files, 'step 1 of 3: file reading'):
                break

            if i < len(filenames_1):    # processing class A
                [current_fs1, x] = read(values['-PATH-'] + '/' + class_names[1] + '/' + filenames_1[i])
                x = x[:, 0]  # grabbing only channel 1 of recording
                time_series_a.extend(x)
                if bool_check:  # grabbing the first returned sampling-rate to check subsequent values against
                    sampling_rate_check = current_fs1
                    bool_check = False
                else:
                    if sampling_rate_check != current_fs1:
                        raise Warning('Please make sure all files have the same sampling rate')

            else:   # processing class B
                [current_fs2, x2] = read(values['-PATH-'] + '/' + class_names[2] + '/' + filenames_2[i - len(filenames_1)])
                x2 = x2[:, 0]  # grabbing only channel 1
                time_series_b.extend(x2)
                if sampling_rate_check != current_fs2:
                    raise Warning('Please make sure all files have the same sampling rate')

        return chunk, hop, time_series_a, time_series_b, sampling_rate_check

    def getFeatures(time_series_a, time_series_b, chunk, hop, sampling_rate):

        def helpGetFeatures(feature, file, f_s):
            [vsf, t] = pyACA.computeFeature(feature, file, f_s, iBlockLength=chunk, iHopLength=hop)
            return vsf

        time_series_a = np.array(time_series_a)     # initializing variables
        time_series_b = np.array(time_series_b)
        features_a = []
        features_b = []

        for f in range(len(feature_names)):
            if not sg.one_line_progress_meter('Feature Extraction Progress', f + 1, len(feature_names),
                                              'step 2 of 3: feature extraction'):
                break
            current_feature_a = helpGetFeatures(feature_list[f], time_series_a, sampling_rate)
            features_a.append(current_feature_a)
            current_feature_b = helpGetFeatures(feature_list[f], time_series_b, sampling_rate)
            features_b.append(current_feature_b)

        return features_a, features_b

    def formatData():
        data1 = np.ones(len(features_list_a[0]))
        data2 = np.zeros(len(features_list_b[0]))
        data3 = np.concatenate((data1, data2))
        data5 = []
        for d in range(len(feature_names)):
            data4 = np.concatenate([features_list_a[d], features_list_b[d]])
            data5.append(data4)

        categories = pd.DataFrame(data3)
        predictors = pd.DataFrame(data5).transpose()
        predictors.columns = feature_names

        predictors_scaled = predictors.copy()  # normalization of audio features
        for i in feature_names:
            predictors_scaled[i] = (predictors_scaled[i] - predictors_scaled[i].min()) / \
                                   (predictors_scaled[i].max() - predictors_scaled[i].min())

        training_categories, testing_categories, training_predictors, testing_predictors \
            = train_test_split(categories, predictors_scaled, test_size=.2, random_state=25)

        return training_categories, testing_categories, training_predictors, testing_predictors

    def initializeModels():
        model1 = LogisticRegression(solver='lbfgs', max_iter=200)
        model2 = svm.SVC()
        model3 = KNeighborsClassifier(n_neighbors=3)
        model4 = RandomForestClassifier(max_depth=2, random_state=0)

        return model1, model2, model3, model4

    def testModels():
        score_list1 = 0  # initial conditions
        score_list2 = 0
        score_list3 = 0
        score_list4 = 0
        sfs1 = 0
        sfs2 = 0
        sfs3 = 0
        sfs4 = 0

        if values['-LR-']:  # if logistic regression model type is selected
            sfs1 = SFS(logistic_regression,
                       k_features=int(values['-TESTNUM-']),
                       forward=True,
                       floating=False,
                       verbose=2,
                       scoring='accuracy',
                       cv=0)
            sfs1 = sfs1.fit(pred_train, cat_train)

            score_list1 = []
            for p in range(1, int(values['-TESTNUM-'])+1):    # making predictions on each feature group of test set
                current_feature_set_names1 = np.array(sfs1.subsets_[p]['feature_names'])
                pred_train_sfs1 = pred_train[current_feature_set_names1]    # transform to desired feature space
                pred_test_sfs1 = pred_test[current_feature_set_names1]      # transform to desired feature space
                logistic_regression.fit(pred_train_sfs1, cat_train)      # fit the model to this feature space
                current_predictions1 = logistic_regression.predict(pred_test_sfs1)       # make predictions on test set
                current_scores1 = accuracy_score(cat_test, current_predictions1)    # accuracy of predictions
                score_list1.append(current_scores1)

        if values['-SVM-']:     # if support vector machine model type is selected
            sfs2 = SFS(support_vector_machine,
                       k_features=int(values['-TESTNUM-']),
                       forward=True,
                       floating=False,
                       verbose=2,
                       scoring='accuracy',
                       cv=0)
            sfs2 = sfs2.fit(pred_train, cat_train)

            score_list2 = []
            for p in range(1, int(values['-TESTNUM-'])+1):    # predictions for each feature set
                current_feature_set_names2 = np.array(sfs2.subsets_[p]['feature_names'])
                pred_train_sfs2 = pred_train[current_feature_set_names2]
                pred_test_sfs2 = pred_test[current_feature_set_names2]
                support_vector_machine.fit(pred_train_sfs2, cat_train)
                current_predictions2 = support_vector_machine.predict(pred_test_sfs2)
                current_scores2 = accuracy_score(cat_test, current_predictions2)
                score_list2.append(current_scores2)

        if values['-KNN-']:     # if K Nearest Neighbor model type is selected
            sfs3 = SFS(k_nearest_neighbor,
                       k_features=int(values['-TESTNUM-']),
                       forward=True,
                       floating=False,
                       verbose=2,
                       scoring='accuracy',
                       cv=0)
            sfs3 = sfs3.fit(pred_train, cat_train)
            score_list3 = []

            for p in range(1, int(values['-TESTNUM-'])+1):    # predictions for each feature set
                current_feature_set_names = np.array(sfs3.subsets_[p]['feature_names'])
                pred_train_sfs3 = pred_train[current_feature_set_names]
                pred_test_sfs3 = pred_test[current_feature_set_names]
                k_nearest_neighbor.fit(pred_train_sfs3, cat_train)
                current_predictions3 = k_nearest_neighbor.predict(pred_test_sfs3)
                current_scores3 = accuracy_score(cat_test, current_predictions3)
                score_list3.append(current_scores3)

        if values['-RF-']:      # if Random Forest model type is selected
            sfs4 = SFS(random_forest,
                       k_features=int(values['-TESTNUM-']),
                       forward=True,
                       floating=False,
                       verbose=2,
                       scoring='accuracy',
                       cv=0)
            sfs4 = sfs4.fit(pred_train, cat_train)
            score_list4 = []

            for p in range(1, int(values['-TESTNUM-'])+1):    # predictions for each feature set
                current_feature_set_names4 = np.array(sfs4.subsets_[p]['feature_names'])
                pred_train_sfs4 = pred_train[current_feature_set_names4]
                pred_test_sfs4 = pred_test[current_feature_set_names4]
                random_forest.fit(pred_train_sfs4, cat_train)
                current_predictions4 = random_forest.predict(pred_test_sfs4)
                current_scores4 = accuracy_score(cat_test, current_predictions4)
                score_list4.append(current_scores4)

        return score_list1, score_list2, score_list3, score_list4, sfs1, sfs2, sfs3, sfs4

    def drawFigure(canvas, figure):
        figure_canvas_agg = FigureCanvasTkAgg(figure, canvas)
        figure_canvas_agg.draw()
        figure_canvas_agg.get_tk_widget().pack(side='top', fill='both', expand=1)
        return figure_canvas_agg

    def generateSnippet():
        snip_read_data = [
            'def readData():',
            '    chunk = ' + values['-CHUNK-'],
            '    hop = ' + (1 - (values['-HOP-']) / 100) * chunk,
            '    class_names = ' + os.listdir(values['-PATH-']) +
            '    filenames_1 = ' + os.listdir(values['-PATH-'] + '/' + class_names[1]) +
            '    filenames_2 = ' + os.listdir(values['-PATH-'] + '/' + class_names[2]) +
            '    num_files = len(filenames_1) + len(filenames_2)',
            '    time_series_a = []  # initializing variables',
            '    time_series_b = []',
            '    bool_check = True',
            '    for i in range(num_files):  # step 1 progress bar',
            '        if i < len(filenames_1):    # processing class A',
            '            [current_fs1, x] = read(values[\'-PATH-\'] + \'/\' + class_names[1] + \'/\' + filenames_1[i])',
            '            x = x[:, 0]  # grabbing only channel 1 of recording',
            '            time_series_a.extend(x)',
            '            if bool_check:  # grabbing the first returned sampling-rate to check subsequent values against',
            '                sampling_rate_check = current_fs1',
            '                bool_check = False',
            '            else:',
            '                if sampling_rate_check != current_fs1:',
            '                    raise Warning(\'Please make sure all files have the same sampling rate\')',
            '        else:   # processing class B',
            '            [current_fs2, x2] = read(values[\'-PATH-\'] + \'/\' + class_names[2] + \'/\' + filenames_2[i - len(filenames_1)])',
            '            x2 = x2[:, 0]  # grabbing only channel 1',
            '            time_series_b.extend(x2)',
            '            if sampling_rate_check != current_fs2:',
            '                raise Warning(\'Please make sure all files have the same sampling rate\')',
            '    return chunk, hop, time_series_a, time_series_b, sampling_rate_check'
        ]

        snip_get_features = [
            'def getFeatures(time_series_a, time_series_b, chunk, hop, sampling_rate):',
            '    def helpGetFeatures(feature, file, f_s):',
            '        [vsf, t] = pyACA.computeFeature(feature, file, f_s, iBlockLength=chunk, iHopLength=hop)',
            '        return vsf',
            '    time_series_a = np.array(time_series_a)     # initializing variables',
            '    time_series_b = np.array(time_series_b)',
            '    features_a = []',
            '    features_b = []',
            '    feature_names = ' + sfs1.subsets_[4]['feature_names'] +
            '    for f in range(len(feature_names)):',
            '        current_feature_a = helpGetFeatures(feature_names[f], time_series_a, sampling_rate)',
            '        features_a.append(current_feature_a)',
            '        current_feature_b = helpGetFeatures(feature_names[f], time_series_b, sampling_rate)',
            '        features_b.append(current_feature_b)',
            '    return features_a, features_b'
        ]

        snip_format_data = [
            'def formatData():',
            '    data1 = np.ones(' + len(features_list_a[0]) +
            '    data2 = np.zeros(' + len(features_list_b[0]) +
            '    data3 = np.concatenate((data1, data2))',
            '    data5 = []',
            '    feature_names = ' + sfs1.subsets_[4]['feature_names'] +
            '    for d in range(len(feature_names)):',
            '        data4 = np.concatenate([features_list_a[d], features_list_b[d]])',
            '        data5.append(data4)',
            '    categories = pd.DataFrame(data3)',
            '    predictors = pd.DataFrame(data5).transpose()',
            '    predictors.columns = feature_names',
            '    predictors_scaled = predictors.copy()  # normalization of audio features',
            '    for i in feature_names:',
            '        predictors_scaled[i] = (predictors_scaled[i] - predictors_scaled[i].min())' + str(' / \\'),
            '                               (predictors_scaled[i].max() - predictors_scaled[i].min())',
            '    training_categories, testing_categories, training_predictors, testing_predictors \\',
            '        = train_test_split(categories, predictors_scaled, test_size=.2, random_state=25)',
            '    return training_categories, testing_categories, training_predictors, testing_predictors'

        ]

        snip_initialize_models = [
            'def initializeModels():',
            '    model1 = LogisticRegression(solver=\'lbfgs\', max_iter=200)',
            '    model2 = svm.SVC()',
            '    model3 = KNeighborsClassifier(n_neighbors=3)',
            '    model4 = RandomForestClassifier(max_depth=2, random_state=0)',
            '    return model1, model2, model3, model4'
        ]
        return snip_read_data, snip_get_features, snip_format_data, snip_initialize_models

    # main landing page layout
    sg.theme('Black')
    layout = \
        [[sg.Text('Path to Folder:'), sg.Input(key='-PATH-', change_submits=True), sg.FolderBrowse(key='-PATH-'),
          sg.Button('HELP', key='-UPLOAD_HELP-')],
         [sg.Text('__'*46)],

         [sg.Text('Enter a Chunk Size (samples):    '), sg.Slider(range=(256, 8192), key='-CHUNK-', default_value=2048,
                                                                  size=(40, 20), orientation='horizontal')],

         [sg.Text('Enter an Overlap Ratio (percent):'), sg.Slider(range=(0, 99), key='-HOP-', default_value=50,
                                                                  size=(40, 20), orientation='horizontal')],

         [sg.Text('Feature Extraction: Audio features are computed and averaged across chunks')],

         [sg.Text('     (Step 1 of 3) Parse Data:   '), sg.Button('LAUNCH', key='-PARSE-'),
          sg.Button('HELP', key='-HELP1-')],

         [sg.Text('__'*46)],

         [sg.Text('    (Step 2 of 3) Extract Features:'), sg.Button('LAUNCH', key='-FEAT-'),
          sg.Button('HELP', key='-HELP2-')],

         [sg.Text('__' * 46)],

         [sg.Text('Model Training: multiple machine learning models which each have their own "flavor" of prediction')],

         [sg.Text('How many features would you like to test?'), sg.Slider(range=(1, 18),
                                                                          key='-TESTNUM-', default_value=9,
                                                                          size=(40, 28), orientation='horizontal')],

         [sg.Text('Model Selection:')],
         [sg.Checkbox('Logistic Regression', default=True, key='-LR-')],
         [sg.Checkbox('K Nearest Neighbor', default=True, key='-KNN-')],
         [sg.Checkbox('Random Forest', default=True, key='-RF-')],
         [sg.Checkbox('Support Vector Machine', default=True, key='-SVM-')],

         [sg.Text('     (Step 3 of 3) Train Models:'), sg.Button('LAUNCH', key='-MODEL-'),
          sg.Button('HELP', key='-HELP2-')],

         [sg.Text('__'*46)],

         [sg.Text('View Results: '), sg.Button('GO', key='-RESULTS-')]
         ]

    # results page layout
    layout2 = \
        [[sg.Canvas(key='figCanvas')],
         [sg.Text('Optimization Options:')],
         [sg.Text('__'*30)],
         [sg.Checkbox('Highest Accuracy', default=True, key='-CODE1-')],
         [sg.Checkbox('Best Trade-off', default=True, key='-CODE2-')],
         [sg.Checkbox('Fastest Processing', default=True, key='-CODE3-')],
         [sg.Text('Generate Code Snippets:'), sg.Button('GET CODE', key='-CODEGEN-')]
         ]

    setup_window = sg.Window('Setup and Training', layout, element_justification='l')
    results_window = sg.Window('Results and Code Generation', layout2, modal=True, finalize=True)
    help_window1 = None
    help_window2 = None
    upload_help_window = None

    feature_bool = False
    model_bool = False      # deactivating buttons that aren't supposed to be used yet
    results_bool = False    # initializing results window in the background
    count = 0               # initializing tracker for window 2 launch event

    # launching landing page:
    while True:
        event, values = setup_window.read()
        if event == '-HELP1-':  # launching help window for stage 1
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

        elif event == '-HELP2-':    # launch help window for stage 2
            help_layout2 = \
                [[sg.Text(
                    'Stage 2 encompasses model training and sequential forward selection of features:')],
                 [sg.Text(
                    'Using the formatted and normalized data from stage 1, multiple machine learning models'
                    ' are trained.')],
                 [sg.Text(
                    'This program supports 4 types of ML models: Logistic Regression, K Nearest Neighbor, '
                    'Random Forest,')],
                 [sg.Text(
                    'and Support Vector Machine. Each of these models interpret training data')],
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

        elif event == '-UPLOAD_HELP-':
            upload_help_layout = \
                [[sg.Text(
                    'Upload the path to your dataset:')],
                 [sg.Text(
                    'Please ensure your dataset is in a folder containing folders for each target class.')],
                 [sg.Text(
                    'Please ensure that each subclass folder contains only .wav files of the same sampling rate')]
                 ]
            upload_help_window = sg.Window('Dataset Upload Help', upload_help_layout, finalize=True)
            if event == sg.WIN_CLOSED:
                upload_help_window.close()

        elif event == '-PARSE-':       # Launch Data Parsing
            feature_bool = True
            [chunk_size, hop_size, files_a, files_b, fs] = readData()

        elif event == '-FEAT-' and feature_bool:    # Launch Feature Extraction
            model_bool = True
            [features_list_a, features_list_b] = getFeatures(files_a, files_b, chunk_size, hop_size, fs)

        elif event == '-MODEL-' and model_bool:          # Launch Model Training / Testing
            [cat_train, cat_test, pred_train, pred_test] = formatData()
            [logistic_regression, support_vector_machine, k_nearest_neighbor, random_forest] = initializeModels()
            [lr_scores, svm_scores, knn_scores, rf_scores, lr_sfs, svm_sfs, knn_sfs, rf_sfs] = testModels()
            results_bool = True

        elif event == "-RESULTS-" and results_bool:          # launch results window
            setup_window.close()

            k_features = [1, 2, 3, 4, 5, 6, 7, 8, 9]
            fig = plt.figure()
            if values['-LR-']:
                plt.plot(k_features, lr_scores, label='Logistic Regression', marker='o')
            if values['-SVM-']:
                plt.plot(k_features, svm_scores, label='Support Vector Machine', marker='o')
            if values['-KNN-']:
                plt.plot(k_features, knn_scores, label='K Nearest Neighbor', marker='o')
            if values['-RF-']:
                plt.plot(k_features, rf_scores, label='Random Forest', marker='o')
            plt.ylim([.48, 1.02])
            plt.ylabel('Accuracy')
            plt.xlabel('Number of features')
            plt.title('Model Performance Evaluation')
            plt.grid()
            plt.tight_layout()
            plt.legend()
            draw_figure(results_window['figCanvas'].TKCanvas, fig)
            while True:
                event2, values2 = results_window.read()
                if event2 == "Exit" or event2 == sg.WIN_CLOSED:
                    break
                if event2 == '-CODEGEN-':
                    [snip1, snip2, snip3, snip4] = generateSnippet()
                    snip5 = [
                        '[chunk_size, hop_size, files_a, files_b, fs] = readData()',
                        '[features_list_a, features_list_b] = getFeatures(files_a, files_b, chunk_size, hop_size, fs)',
                        '[cat_train, cat_test, pred_train, pred_test] = formatData()',
                        '[logistic_regression, support_vector_machine, k_nearest_neighbor, random_forest] = initializeModels()',
                        ]

                    if values2['-CODE1-']:
                        optimal_num_features = 7
                        optimal_model = ['sfs' + optimal_num_features +
                                         '.subsets_[' + optimal_num_features + '][\'feature_names\']']
                        snip6 = [optimal_model + '.fit(pred_train, cat_train)']

                    if values2['-CODE1-']:
                        optimal_num_features = 5
                        optimal_model = ['sfs' + optimal_num_features +
                                         '.subsets_[' + optimal_num_features + '][\'feature_names\']']
                        snip6 = [optimal_model + '.fit(pred_train, cat_train)']

                    if values2['-CODE3-']:
                        optimal_num_features = 3
                        optimal_model = ['sfs' + optimal_num_features +
                                         '.subsets_[' + optimal_num_features + '][\'feature_names\']']
                        snip6 = [optimal_model + '.fit(pred_train, cat_train)']

        elif event == sg.WIN_CLOSED or event == "Exit":
            break

    setup_window.close()


if __name__ == "__main__":
    main()
