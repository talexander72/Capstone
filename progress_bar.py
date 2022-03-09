#import PySimpleGUI as sg
#import os
#from scipy.io.wavfile import read
#import pyACA
import librosa

#progress bar

#sg.theme('Black')

#BAR_MAX = 100

# layout the Window
#layoutB = [[sg.Text('custom progress')],
#          [sg.ProgressBar(BAR_MAX, orientation='h', size=(20,20), key='-PROG-')],
#          [sg.Cancel()]]

# create the Window
#window = sg.Window('Custom Progress Meter', layout)
# loop that would normally do something useful
#for i in range(1000):
    # check to see if the cancel button was clicked and exit loop if clicked
#    event, values = window.read(timeout=10)
#    if event == 'Cancel' or event == sg.WIN_CLOSED:
#        break
        # update bar with loop value +1 so that bar eventually reaches the maximum
#    window['-PROG-'].update(i+1)

#window.close()
#entries1 = os.listdir('/Users/tuckeralexander/Desktop/Classes/Capstone/data/MIMII/fan/id_00/normal')
#entries2 = os.listdir('/Users/tuckeralexander/Desktop/Classes/Capstone/data/MIMII/fan/id_00/abnormal')
#del(entries1[-1])
#del(entries2[-1])
#classA = []
#classB = []
#count=0
#print(len(entries1)+len(entries2))
#window2 = sg.Window('progress', layoutB)


#for i in range(1,10000):
#    sg.one_line_progress_meter('My Meter', i+1, 10000, 'key','Optional message',RETURN)
#    i = i+1
##    if i > 500:
#        break

