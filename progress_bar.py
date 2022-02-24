import PySimpleGUI as sg
import os
from scipy.io.wavfile import read
#import pyACA

#progress bar

sg.theme('Black')

BAR_MAX = 100

# layout the Window
layoutB = [[sg.Text('custom progress')],
          [sg.ProgressBar(BAR_MAX, orientation='h', size=(20,20), key='-PROG-')],
          [sg.Cancel()]]

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
entries1 = os.listdir('/Users/tuckeralexander/Desktop/Classes/Capstone/data/MIMII/fan/id_00/normal')
entries2 = os.listdir('/Users/tuckeralexander/Desktop/Classes/Capstone/data/MIMII/fan/id_00/abnormal')
del(entries1[-1])
del(entries2[-1])
classA = []
classB = []
count=0
print(len(entries1)+len(entries2))
window2 = sg.Window('progress', layoutB)

for k in range(len(entries1)+len(entries2)):
    event, values = window2.read(timeout=10)
    if event == 'Cancel' or event == sg.WIN_CLOSED:
        break
    [fs, x] = read("/Users/tuckeralexander/Desktop/Classes/Capstone/data/MIMII/fan/id_00/normal/" + entries1[k])
    classA.append([fs, x[:, 2]])
    count = count+1
    window2['-PROG-'].update(k + 1)

    while k >= len(entries1):
        [fs, x2] = read("/Users/tuckeralexander/Desktop/Classes/Capstone/data/MIMII/fan/id_00/abnormal/" + entries2[k-len(entries1)])
        classB.append([fs, x[:, 2]])
        count = count+1
        window2['-PROG-'].update(k + 1)
    #if divmod(count,2) == 0:
        #window2['-PROG-'].update(k + 1)
window2.close()
