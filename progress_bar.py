import PySimpleGUI as sg

# progress bar

sg.theme('Black')
BAR_MAX = 100
layout = [[sg.Text('A custom progress meter')],
          [sg.ProgressBar(BAR_MAX, orientation='h', size=(20,20), key='-PROG-')],
          [sg.Cancel()]]

window = sg.Window('Custom Progress Meter', layout)

for i in range(1000):
    event, values = window.read(timeout=10)
    if event == 'Cancel' or event == sg.WIN_CLOSED:
        break
    window['-PROG-'].update(i+1)

window.close()
