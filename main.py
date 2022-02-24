import PySimpleGUI as sg

sg.theme('Black')

layout = \
    [[sg.Text("Path to Folder:"), sg.Input(key="-IN-", change_submits=True), sg.FolderBrowse(key="-IN-")],
    [sg.Text('Preprocessing: before feature extraction, audio must be "chunked" into equal size segments')],
    [sg.Text('Enter a Chunk Size (samples)'), sg.Slider(range=(100, 500), default_value=222, size=(40, 20), orientation='horizontal')],
    [sg.Text('Enter a Hop Size (samples)'), sg.Slider(range=(100, 500), default_value=500, size=(40, 20), orientation='horizontal')],
    [sg.Text('Preprocessing: Would You Like to Cross Validate?)'), sg.Radio('Yes', "RADIO1"), sg.Radio('No', "RADIO1", default=True)],
    [sg.Button('Ok'), sg.Button('Cancel')]]

window = sg.Window('Enhanced Feature Selection', layout)
#test comment
while True:
    event, values = window.read()
    print(values["-IN-"])
    if event == sg.WIN_CLOSED or event == "Exit" or event == 'Cancel':
        break
    elif event == "Ok":
        print("You chose a chunk size of %s samples, with cross validation = %s" % (values[0], values[2]))

window.close()
