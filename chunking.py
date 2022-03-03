import os
#from pydub import AudioSegment
import pandas as pd
import numpy as np
import math
from scipy.io.wavfile import read


def read_in(name, CV):
    file = os.listdir(name)
    del(file[0])
    filesize = len(file)
    if filesize < 2:
        print("The folder needs at least two subfolders")
        filelist = []
        filesize = []
    else:
        filelist = []
        for i in file:
            sub = os.listdir(name+str(i))
            for j in sub:
                if j.find(".wav") != -1:
                    continue
                else:
                    print("Only accepting .wav format")
                    return [],[]
            if len(sub) < 30:
                print("File "+str(i)+" has less than 30 files, would you like to use cross validation?")
                break
            else:
                filelist.append([i, len(sub), sub])
    return filelist, filesize


def chunk_fit(in1, chunksize, hopsize):      #in1 is list of each audio file
    fit = ((len(in1)-(chunksize + 1))/hopsize) + 1             #fit determines if chunk size divides evenly into the audio sample length
    if fit - math.floor(fit) != 0: #if fit value isn't zero, zeros need to be added to the audio so all chunks are same size
        add = ((math.ceil(fit)-1)*hopsize + chunksize + 1) - len(in1)
        add1 = (add)//2         #chunksize-fit equates number of zeros that need to be added
        add2 = (add)-add1       #add1 value is number of zeros before audio samples, add2 value is after
        add1 = np.zeros(add1, dtype=int)  #add1 and add2 values will either be equal or differ by one
        add2 = np.zeros(add2, dtype=int)
        fitted = [*add1, *in1, *add2]     #compiles and returns the fitted samples
    else:
        fitted = [*in1]
    return fitted


def chunker(fit_list, chunksize, hopsize):
    frames = {}
    for j in range(0, len(fit_list)):
        df = pd.DataFrame()
        fitted = fit_list[j]
        fitted = chunk_fit(fitted, chunksize, hopsize)
        numChunks = ((len(fitted)-(chunksize + 1))/hopsize) + 1
        for i in range(0, int(numChunks)):
            window = (hopsize/chunksize)
            chunk = fitted[hopsize*i:(hopsize*i)+chunksize]
            print(len(chunk))
            df[str(i)] = chunk
        frames[str(j)] = df
    return frames


name = '/Users/tuckeralexander/Desktop/Classes/Capstone/data/MIMII/fan/id_00/'
filelist, filesize = read_in(name, CV=False)

audio2 = []
for i in range(0, filesize):
    audio1 = []
    for j in filelist[i][2]:
        [fs, sound] = read(name + filelist[i][0] + "/" + str(j))
        sound = sound[:,3]
        sound = sound.tolist()
        audio1.append(sound)
    audio2.append([str(filelist[i][0]), audio1])

chunksize = 15000
hopsize = 6000
frames = chunker(audio2[0][1], chunksize, hopsize)
frames['1'].head()
