#!/usr/bin/env python
# -*- coding: utf-8 -*

from rawManipulation import *
import sklearn as sk


def savePartOfData(filenameI, filenameO):
    data = sio.loadmat(filenameI)
    miniData = np.array(data['X'][0])
    miniData = miniData.transpose()
    y = data['y']
    np.save(filenameO, miniData)
    np.save('BCI/yFile', y)

def reformatRawData(filenameI):
    data = sio.loadmat(filenameI)
    data = np.array(data['X'])

    newData = []

    for exemple in np.size(data,3):
        newData.append(np.concatenate( (data[i,exemple,:] for i in range(64))))

    print( newData[5][:160]) )
    print( data[0,:,5] )






# savePartOfData("BCI/Subject_A_Train_reshaped.mat", "BCI/quickTest")

X = np.load("BCI/quickTest.npy")
y = np.load("BCI/yFile.npy")

p300 = np.where(y==1)

signal = X[p300[0][0]]
fs = 240

mySig = SignalHandler(signal, fs)
mySig.multiplePlot()

plt.show()


signal = X[p300[0][1]]
fs = 240

mySig = SignalHandler(signal, fs)
mySig.multiplePlot()

plt.show()
