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

data = sio.loadmat("BCI/Subject_A_Train_reshaped.mat")

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
