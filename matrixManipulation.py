#!/usr/bin/env python
# -*- coding: utf-8 -*-

from scipy.fftpack import fft, ifft, fftshift
from scipy import  sin, pi, ceil
import scipy.io as sio

import numpy as np
from numpy import ceil

import matplotlib.pyplot as plt

class SignalHandler(object):

    def __init__(self, signal, fs=240):

        self.mainSignal = signal

        self.fs = fs
        self.numPoints = len(self.mainSignal)
        self.duration = fs/self.numPoints

        self.x = np.linspace(0, self.duration, self.numPoints, endpoint=False)

    def plotSignal(self):

        plt.plot(self.x,self.mainSignal)
        plt.ylabel('mV')
        plt.xlabel('Time in S')
        plt.title("Signal")
        plt.grid()
        # plt.show()

    def splitSignal(self,n):
        newN = ceil(len(self.mainSignal) / n)
        for i in range(0, n-1):
            yield self.mainSignal[i*newN:i*newN+newN]
        yield self.mainSignal[n*newN-newN:]


    def stft(self, hanning=False, numWindow=4):

        splitedSignal = self.splitSignal(numWindow)

        mergedSTFT = []
        for frame in splitedSignal:
            if hanning:
                w = np.hanning(len(frame))
            else:
                w = [1 for i in range(len(frame))]
            mergedSTFT.extend(fftshift(fft(w*frame)))

        return np.array(mergedSTFT)


class ElectrodeManipulator(SignalHandler):

    def __init__(self, fileName="BCI/Subject_A_Train.mat", numTrial=0, numElec=0):

        self.data = sio.loadmat(fileName)
        print(self.data.keys())
        self.targetLetters = self.data['TargetChar'][0]
        self.numTrial = numTrial
        self.numElec = numElec
        self.mainSignal = self.data['Signal'][numTrial,:,numElec]
        self.fs = 240 #Hz : Fréquence d'échantillonage du signal
        self.numPoints = len(self.mainSignal)
        self.duration = self.numPoints/self.fs

        self.x = np.linspace(0,self.duration,self.numPoints)

    def plotPower(self, hanning=False, numWindow=4):

        powerSig = self.stft(hanning, numWindow)

        print(len(powerSig))

        plt.plot(self.x,powerSig)
        plt.ylabel('FreqPower')
        plt.xlabel('Time in S')
        plt.title("Power Signal :\nLetter : {}  Elec = {}".format(self.targetLetters[self.numTrial], self.numElec))
        plt.grid()
        plt.show()




def main():
    f0 = 10
    fs = 240
    T = 1

    x = np.linspace(0, fs*T, fs)
    signal = sin(2*pi*f0*x)

    mySig = SignalHandler(signal, fs)
    mySig.plotSignal()

    mySig = ElectrodeManipulator()
    mySig.plotPower()


if __name__ == '__main__':
    main()
