#!/usr/bin/env python
# -*- coding: utf-8 -*-

from scipy.fftpack import fft, fftshift
from scipy import  sin, pi, ceil
import scipy.io as sio

import numpy as np

import matplotlib.pyplot as plt

class SignalHandler(object):
    """
    Native Object used to manipulate simple signal

    Use :
        mySignalHandler(signal, frequencySampling)

        You can :
            - Plot the signal, using .plotSignal()
            - Plot the Fourier Transform .plotFFT()
            - Plot the Short Term Fourier Transform .plotStft(windowCutting)

            - compute the Short Term Fourier Transform .stft()
            - compute the absolute shifted Fast Fourier Transform shifted .afft()

        The signal is stored in self.mainSignal

    """
    def __init__(self, signal, fs=240):
        self.mainSignal = signal

        self.fs = fs #frequency Sampling
        self.numPoints = len(self.mainSignal)
        self.duration = fs/self.numPoints

        self.x = np.linspace(0, self.duration, self.numPoints, endpoint=False)

    def plotSignal(self):
        plt.plot(self.x,self.mainSignal)
        plt.ylabel('mV')
        plt.xlabel('Time in S')
        plt.title("Signal")
        plt.grid()
        plt.show()

    def plotFft(self):
        y = self.afft()
        plt.plot(self.x,y)
        plt.show()

    def afft(self):
        """ Compute the absolute shifted Fast Fourier Transform """
        return np.abs(fftshift(fft(self.mainSignal)))

    def splitSignal(self,n):

        newN = ceil(self.numPoints/ n)
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
            mergedSTFT.extend(np.abs(fftshift(fft(w*frame))))

        return np.array(mergedSTFT)

    def plotStft(self, hanning=False, numWindow=4):

        powerSig = self.stft(hanning, numWindow)

        plt.plot(self.x,powerSig)
        plt.ylabel('FreqPower')
        plt.xlabel('Time in S')
        plt.grid()
        plt.show()


class SingleElectrodeProcessor(SignalHandler):

    def __init__(self, fileName="BCI/Subject_A_Train.mat", numTrial=0, numElec=0):

        data = sio.loadmat(fileName)
        self.targetLetters = data['TargetChar'][0]
        self.numTrial = numTrial
        self.numElec = numElec
        self.mainSignal = data['Signal'][numTrial,:,numElec]
        self.fs = 240 #Hz : Fréquence d'échantillonage du signal
        self.numPoints = len(self.mainSignal)
        self.duration = self.numPoints/self.fs

        self.x = np.linspace(0,self.duration,self.numPoints)

    def plotSTFT(self, hanning=False, numWindow=4):

        plt.title("Power Signal :\nLetter : {}  Elec = {}".format(self.targetLetters[self.numTrial], self.numElec))
        super().plotStft(self)

class MultipleElectrodeProcessor(SignalHandler):

    def __init__(self, fileName="BCI/Subject_A_Train.mat", numTrial=0, numElec=0):

        self.data = sio.loadmat(fileName)
        self.fs = 240 #Hz : Fréquence d'échantillonage du signal


def main():
    f0 = 10
    fs = 240
    T = 1

    x = np.linspace(0, fs*T, fs)
    signal = sin(2*pi*f0*x)

    # mySig = SignalHandler(signal, fs)
    # mySig.plotSignal()
    # mySig.plotFft()

    # mySig = SingleElectrodeProcessor(numTrial=10, numElec=10)
    # mySig.plotSignal()
    # mySig.plotFft()

if __name__ == '__main__':
    main()
