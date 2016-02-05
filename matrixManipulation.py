#!/usr/bin/env python
# -*- coding: utf-8 -*-

from scipy.fftpack import fft, fftshift
from scipy import  sin, pi, ceil
import scipy.io as sio

import numpy as np

import matplotlib.pyplot as plt

from copy import copy, deepcopy

class SignalHandler(object):
    """
    Native Object used to manipulate simple signal

    Use :
        mySignalHandler(signal, frequencySampling)

        You can :
            - compute the Short Term Fourier Transform .stft()
            - compute the absolute shifted Fast Fourier Transform shifted .afft()

            - Plot the signal, using .plotSignal()
            - Plot the Fourier Transform .plotFFT()
            - Plot the Short Term Fourier Transform .plotStft(windowCutting)

        The signal is stored in self.mainSignal
        This is the only signal that should be MANIPULATED
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
    """
    Class to manipulated one electrode

    Not supposed to be used alone
    -----------------------------
    Prefer MultipleElectrodeProcessor

    You can (From Signal handler) :

        - compute the Short Term Fourier Transform .stft()
        - compute the absolute shifted Fast Fourier Transform shifted .afft()

        - Plot the signal, using .plotSignal()
        - Plot the Fourier Transform .plotFFT()
        - Plot the Short Term Fourier Transform .plotStft(windowCutting)
    """

    def __init__(self, fileName="BCI/Subject_A_Train.mat", numSession=0, numElec=0):

        data = sio.loadmat(fileName)
        self.allSignalUnformated = data['Signal']
        self.targetLetters = data['TargetChar'][0]
        self.numSession = numSession
        self.numElec = numElec
        self.mainSignal = data['Signal'][numSession,:,numElec]
        self.fs = 240 #Hz : Fréquence d'échantillonage du signal
        self.numPoints = len(self.mainSignal)
        self.duration = self.numPoints/self.fs

        print(self.numPoints)

        self.x = np.linspace(0,self.duration,self.numPoints)

    def plotSTFT(self, hanning=False, numWindow=4):

        plt.title("Power Signal :\nLetter : {}  Elec = {}".format(self.targetLetters[self.numSession], self.numElec))
        super().plotStft(self)

class MultipleElectrodeProcessor(SingleElectrodeProcessor):

    def __init__(self, fileName="BCI/Subject_A_Train.mat"):
        super().__init__(fileName=fileName)
        self.allSignalFormated = []
        self.cutAllSignals()

    def cutAllSignals(self):
        """
        Format used self.allSignalFormated :
        [   NUM SESSION,  NUM ELEC, NUMTRIAL,      POINT     ]
         85train 100test | (0,64) |  (0,15)  | (0,504) (0,522)
        """

        for numSession, session in enumerate(self.allSignalUnformated):
            sessionSignals = []

            for numElec in range(64):
                self.mainSignal = session[:,numElec]

                #Cut the signal
                sessionSignals.append(self.signalCutting())

            self.allSignalFormated.append(sessionSignals)

        self.allSignalFormated = np.array(self.allSignalFormated)

        assert len(self.allSignalFormated) == 85
        assert len(self.allSignalFormated[0]) == 64
        assert len(self.allSignalFormated[0][0]) == 15
        assert len(self.allSignalFormated[0][0][-1]) == 504



    def signalCutting(self):
        """
        Function to Process the signal sent
        Input : One session of 7794 points (32,475s)
        Ouput : List of Trial (504 or 522 points)

        One session look like this :

        12*175 ms : First Sample 2100 ms : 504 Points
        12*175 ms : Last Sample 2100 ms : 504 Points

        13*(13*75+12*100) ms : 2-14 Samples : 13 x 2175 ms : 13 x 522 points

        Total : 32.475 s = 7794 points at 240Hz
        """

        oldSignal = self.mainSignal
        newSignal = [ oldSignal[:504] ] #First Trial

        for trial in range(13):
            newSignal.append( oldSignal[504+trial*522:504+(trial+1)*522] )

        newSignal.append( oldSignal[504+13*522:])

        assert len(newSignal)==15
        print('signalCut')

        return newSignal



    def meanSignal(self):
        pass
        #TODO


def main():
    pass



if __name__ == '__main__':
    main()
