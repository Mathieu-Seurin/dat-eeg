#!/usr/bin/env python
# -*- coding: utf-8 -*-

from scipy.fftpack import fft, fftshift, fftfreq
from scipy import  sin, pi, ceil, array, absolute
import scipy.io as sio

import numpy as np

import matplotlib.pyplot as plt

from copy import copy, deepcopy

FRAMESIZE = 0.10
HOPSIZE = 0.05

class SignalHandler(object):
    """
    Native Object used to manipulate a single signal
    ---------------------------------------------

    Use :
        mySignalHandler(signal, frequencySampling)

        You can :
            - compute the Short Term Fourier Transform .stft()
            - compute the absolute shifted Fast Fourier Transform shifted .afft()

            - Plot the signal, using .plot()
            - Plot the Fourier Transform .plotFFT()
            - Plot the Short Term Fourier Transform .plotStft(windowCutting)

        The signal is stored in self.mainSignal
        This is the only signal that should be MANIPULATED
    """
    def __init__(self, signal, fs=240):
        self.mainSignal = signal

        self.fs = fs #frequency Sampling
        self.numPoints = len(self.mainSignal)
        self.duration = self.numPoints/fs

        self.x = np.linspace(0, self.duration, self.numPoints, endpoint=False)

    def plot(self):
        plt.plot(self.x,self.mainSignal)
        plt.ylabel('mV')
        plt.xlabel('Time in S')
        plt.grid()

    def plotFft(self):
        y = abs(fft(self.mainSignal))
        x = fftfreq(len(self.mainSignal), 1.0/240)
        plt.plot(x,y)

    def afft(self, signal=None):
        """ Compute the absolute shifted Fast Fourier Transform """
        if signal!=None:
            padding = np.zeros(np.size(signal,0))
            signal = np.concatenate((signal, padding))
            spectrum = fft(signal)
            return np.abs(spectrum)
        else:
            padding = np.zeros(np.size(self.mainSignal,0))
            signal = np.concatenate((self.mainSignal, padding))
            spectrum = fft(signal)
            return np.abs(spectrum)



    def stft(self, hanning=True, frameSize=FRAMESIZE, hop=HOPSIZE):

        frameAmp = int(self.fs*frameSize)
        hopAmp = int(self.fs*hop)

        if hanning:
            w = np.hanning(frameAmp)
        else:
            w = [1 for i in range(frameAmp)]

        mergedSTFT = []

        for i in range(0, len(self.mainSignal)-frameAmp, hopAmp):
            fftshifted = self.afft(w*self.mainSignal[i:i+frameAmp])
            mergedSTFT.append(fftshifted[:len(fftshifted)//2]/frameAmp)

        print(len(mergedSTFT[0]))
        print(mergedSTFT[0])

        return array(mergedSTFT).transpose()

    def plotStft(self, hanning=True,frameSize=FRAMESIZE, hop=HOPSIZE):

        T = len(self.mainSignal)/self.fs
        powerSig = self.stft(hanning,frameSize,hop)
        freq = fftfreq(len(powerSig[0]), 1.0/240)

        plt.imshow(powerSig, origin='lower', extent=[0,T,0,120],aspect='auto')
        plt.ylabel('Frequency')
        plt.xlabel('Time in S')

    def multiplePlot(self):

        plt.subplot(4,2,1) #NORMAL
        self.plot()

        plt.subplot(4,2,2) #FFT
        self.plotFft()

        plt.subplot(4,2,3) #STFT hanning
        self.plotStft(hanning=True)

        plt.subplot(4,2,4) #STFT
        self.plotStft(hanning=False)

        plt.subplot(4,2,5) #STFT NO OVERLAP
        self.plotStft(hanning=False,frameSize=FRAMESIZE, hop=FRAMESIZE)


        plt.subplot(4,2,6) #STFT Shorter Window
        self.plotStft(hanning=True,frameSize=FRAMESIZE/2, hop=HOPSIZE/2)

        plt.subplot(4,2,7) #STFT Shorter Window
        self.plotStft(hanning=True,frameSize=FRAMESIZE/4, hop=HOPSIZE/4)

        plt.subplot(4,2,8) #STFT Shorter Window
        self.plotStft(hanning=True,frameSize=FRAMESIZE/8, hop=HOPSIZE/8)


class SingleElectrodeProcessor(SignalHandler):
    """
    Class to manipulated one electrode
    ----------------------------------

    **Not supposed to be used alone**
    Prefer MultipleElectrodeProcessor

    You can (From Signal handler) :

        - compute the Short Term Fourier Transform .stft()
        - compute the absolute shifted Fast Fourier Transform shifted .afft()

        - Plot the signal, using .plot()
        - Plot the Fourier Transform .plotFFT()
        - Plot the Short Term Fourier Transform .plotStft(windowCutting)
    """

    def __init__(self, fileName="BCI/Subject_A_Train.mat", numSession=0, numElec=0):

        self.data = sio.loadmat(fileName)
        self.allSignalUnformated = self.data['Signal']
        self.targetLetters = self.data['TargetChar'][0]
        self.numSession = numSession
        self.numElec = numElec
        self.mainSignal = self.data['Signal'][numSession,:,numElec]
        self.fs = 240 #Hz : Fréquence d'échantillonage du signal
        self.numPoints = len(self.mainSignal)
        self.duration = self.numPoints/self.fs

        self.x = np.linspace(0,self.duration,self.numPoints)

    def plotSTFT(self, hanning=False, numWindow=4):

        plt.title("Power Signal :\nLetter : {}  Elec = {}".format(self.targetLetters[self.numSession], self.numElec))
        super().plotStft(self)

class MultipleElectrodeProcessor(SingleElectrodeProcessor):

    def __init__(self, fileName="BCI/Subject_A_Train.mat"):
        super().__init__(fileName=fileName)
        #self.reorganizeSignal()
        self.cutAllSignals()


    def cutAllSignals(self):
        """
        Format used self.allSignalFormated :
        [   NUM SESSION,   NUM ELEC,    NUMTRIAL,      POINT        ]
         85train 100test |  [0,63]   |  [0,14]  | [0,504] or [0,522]
        """
        self.allSignalFormated = []

        for numSession, session in enumerate(self.allSignalUnformated):
            sessionSignals = []

            for numElec in range(64):
                self.mainSignal = session[:,numElec]

                #Cut the signal
                sessionSignals.append(self.signalCutting())

            self.allSignalFormated.append(sessionSignals)

        self.allSignalFormated = np.array(self.allSignalFormated)

        assert len(self.allSignalFormated) in (85,100)
        assert len(self.allSignalFormated[0]) == 64
        assert len(self.allSignalFormated[0][0]) == 15
        assert len(self.allSignalFormated[0][0][-1]) == 504
        assert len(self.allSignalFormated[0][0][0]) == 504
        assert len(self.allSignalFormated[0][0][1]) == 504



    def signalCutting(self):
        """
        Function to segment the signal into trial (instead of whole session)
        Input : One session of 7794 points (32,45s)
        Ouput : List of Trial (504 points)

        One session look like this :

        12*175 ms : One sample (set of 12 intensification) 2100 ms : 504 Points

        15*504 points : 15 Samples

        7560 points at 240Hz
        """

        oldSignal = self.mainSignal
        newSignal = []

        for trial in range(15):
            newSignal.append( oldSignal[trial*504:(trial+1)*504] )

        assert len(newSignal)==15
        return newSignal

    def get(self, numSession=0, numTrial=0, numElec=0):
        return self.allSignalFormated[numSession, numTrial, numElec]


    def meanSignal(self, numSession, numElec):
        pass
        #TODO



def main():
    pass

if __name__ == '__main__':
    main()
