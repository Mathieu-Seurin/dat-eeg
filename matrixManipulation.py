#!/usr/bin/env python
# -*- coding: utf-8 -*-

from scipy.fftpack import fft, fftshift, fftfreq
from scipy import  sin, pi, ceil, array, absolute
import scipy.io as sio

import numpy as np

import matplotlib.pyplot as plt

from copy import copy, deepcopy

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
        plt.title("Signal")
        plt.grid()
        plt.show()

    def plotFft(self):
        y = abs(fft(self.mainSignal))
        x = fftfreq(len(self.mainSignal), 1/240)
        plt.plot(x,y)
        plt.title('FFT')
        plt.show()

    def afft(self):
        """ Compute the absolute shifted Fast Fourier Transform """
        return np.abs(fftshift(fft(self.mainSignal)))

    def splitSignal(self,n):

        newN = ceil(self.numPoints/ n)
        for i in range(0, n-1):
            yield self.mainSignal[i*newN:i*newN+newN]
        yield self.mainSignal[n*newN-newN:]


    def stft(self, hanning=True, frameSize=0.5, hop=0.25):

        frameAmp = int(self.fs*frameSize)
        hopAmp = int(self.fs*hop)

        print(frameAmp, hopAmp)

        if hanning:
            w = np.hanning(frameAmp)
        else:
            w = [1 for i in range(frameAmp)]

        mergedSTFT = []

        for i in range(0, len(self.mainSignal)-frameAmp, hopAmp):
            mergedSTFT.extend(fftshift(fft(w*self.mainSignal[i:i+frameAmp])))

        return array(mergedSTFT)

    def plotStft(self, hanning=True,frameSize=0.5, hop=0.25):

        powerSig = self.stft(hanning,frameSize,hop)

        print(len(powerSig))
        x = np.linspace(0,self.duration,len(powerSig))

        plt.plot(x,absolute(powerSig))
        plt.ylabel('FreqPower')
        plt.xlabel('Time in S')
        plt.title('STFT')
        plt.grid()
        plt.show()

    def multiplePlot(self):

        plt.subplot(4,2,1) #NORMAL

        plt.plot(self.x,self.mainSignal)
        plt.ylabel('mV')
        plt.xlabel('Time in S')
        plt.title("Signal")
        plt.grid()

        plt.subplot(4,2,2) #FFT

        y = abs(fft(self.mainSignal))
        x = fftfreq(len(self.mainSignal), 1/240)
        plt.plot(x,y)
        plt.title('FFT')
        plt.grid()


        plt.subplot(4,2,3) #STFT hanning

        powerSig = self.stft()
        x = np.linspace(0,self.duration,len(powerSig))

        plt.plot(x,absolute(powerSig))
        plt.ylabel('FreqPower')
        plt.xlabel('Time in S')
        plt.title('STFT HANNING')
        plt.grid()

        plt.subplot(4,2,4) #STFT

        powerSig = self.stft(hanning=False)
        x = np.linspace(0,self.duration,len(powerSig))

        plt.plot(x,absolute(powerSig))
        plt.ylabel('FreqPower')
        plt.xlabel('Time in S')
        plt.title('STFT')
        plt.grid()

        plt.subplot(4,2,4) #STFT NO OVERLAP

        powerSig = self.stft(hanning=False,frameSize=0.5, hop=0.5)
        x = np.linspace(0,self.duration,len(powerSig))

        plt.plot(x,absolute(powerSig))
        plt.ylabel('FreqPower')
        plt.xlabel('Time in S')
        plt.title('STFT')
        plt.grid()

        plt.show()

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

    def reorganizeSignal(self):
        """
        Organize all signals in a new fashion

        self.signals['A'] : All trials where the 'letter' is the target and appears on screen (p300 must be in the set)
        self.signals['letter'] is a np.array of trials

        self.signals['A'][0] : First trial where the letter appears : 64 lists of points
        self.signals['letter'][X] is a trial made of 64 lists of point (64 electrodes)

        self.signals['A'][0][0]: First electrode where the letter appears : 64 lists of points
        self.signals['letter'][X] is a trial made of 64 lists of point (64 electrodes)


        self.signals['absent'] trial where the target letter is not enlightened
        self.signals['absent'] is np.array of trial
                               A trial is made of 64 lists of points (electrode)
        """
        self.allSignalFormated = dict()



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
