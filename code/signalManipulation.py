#!/usr/bin/env python
# -*- coding: utf-8 -*-

#Perso
from constants import *

from scipy.fftpack import fft, fftshift, fftfreq
from scipy.signal import butter, lfilter, decimate

from scipy import  sin, pi, ceil, array, absolute
import scipy.io as sio

import numpy as np
import matplotlib.pyplot as plt

#from wavelets import WaveletAnalysis

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
    def __init__(self, signal, fs):

        self.mainSignal = signal
            
        self.fs = fs #frequency Sampling
        self.numPoints = len(self.mainSignal)
        self.duration = float(self.numPoints)/fs
        self.a = None
        self.b = None

    def plot(self):

        x = np.linspace(0, self.duration, len(self.mainSignal) , endpoint=False)
        
        plt.plot(x,self.mainSignal)
        plt.ylabel('mV')
        plt.xlabel('Time in S')
        plt.grid()

    def plotFft(self):
        y = abs(fft(self.mainSignal))
        x = fftfreq(len(self.mainSignal), 1.0/self.fs)
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


    def stft(self, frameSize, hanning=True):

        hop = frameSize/2

        frameAmp =  int(np.round(self.fs*frameSize))
        hopAmp = int(np.round(self.fs*hop))

        if hanning:
            w = np.hanning(frameAmp)
        else:
            w = [1 for i in range(frameAmp)]

        mergedSTFT = []

        for i in range(0, len(self.mainSignal)-frameAmp, hopAmp):
            fftshifted = self.afft(w*self.mainSignal[i:i+frameAmp])
            mergedSTFT.append(fftshifted[:int(np.ceil(float(len(fftshifted))/2))]/frameAmp)

            # print(len(mergedSTFT), len(mergedSTFT[0]))
            # print(mergedSTFT[0])
        return array(mergedSTFT).T

    def plotStft(self, frameSize, hanning=True):

        powerSig = self.stft(frameSize, hanning=hanning)

        _, numWindows = powerSig.shape
        T = float(numWindows*frameSize)/2
        freq = fftfreq(len(powerSig[:,0]), 1.0/self.fs)

        plt.imshow(powerSig, origin='lower', extent=[0,T,0,np.max(freq)],aspect='auto')
        plt.ylabel('Frequency')
        plt.xlabel('Time in S')

    def plotFiltered(self):

        copySig = self.mainSignal

        plt.subplot(3,2,1) #Normal
        self.plot()

        plt.subplot(3,2,2) #FFT
        self.plotStft(0.2)

        plt.subplot(3,2,3)
        self.filterSig(4, 0, 30, 4)
        self.plot()

        plt.subplot(3,2,4)
        self.plotStft(0.2)

        self.mainSignal = copySig
        self.fs = 240
        
        plt.subplot(3,2,5)
        self.filterSig(4, 0, 30, 8)
        self.plot()

        plt.subplot(3,2,6)
        self.plotStft(0.25)

        
    def multipleStftPlot(self, frameSize=0.2):

        plt.subplot(2,3,1) #NORMAL
        self.plot()

        plt.subplot(2,3,2) #FFT
        self.plotFft()

        plt.subplot(2,3,4) #Default STFT
        self.plotStft(frameSize)

        plt.subplot(2,3,5) #STFT Bigger Window
        self.plotStft(frameSize*2)

        plt.subplot(2,3,6) #STFT Much Bigger Window
        self.plotStft(frameSize*3)

    def _createFilter(self, lowcut, highcut, order=5):
        nyq = 0.5 * self.fs
        low = float(lowcut) / nyq
        high = float(highcut) / nyq
        self.b, self.a = butter(order, [low, high], btype='band')

    def filterSig(self, order, lowcut, highcut, decimationFactor):
        if self.a==None:
            self._createFilter(lowcut, highcut, order=order)
        
        #Filter and decimate signal by a factor given
        self.mainSignal = decimate(lfilter(self.b, self.a, self.mainSignal), decimationFactor)
        self.fs = self.fs//decimationFactor
        self.numPoints = len(self.mainSignal)
        return self.mainSignal

