#!/usr/bin/env python
# -*- coding: utf-8 -*-

from scipy.fftpack import fft, ifft, rfft, irfft, fftshift
from scipy import  sin, pi
from math import ceil
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt


class SignalManipulator(object):

    def __init__(self, fileName="BCI/Subject_A_Train.mat"):
        dataX = sio.loadmat(fileName)
        self.signal = dataX['Signal']
        self.targetLetters = dataX['TargetChar'][0]
        self.freq = 240 #Hz : Fréquence d'échantillonage du signal
        self.x = x = np.linspace(0,7794/self.freq,7794)

    def plotPower(self, numTrial, numElec, frameCutting, hanning=False):
        powerSig = self.stft(numTrial, numElec, frameCutting, hanning=False)

        print(len(powerSig))

        plt.plot(self.x,powerSig)
        plt.ylabel('FreqPower')
        plt.xlabel('Time in S')
        plt.title("Power Signal :\nLetter : {}  Elec = {}".format(self.targetLetters[numTrial], numElec))
        plt.grid()
        plt.show()

    def plotElec(self, numTrial, numElec):
        if not(numTrial in range(0,85) and numElec in range(64)):
            raise ValueError("Trial must be in [0,84] and number of Electrode in [0,63]")

        y = self.signal[numTrial, :, numElec]

        plt.plot(self.x,y)
        plt.ylabel('mV')
        plt.xlabel('Time in S')
        plt.title("Signal:\nLetter : {}        Elec = {} ".format(self.targetLetters[numTrial], numElec))
        plt.grid()
        # plt.show()

    # def _stft(x, fs, framesz, hop):
    #     framesamp = int(framesz*fs)
    #     hopsamp = int(hop*fs)
    #     w = scipy.hanning(framesamp)
    #     X = scipy.array([scipy.fft(w*x[i:i+framesamp])
    #                     for i in range(0, len(x)-framesamp, hopsamp)])

    # def stft(self, numTrial, numElec, frameCutting, hanning):
    #     splitedSignal = splitSignal(self.signal[numTrial,:,numElec], frameCutting)
    #     #Split the signal in N almost equal pieces
    #     mergedSTFT = []
    #
    #     for frame in splitedSignal:
    #         if hanning:
    #             w = np.hanning(len(frame))
    #         else:
    #             w = [1 for i in range(len(frame))]
    #
    #         mergedSTFT.extend(fftshift(fft(w*frame)))
    #
    #     return np.array(mergedSTFT)


def stftForAnySignal(signal, hanning=False):

    splitedSignal = splitSignal(signal,10)

    print(signal[0:10])

    mergedSTFT = []

    for frame in splitedSignal:

        input(frame[0:10])

        if hanning:
            w = np.hanning(len(frame))
        else:
            w = [1 for i in range(len(frame))]

        mergedSTFT.extend(fftshift(fft(w*frame)))

    return np.array(mergedSTFT)




def splitSignal(X,n):

    newN = ceil(len(X) / n)
    for i in range(0, n-1):
        yield X[i*newN:i*newN+newN]
    yield X[n*newN-newN:]

def showWindow():

    f0 = 440         # Compute the STFT of a 440 Hz sinusoid
    fs = 8000        # sampled at 8 kHz
    T = 5            # lasting 5 seconds
    framesz = 0.050  # with a frame size of 50 milliseconds
    hop = 0.025

    t = np.linspace(0, T, T*fs, endpoint=False)

    y = sin(2*pi*f0*t)

    Y = stftForAnySignal(y)

    x = [i for i in range(len(y))]



    plt.plot(x,Y)
    plt.grid()
    plt.show()







def main():


    myVisu = SignalManipulator()

    #myVisu.plotElec(0,0)
    #myVisu.plotPower(0,0,20, True)

    showWindow()

if __name__ == '__main__':
    main()
