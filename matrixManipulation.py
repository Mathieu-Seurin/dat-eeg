#!/usr/bin/env python
# -*- coding: utf-8 -*-

from scipy.fftpack import fft, ifft
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt

class Visualiser(object):

    def __init__(self, fileName="BCI/Subject_A_Train.mat"):
        dataX = sio.loadmat(fileName)
        self.signal = dataX['Signal']
        self.targetLetters = dataX['TargetChar'][0]
        self.freq = 240 #Hz
        self.x = x = np.linspace(0,7794/self.freq,7794)
    def plotElec(self, numTrial=0, numElec=0):
        if not(numTrial in range(0,85) and numElec in range(64)):
            raise ValueError("Trial must be in [0,85] and number of Electrode in [0,64]")

        y = self.signal[numTrial, :, numElec]

        plt.plot(self.x,y)
        plt.ylabel('mV')
        plt.xlabel('Time in S')
        plt.title("Letter : {}        Elec = {} ".format(self.targetLetters[numTrial], numElec))
        plt.grid()
        plt.show()


def main():


    # dataX['StimulusCode'] = np.array(dataX['StimulusCode'])
    #
    # print( len(dataX['StimulusCode'][0]))
    # print( len(dataX['TargetChar'][0))

    myVisu = Visualiser()
    myVisu.plotElec(84,63)



if __name__ == '__main__':
    main()
