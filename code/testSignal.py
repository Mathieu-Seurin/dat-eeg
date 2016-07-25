#!/usr/bin/env python
# -*- coding: utf-8 -*-

from signalManipulation import *
from manipulateData import *
from scipy import signal

import wavelets
from wavelets import WaveletAnalysis

import sys

f0 = 5
fs = 60
T = 2

if sys.argv[1] == 'sin':

    x = np.linspace(0, T, fs*T, fs)

    signal = sin(2*pi*f0*x) #5Hz
    signal += sin(2*pi*2*f0*x) #10Hz

    signal = np.concatenate((signal, sin(2*pi*1.75*f0*x))) #Then 8.75Hz


    mySig = SignalHandler(signal, fs)
    mySig.multipleStftPlot(frameSize=0.3)

elif sys.argv[1] == 'mean':

    decimation = 4
    X, y, _, _ = prepareFiltered('A',0.5,30,decimation)

    lenSig = np.size(X,1)/64
    fs = 240//decimation

    frameSize = 0.2

    p300 = np.where(y==1)[0][:50]
    nonp300 = np.where(y==-1)[0][:5]

    for numElec in [11,47]:

        elec = slice(lenSig*numElec,lenSig*numElec+lenSig)
        print(elec)

        signal = X[p300[0],elec]
        mySig = SignalHandler(signal, fs)
        mySig.multipleStftPlot(frameSize=frameSize)
        plt.title("Single{}".format(numElec))
        plt.show()

        
        xP300 = X[p300,elec]
        print("P300",xP300.shape)
        mySig = SignalHandler(xP300.mean(axis=0), fs)
        mySig.multipleStftPlot(frameSize=frameSize)
        plt.title("Mean P{}".format(numElec))
        plt.show()


        xNonP300 = X[nonp300,elec]
        print("NonP300",xNonP300.shape)
        mySig = SignalHandler(-xNonP300.mean(axis=0), fs)
        mySig.multipleStftPlot(frameSize=frameSize)
        plt.title("Mean non P{}".format(numElec))
        plt.show()

elif sys.argv[1] == 'coherence':

    freqMin = 5
    freqMax = 50
    decimation = 1

    X, y, _, _ = prepareFiltered('A',freqMin,freqMax,decimation)

    lenSig = np.size(X,1)/64
    fs = 240//decimation

    cardSig = 5
    numElec = 11
    elec = slice(lenSig*numElec,lenSig*numElec+lenSig)

    frameSize = 0.2

    p300 = np.where(y==1)[0]
    nonp300 = np.where(y==-1)[0][:cardSig]

    sigMean = X[p300[0],elec]
    
    for numSig in range(cardSig):
        f,Cxy = signal.csd(sigMean,X[nonp300[numSig],elec],fs=fs,nperseg=lenSig)

        plt.plot(f, Cxy)
        plt.xlabel('frequency [Hz]')
        plt.ylabel('Coherence')
        plt.show()
    
    sigMean = X[nonp300[1],elec]
    
    for numSig in range(cardSig):
        f,Cxy = signal.csd(sigMean,X[nonp300[numSig],elec],fs=fs,nperseg=lenSig)

        plt.plot(f, Cxy)
        plt.xlabel('frequency [Hz]')
        plt.ylabel('Coherence')
        plt.show()
    

elif sys.argv[1] == 'filter':

    fs = 240
    f0 = 40
    x = np.linspace(0, T, fs*T, fs)
        
    signal = sin(2*pi*f0*x) #40Hz
    signal += sin(2*pi*2*f0*x) ##80Hz
    signal += sin(2*pi*0.5*f0*x) ##20Hz
    signal += sin(2*pi*0.025*f0*x) #10Hz
    signal = np.concatenate((signal, sin(2*pi*1.75*f0*x))) #Then 70Hz

    mySig = SignalHandler(signal, fs)
    mySig.plotFiltered()
    plt.show()


    X, _, _, _ = prepareRaw('A')

    mySig = SignalHandler(X[0, 0:160], 240)
    mySig.plotFiltered()


    # mySig.plotFiltered()

elif sys.argv[1] == 'wavelets':

    x = np.linspace(0, T, fs*T, fs)

    signal = sin(2*pi*f0*x) #40Hz
    signal += sin(2*pi*2*f0*x) #80Hz
    signal = np.concatenate((signal, sin(2*pi*1.75*f0*x))) #Then 70Hz

    freqMin = 0.1
    freqMax = 15
    decimation = 8

    X, y, _, _ = prepareFiltered('A',freqMin,freqMax,decimation)

    lenSig = np.size(X,1)/64
    fs = 240//decimation

    cardSig = 10
    numElec = 10
    elec = slice(lenSig*numElec,lenSig*numElec+lenSig)

    frameSize = 0.2

    p300 = np.where(y==1)[0]
    nonp300 = np.where(y==-1)[0][:cardSig]

    for numSig in range(cardSig):
        signal = X[p300[numSig],elec]

        dt = 1.0/fs
        wa = WaveletAnalysis(signal, dt=dt)

        # wavelet power spectrum
        power = wa.wavelet_power
        # scales
        scales = wa.fourier_frequencies
        # associated time vector
        t = wa.time
        print wa.fourier_frequencies
        # reconstruction of the original data
        rx = wa.reconstruction()

        fig, ax = plt.subplots()
        T, S = np.meshgrid(t, scales)
        ax.contourf(T, S, power, 100)
        print(S)
    

elif sys.argv[1] == 'meanS':

    decimation = 4
    X, y, _, _ = prepareFiltered('A',0.5,30,decimation)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)


    lenSig = np.size(X,1)/64
    fs = 240//decimation

    frameSize = 0.2

    p300 = np.where(y==1)[0][:50]
    nonp300 = np.where(y==-1)[0][:5]

    for numElec in [11,47]:

        elec = slice(lenSig*numElec,lenSig*numElec+lenSig)
        print(elec)

        signal = X[p300[0],elec]
        mySig = SignalHandler(signal, fs)
        mySig.multipleStftPlot(frameSize=frameSize)
        plt.title("Single{}".format(numElec))
        plt.show()

        
        xP300 = X[p300,elec]
        print("P300",xP300.shape)
        mySig = SignalHandler(xP300.mean(axis=0), fs)
        mySig.multipleStftPlot(frameSize=frameSize)
        plt.title("Mean P{}".format(numElec))
        plt.show()


        xNonP300 = X[nonp300,elec]
        print("NonP300",xNonP300.shape)
        mySig = SignalHandler(-xNonP300.mean(axis=0), fs)
        mySig.multipleStftPlot(frameSize=frameSize)
        plt.title("Mean non P{}".format(numElec))
        plt.show()



elif sys.argv[1] == 'patch':
    pass


plt.show()

