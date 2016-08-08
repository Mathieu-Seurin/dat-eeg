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
    nonp300 = np.where(y==-1)[0][:50]

    for numElec in [11,31,47]:

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

elif sys.argv[1] == 'filterDiff':
        
    decimation = 4
    fmax = 30
    fmin = 0.5

    # A, yA, _, _ = prepareRaw('A') #Sujet A
    # C, yC, _, _ = prepareRaw('C') #Jouet 
    # Z, yZ, _, _ = prepareRaw('Z') #Free P300
    # five, yFive, _, _ = prepareRaw('5') #Marseille Sujet 5
    
    subjects = ('A','C','Z','5')
    numSubjecs = 4
    #scaler = StandardScaler()
    #X = scaler.fit_transform(X)

    for i in range(numSubjecs):
        X, y, _, _ = prepareRaw(subjects[i])

        if i==2:
            print "here"
            print X.shape
            print y
            cardElec = 32
            fs = 2048
        else:
            cardElec = 64
            fs = 240

        lenSig = np.size(X,1)/cardElec
        frameSize = 0.2

        if i==3:
            p300 = np.where(y==15)[0][:50]
            nonp300 = np.where(y==23)[0][:50]

        else:
            p300 = np.where(y==1)[0][:15]
            nonp300 = np.where(y==-1)[0][:50]


        for numElec in [11]:

            elec = slice(lenSig*numElec,lenSig*numElec+lenSig)

            oneP300 = X[p300[0],elec]
            meanP300 = X[p300,elec].mean(axis=0)
            meanNonP300 = X[nonp300,elec].mean(axis=0)

            print meanNonP300.shape
            print meanP300.shape
            print X[p300,elec].shape



            #Normal ==================================
            plt.subplot(3,3,1)
            signal = SignalHandler(oneP300,fs)
            signal.plot()

            plt.subplot(3,3,2)
            signal = SignalHandler(meanP300,fs)
            signal.plot()

            plt.subplot(3,3,3)
            signal = SignalHandler(meanNonP300,fs)
            signal.plot()

            #filtfilt ====================================
            plt.subplot(3,3,4)
            signal= SignalHandler(oneP300, fs)
            signal.filterSig(4,fmin,fmax,decimation)
            signal.plot()

            plt.subplot(3,3,5)
            signal = SignalHandler(meanP300,fs)
            signal.filterSig(4,fmin,fmax,decimation)
            signal.plot()

            plt.subplot(3,3,6)
            signal = SignalHandler(meanNonP300,fs)
            signal.filterSig(4,fmin,fmax,decimation)
            signal.plot()

            #lfilt ===============================
            plt.subplot(3,3,7)
            signal= SignalHandler(oneP300, fs)
            signal.filterSig(4,fmin,fmax,decimation,'lfilter')
            signal.plot()

            plt.subplot(3,3,8)
            signal = SignalHandler(meanP300,fs)
            signal.filterSig(4,fmin,fmax,decimation,'lfilter')
            signal.plot()

            plt.subplot(3,3,9)
            signal = SignalHandler(meanNonP300,fs)
            signal.filterSig(4,fmin,fmax,decimation,'lfilter')
            signal.plot()

            plt.title(subjects[i])
            plt.show()

