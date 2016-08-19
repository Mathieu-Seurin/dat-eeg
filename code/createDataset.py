#!/usr/bin/env python
#coding: utf-8
from __future__ import division 

import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt

f0 = 5
fs = 240
numPoints = 160
numElec = 64
numTrial = 2500

y = np.random.choice([-1,1],numTrial)

indexPos = np.where(y==1)[0]
# indexPos = np.ones(numTrial, dtype=bool)

#x = np.linspace(0, numPoints/fs, numPoints)
x = np.linspace(0,numPoints/fs,numPoints)

sinus = np.ones((len(indexPos),20))*5*np.sin(2*np.pi*f0*x[:20]) #5Hz

sig = np.random.normal(-1,20,(numTrial,numPoints*numElec))
sig += np.random.normal(0,10,(numTrial,numPoints*numElec))


for elec in [11,4,10,12,50,51,52,53,18,17,19,58]:
    sig[indexPos,elec*160+80:elec*160+100] += sinus

data = {'X':sig, 'y':y}

#sio.savemat("Data/Subject_C_Train_reshaped.mat", data)
print "Done and Saved"

plt.plot(sig[:,1600:1760].mean(axis=0))
plt.plot(sig[0,1600:1760])

plt.show()
