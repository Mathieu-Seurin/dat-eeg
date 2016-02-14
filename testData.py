#!/usr/bin/env python
# -*- coding: utf-8 -*

from matrixManipulation import *


data = sio.loadmat("BCI/Subject_A_Test_reshaped.mat")

print(len(data['X'][0][0]))
