import numpy as np
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile


files = ['l1HyperSelection'+score.title() for score in ('f1', 'accuracy',\
                                                        'roc_auc', 'precision','recall')]
print(files)

path = 'Results/True/TrATeA/'

# lf = [f for f in os.listdir(path)]

# for file_ in lf:

#     with open(path+file_, 'r') as results:
#         x = [0 for i in range(18)]
#         y = [0 for i in range(18)]
#         for i, line in enumerate(results):
#             if line[0] == '[':
#                 break

#             print(line)
#             x[i], y[i], _ = (j for j in map(float, line.split(' ')))

#     plt.plot(x,y)
#     plt.xscale('log')
#     plt.grid()

#     if file_[-6] == '0':
#         title = file_[16:-8] + ' Stft = ' + file_[-8:-4]
#     elif file_[-6] == '2':
#         title = file_[16:-9] + ' Stft = ' + file_[-9:-4]
#     else :
#         title = file_[16:-4]
#     plt.title(title)
#     plt.show()


with open(path+'l2HyperSelectionLinearAccuracyRaw.txt', 'r') as results:
    x = [0 for i in range(16)]
    y = [0 for i in range(16)]
    for i, line in enumerate(results):
        if line[0] == '[':
            break

        print(line)
        x[i], y[i], _ = (j for j in map(float, line.split(' ')))


maxY = max(y)
maxX = np.argmax(x)

print(maxX)

plt.plot(x,y)
plt.xscale('log')
plt.grid()
plt.xlabel('C (log scale)')
plt.ylabel('Score ')


plt.show()


