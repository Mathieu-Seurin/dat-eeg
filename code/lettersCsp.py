from learnData import *

def lettersComp(X,y):

    results = np.zeros((26,26))
    totalCompo = 0
    
    for l1 in range(25):
        for l2 in range(l1+1,26):

            index = ((y==l1+1) | (y==l2+1))
            xTemp, yTemp = X[index,:], y[index]

            result, cardCompo = learnCspPipeline(xTemp,yTemp,[],[],'f1','letters',jobs=8,classifier='ridge')
            print letters[l1],letters[l2], result
            results[l1,l2] = result
            totalCompo += cardCompo

    np.save('lettersComparisonWithoutCsp', results)
    print cardCompo/(26*25)


def concatenateAllSubjects():
    freqMin = 0.05
    freqMax = 30
    decimation = 8
    X,y,_,_ = prepareFiltered('1',freqMin,freqMax,decimation)

    for i in range(2,6):
        X2, y2, _, _ = prepareFiltered(str(i),freqMin,freqMax,decimation)

        print X2.shape
        X = np.concatenate((X,X2))
        y = np.concatenate((y,y2))

    return X,y

def loadAll():
    data = sio.loadmat('Data/Subject_6_Train_reshaped.mat')
    return data['X'],data['y'].reshape((np.size(data['y'],1)))

freqMin = 0.1
freqMax = 30
decimation = 4

letters = ('A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z')


#X,y = loadAll()
#lettersComp(X,y)

withCsp = np.load('lettersComparisonWithCsp.npy')
withoutCsp = np.load('lettersComparisonWithoutCsp.npy')

plt.subplot(1,2,1)
plt.imshow(withoutCsp)

plt.subplot(1,2,2)
plt.imshow(withCsp)


plt.show()
