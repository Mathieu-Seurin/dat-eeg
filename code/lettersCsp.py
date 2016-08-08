from learnData import *

def concatenateAllSubjects():
    freqMin = 0.1
    freqMax = 30
    decimation = 4
    X,y,_,_ = prepareFiltered('1',freqMin,freqMax,decimation)

    for i in range(2,6):
        X2, y2, _, _ = prepareFiltered(str(i),freqMin,freqMax,decimation)
        X = np.concatenate((X,X2))
        y = np.concatenate((y,y2))

    print X.shape
    print y.shape
    return X,y



freqMin = 0.1
freqMax = 30
decimation = 4
X,y,_,_ = prepareFiltered('5',freqMin,freqMax,decimation)

letters = ('A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z')


def lettersCompCsp(X,y):
    results = np.zeros((26,26))
    totalCompo = 0

    for l1 in range(1,25):
        for l2 in range(l1+1,26):

            index = ((y==l1) | (y==l2))
            xTemp, yTemp = X[index,:], y[index]

            result, cardCompo = learnCspPipeline(xTemp,yTemp,[],[],'f1','letters',jobs=2,classifier='ridge')
            print letters[l1-1],letters[l2-1], result

            results[l1,l2] = result
            totalCompo += cardCompo

    np.save('lettersComparisonCsp', results)
    print cardCompo/(26*25)

def lettersCompWithoutCsp(X,y):

    results = np.zeros((26,26))
    
    for l1 in range(14,25):
        for l2 in range(l1+1,26):

            index = ((y==l1+1) | (y==l2+1))
            xTemp, yTemp = X[index,:], y[index]

            result = learnCspPipeline(xTemp,yTemp,[],[],'f1','letters',jobs=2,classifier='lin')
            print letters[l1],letters[l2], result, set(yTemp)
            results[l1,l2] = result

    np.save('lettersComparisonWithoutCsp', results)

X = np.load('lettersComparisonWithoutCsp.npy')
print np.max(X)
