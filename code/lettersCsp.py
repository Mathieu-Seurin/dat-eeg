from learnData import *

def learnCspPipeline(X, y, xTest, yTest, scoring, transformedData,jobs=1, classifier='lin'):

    testAvailable = np.size(xTest)
    
    X = vecToMat(X)

    if testAvailable:
        xTest = vecToMat(xTest)

    if classifier=='lin':
        classif = svm.LinearSVC(penalty='l2',class_weight=CLASS_WEIGHT)
        params = np.logspace(-5,1,3)
        hyper = 'classif__C'

    else:
        classif = RidgeClassifier(class_weight=CLASS_WEIGHT)
        params = np.logspace(-1,3,10)
        hyper = 'classif__alpha'

    csp = CSP(reg='ledoit_wolf',log=False)
    scaler = StandardScaler()
    pipe = Pipeline(steps = [('csp',csp), ('scaler',scaler), ('classif',classif)])
#    pipe = Pipeline(steps = [('csp',csp), ('classif',classif)])

    n_components = [1,2,5,10,20,30,40,50]
    dico = {'csp__n_components':n_components, hyper:params}

    grd = grid_search.GridSearchCV(pipe,dico, cv=3, verbose=0, refit=False)
    grd.fit(X,y)

    return grd.best_score_, grd.best_params_['csp__n_components']
    
def lettersComp(X,y):

    results = np.zeros((26,26))
    totalCompo = 0
    
    for l1 in range(25):
        for l2 in range(l1+1,26):

            index = ((y==l1+1) | (y==l2+1))
            xTemp, yTemp = X[index,:], y[index]

            result, cardCompo = learnCspPipeline(xTemp,yTemp,[],[],'f1','letters',jobs=4,classifier='ridge')
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


X,y = loadAll()
lettersComp(X,y)

# withCsp = np.load('lettersComparisonWithCsp.npy')
# withoutCsp = np.load('lettersComparisonWithoutCsp.npy')

# f, ax = plt.subplots(2,sharex=True,sharey=True)
# im = ax[0].imshow(withoutCsp,interpolation='none')

# ax[1].imshow(withCsp,interpolation='none')

# plt.xticks(range(26), letters)
# plt.yticks(range(26), letters)

# f.subplots_adjust(right=0.8)
# cbar_ax = f.add_axes([0.85, 0.15, 0.05, 0.7])
# f.colorbar(im,cax=cbar_ax)


# plt.show()
