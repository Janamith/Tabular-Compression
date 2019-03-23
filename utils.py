import numpy as np
import matplotlib.pyplot as plt
import csv
import mst as mst

def loadCsvData(fileName):
    matrix = []
    first = 1
    with open(fileName) as f:
        reader = csv.reader(f)
        for row in reader: 
            if first:
                first = 0
                continue
            doubleRow = []
            for value in row:
                            #if value == 'TRUE':
                            #    doubleRow.append(1.0)
                           # else: 
                            #    doubleRow.append(0.0)
                doubleRow.append(value)
            matrix.append(doubleRow)
        return np.asarray(matrix)


#given two vectors, compute the mutual information between two features.
def computeMutualInformation(col1,col2):
    return computeEntropy(col1)  + computeEntropy(col2) -computeJointEntropy(col1,col2);#- computeCondEntropy(col1,col2)

#given a column index, returns a list of lists mapping each unique item in the given column with
#the data rows that match that item. I.e, if column k in data matrix A is the outcome of a dice roll,
#generateCondDist(k,A) returns a map mapping '1' to all rows in A where A[k] = 1, '2' to all rows in A
#where A[k] = 2, etc.
#Assumptions: inputs are ndarrays.
def generateCondDist(colInd,matrix):
    mapdist = [];
    for si in list(set(matrix[:,colInd])):
        mapdist += [[si,matrix[matrix[:,colInd] == si]]]
    return np.asarray(mapdist);

#given a predicate, returns all rows of the matrix for which the predicate is true.
def getConditionalDist(predicate, matrix):
    return matrix[predicate]

#given a vector, return a vector "histogram" which contains the probabilities of all unique items.
#Also return the corresponding item for each probability.
def getEventProbabilities(vector):
    #labels = list(set(vector));
    labels,cnts = np.unique(vector,axis=0,return_counts=True)
    vectorP = cnts/(len(vector))
    #for i in range(len(labels)):
    #    vectorP[i] = (np.asarray([vi ==labels[i] for vi in vector]).sum())/len(vector)
    return vectorP, labels

#given a vector of events, return the estimated information entropy.
def computeEntropy(vector):
    vectorP = getEventProbabilities(vector)[0]
    return computeEntropyP(vectorP)

#given a probability mass distribution, return the information entropy
def computeEntropyP(vectorP):
    return -np.dot(vectorP,np.log2(vectorP))

#given two vectors, return the estimated joint information entropy.
def computeJointEntropy(col1,col2):
    c2 = np.reshape(col2,(len(col2),1))
    c1 = np.reshape(col1,(len(col1),1))
    return computeEntropy(np.concatenate((c1,c2),axis=1))#list(zip(col1,col2)))

#given two vectors, return the estimated condition information entropy of the first given the second.
def computeCondEntropy(col1,col2):
    c1 = np.reshape(col1,(len(col1),1))
    c2 = np.reshape(col2,(len(col2),1))
    condDist = generateCondDist(1,np.concatenate((c1,c2),axis=1))
    sumH = 0
    N = len(col1);
    for disti in condDist[:,-1]:
        sumH += (len(disti)/N)*computeEntropy(list(map(tuple,disti))) #p(y)H(X|Y=y)
    return sumH

def getMutualInfoMatrix(matrix):
    N = np.shape(matrix)[1]
    MI = np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            if MI[j,i] == 0:
                #print((i,j))
                MI[i,j] = computeMutualInformation(matrix[:,i],matrix[:,j]) #I(I;J)
            else:
                MI[i,j] = MI[j,i]
    return MI

def computeCompRatio(bnetdict,mutualIMat):
    cR = 0
    for key in [*bnetdict]:
        cR += mutualIMat[key,key]
        for dependency in bnetdict[key]: #tuple containing (k,I(key;k))
            cR-=dependency[1]
    return cR

def computeDataSize(cR, matrix):
    return cR*np.shape(matrix)[0]

#given a vector, returns a vector with the information arithmetically encoded.
#Also return the dictionary mapping original strings to integers, and the number of unique strings.
def arithEncode(vector):
    return
    #return dictionary,vecEncoded,N

#Plot the mutual information heatmap.
def plotMutualInfo(mutualI):
    plt.imshow(mutualI, interpolation='nearest', cmap=plt.cm.hot)
    plt.colorbar()
    plt.show()

matrix =loadCsvData("connect-4.data");
print(matrix[0]);
MI =getMutualInfoMatrix(matrix)
print(MI)
treed = mst.MST(MI)
print(treed)
cR = computeCompRatio(treed,MI)
print(cR)
datasz = computeDataSize(cR,matrix)
print(datasz);
plotMutualInfo(MI);
