import numpy as np
import matplotlib.pyplot as plt
import csv
import collections 
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
    return computeEntropy(col1) - computeCondEntropy(col1,col2)

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
    labels = list(set(vector));
    vectorP = np.zeros(len(labels));
    for i in range(len(labels)):
        vectorP[i] = (np.asarray([vi ==labels[i] for vi in vector]).sum())/len(vector)
    return vectorP, labels

#given a vector of events, return the estimated information entropy.
def computeEntropy(vector):
    vectorP = getEventProbabilities(vector)[0]
    return computeEntropyP(vectorP)

#given a probability mass distribution, return the information entropy
def computeEntropyP(vectorP):
    return -(vectorP*np.log2(vectorP)).sum()

#given two vectors, return the estimated joint information entropy.
def computeJointEntropy(col1,col2):
    return computeEntropy(list(zip(col1,col2)))

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

#given a vector, returns a vector with the information arithmetically encoded.
#Also return the dictionary mapping original strings to integers, and the number of unique strings.
def arithEncode(vector):
    return 0
    #return dictionary,vecEncoded,N

def MST(MImatrix):
    def getEdges(MImatrix):
        edges = []
        for i in range(MImatrix.shape[0]):
            for j in range(MImatrix.shape[1]):
                edges.append((i, j, MImatrix[i][j]))
        return edges

    sorted_edges = sorted(getEdges(MImatrix), key=lambda tup: tup[2], reverse = True)
    print (sorted_edges)
    MST = collections.defaultdict(list)
    num_edges = 0
    for edge in sorted_edges:
        if num_edges == MImatrix.shape[0]-1:
            break
        if (edge[1] in MST.keys() or edge[0] == edge[1]):
            continue
        MST[edge[0]].append((edge[1], edge[2]))
        num_edges += 1

    return MST


#Plot the mutual information heatmap.
def plotMutualInfo(mutualI):
    plt.imshow(mutualI, interpolation='nearest', cmap=plt.cm.hot)
    plt.colorbar()

MImatrix = [[1, 3, 5], [10, 2, 1], [40, 3, 9]]
MST = MST(np.asarray(MImatrix))
print(MST)
#matrix =loadCsvData("connect-4.data");
#print(matrix[0]);
