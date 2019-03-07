import numpy as np
import matplotlib.pyplot as plt
import csv
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


#given two column indices and the data, compute the mutual information between two features.
def computeMutualInformation(col1,col2,matrix):
    return
    #return I;

#given a column index, returns a list of lists mapping each unique item in the given column with
#the data rows that match that item. I.e, if column k in data matrix A is the outcome of a dice roll,
#generateCondDist(k,A) returns a map mapping '1' to all rows in A where A[k] = 1, '2' to all rows in A
#where A[k] = 2, etc.
#Assumptions: inputs are ndarrays.
def generateCondDist(column,matrix):
    mapdist = [];
    for si in set(matrix[:,column]):
        mapdist += [[si,matrix[matrix[:,column] == si]]]
    return mapdist;

#given a predicate, returns all rows of the matrix for which the predicate is true.
def getConditionalDist(predicate, matrix):
    return matrix[predicate]

#given a vector, return a vector "histogram" which contains the probabilities of all unique items.
#Also return the corresponding item for each probability.
def getEventProbabilities(vector):
    return
    #return vectorP, labels

#given a vector of probabilities of each unique event in an event space, return the information entropy.
def computeEntropy(vectorP):
    return -(vectorP*np.log2(vectorP)).sum()/len(vectorP)

def computeJointEntropy(col1,col2,matrix):
    return
def computeCondEntropy(col1,col2,matrix):
    return
#given a vector, returns a vector with the information arithmetically encoded.
#Also return the dictionary mapping original strings to integers, and the number of unique strings.
def arithEncode(vector):
    return
    #return dictionary,vecEncoded,N

#Plot the mutual information heatmap.
def plotMutualInfo(mutualI):
    plt.imshow(mutualI, interpolation='nearest', cmap=plt.cm.hot)
    plt.colorbar()

matrix =loadCsvData("connect-4.data");
print(matrix[0]);
