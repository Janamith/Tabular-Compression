import numpy as np
import matplotlib.pyplot as plt
import csv
import mst as mst
import copy 
import random
import networkx as nx

LOAD_MI_MAT = True
MI_MAT_FILE = "mi_connect4.npy"
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
    print(N)
    for i in range(N):
        for j in range(N):
            if MI[j,i] == 0:
                #print((i,j))
                MI[i,j] = computeMutualInformation(matrix[:,i],matrix[:,j]) #I(I;J)
            else:
                MI[i,j] = MI[j,i]
    return MI
def insertRandomVstructure(MST):
    MSTcpy = copy.deepcopy(MST);
    vertex_1 = random.choice(list(MST.keys())) 
    vertex_2 = random.choice(list(MST.keys()))
    while(vertex_1 == vertex_2):
        vertex_2 = random.choice(list(MST.keys()))
    #print("vertex added from feature %s to feature %s" %(vertex_1,vertex_2))
    MSTcpy[vertex_1].append((vertex_2, computeMutualInformation(matrix[:,vertex_1], matrix[:,vertex_2])))
    return MSTcpy, (vertex_1,vertex_2)

def insertVstructure(MST,v1,v2):
    MSTcpy = copy.deepcopy(MST);
    vertex_1 = v1
    vertex_2 = v2 
    mi = computeMutualInformation(matrix[:,vertex_1],matrix[:,vertex_2]);
    if (vertex_2,mi) in MSTcpy[vertex_1]:
        return 0,0
    #print("vertex added from feature %s to feature %s" %(vertex_1,vertex_2))
    MSTcpy[vertex_1].append((vertex_2,mi))
    return MSTcpy, (vertex_1,vertex_2)
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
    f = plt.colorbar()
    #f = plt.figure();
    plt.title("Connect 4 Dataset Mutual Information Between Columns");
    f.set_ticklabels("Column Number")
    f.set_label("Mutual Information (bits)")
    #plt.show()
    plt.savefig("mutual_I.png");

matrix =loadCsvData("connect-4.data");
if LOAD_MI_MAT == True: 
    MI = np.load(MI_MAT_FILE);
else: 
    MI =getMutualInfoMatrix(matrix)
    np.save(MI_MAT_FILE,MI);
#plotMutualInfo(MI);
HBase = 0
for n in range (np.shape(matrix)[1]):
    HBase += computeEntropy(matrix[:,n])
print("baseline compression rate/sum of entropies = %f" % HBase);

treed = mst.MST(MI)
cR = computeCompRatio(treed,MI)
print("MST comp rate: %f " % cR)
datasz = computeDataSize(cR,matrix)
mstkeys = [*treed];
mstedges = []
elabels={}
ml = {}
for k in mstkeys:
    for e in treed[k]:
        mstedges.append((k,e[0]))
        ml[(k,e[0])] = 1
        elabels[(k,e[0])]="%.3f" % e[1]
print(mstkeys)
print(mstedges)

G = nx.Graph()
G.add_nodes_from(mstkeys);
G.add_edges_from(mstedges)
nx.draw(G,nx.kamada_kawai_layout(G),with_labels=True);
nx.draw_networkx_edge_labels(G,nx.kamada_kawai_layout(G),edge_labels=elabels)
plt.savefig("mst_graph.png")
plt.show();
#plt.figure();
#G.add_edges_from([(7,8)])
#nx.draw(G,nx.kamada_kawai_layout(G),with_labels=True);
#nx.draw_networkx_edge_labels(G,nx.kamada_kawai_layout(G),edge_labels=elabels)
#plt.show();
#plt.savefig("mst_Vopt_graph.png")
print("MST dataSz: %f " % datasz);
minCR = HBase
minDataSz =HBase*np.shape(matrix)[0]
minV = ''
for i in range(np.shape(matrix)[1]):
    for j in range(np.shape(matrix)[1]):
        if i != j:
            MSTi, vi =  insertVstructure(treed,i,j)
            if MSTi ==0: continue
            cr =computeCompRatio(MSTi,MI)
            if cr < minCR: 
                minV = vi
                minCR = cr
                minDataSz = computeDataSize(minCR,matrix)
        #print("comp rate: %f " % cr)
        #print("dataSz: %f " % computeDataSize(cr,matrix))
print("best V structure: ",minV)
print("compression rate: %f" % minCR)
print("data size: %f" % minDataSz)

