import matplotlib.pyplot as plt
import numpy as np
from plotWind import *
import sys
import math
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from sklearn.cluster import SpectralClustering, KMeans

"""
    windSystem class keeps track of the wind nodes and meta data of the system

    plotSystem: calls an external file function to plot the data with temperature,
    location, and wind direction indicated in the plot
"""
class windSystem:
    def __init__(self, name, width, height, c=2):
        self.windNodes = []
        self.name = name
        self.width = width
        self.height = height
        self.c = c

    def plotSystem(self):
        plotAll(self.windNodes, self.width, self.height)
        print("Printing Graph of System")


"""
    windNode class makes a single wind data point, represents a single weather
    station's weather readings

    A wind node needs the following parameters specified:
    u = lat position ex: -1 (Straight West)
    v = long position ex: 1 (Straight North)
    x = horizontal position in system
    y = vertical position in system
    temp = discrete temp value ex: C1 (coldest relative temp)
"""
class windNode:
    def __init__(self, u, v, t, x, y):
        self.u = u
        self.v = v
        self.x = x
        self.y = y
        self.temp = t


"""
    make_wind_data reads the data text file and creates teh windNodes to populate
    the windSystem.

    windDataFile: name of .txt file to read data from, file must be in particular
    format (see data folder)
"""
def make_wind_data(windDataFile):

    #Opening and storing lines of data .txt file
    f = open(windDataFile, "r")
    lines = [line.split() for line in f.readlines()]
    f.close()

    #Extracting meta data, width and height must be equal
    width = int(lines[0][0])
    height = int(lines[0][1])
    c = int(lines[1][0]) # Number of clusters

    #Creating overall system
    wS = windSystem("USA", width, height, c=c)

    lines = lines[2:]

    xVal = 0
    yVal = 0

    #Making wind nodes
    for line in lines:
        if(line):
            tempVal = line[0]
            uVal = float(line[1])
            vVal = float(line[2])
            wS.windNodes.append(windNode(uVal, vVal, tempVal, xVal, yVal))
            xVal+=1
            if(xVal==width):
                xVal=0
                yVal+=1

    #Makes plot of whole system including arrows for wind direction, color for temp, and (x,y) location
    wS.plotSystem()

    return wS

"""
    Returns the eigenvalues of given matrix M

    M must be a square matrix
"""
def find_eigens(M):
    vals, vecs = np.linalg.eig(M)
    return vals, vecs

"""
    makeXVals finds the x,y tuples for position from a list of wind nodes
"""
def makeXVals(wS):
    m = []
    max = wS.width
    for node in wS.windNodes:
        m.append((node.x,max-node.y))

    return m

"""
    getTempConvDict coverts the discrete temp values into a tuple look up dictionary

    With the result, look up a combination of two temp values to see their diff
    ex: tempLookUp[(C1,C2)] = 0.01
"""
def getTempConvDict():
    tempConversion = {
        "C1": 0,
        "C2": 0.4,
        "C3": 0.9,
        "W3": 6,
        "W2": 7,
        "W1": 8,
        "D3": 10,
        "D2": 11,
        "D1": 12,
    }

    tempLookUp = {}
    factor = 1
    for temp1 in tempConversion.keys():
        for temp2 in tempConversion.keys():
            val = abs(tempConversion[temp1]-tempConversion[temp2])
            if(val>1):
                tempLookUp[(temp1,temp2)] =  factor*(12 -val)
            else:
                tempLookUp[(temp1,temp2)] = factor*(12 -val)
    return tempLookUp


"""
    findAbsPos coverts x,y position tuple into position in simple list of wind nodes

    xY: position tuple
    size: width of data matrix
"""
def findAbsPos(xY, size):
    x,y = xY
    val = y*size + x
    return val

"""
    make_weights_matrix finds the weights between each node based on temp
    (possible extensions would be to base the weights on position or direction)

    wSystem: a windSystem object
"""
def make_weights_matrix(wSystem, type=0):

    #Creating empty matrices
    connMatrix = np.zeros((len(wSystem.windNodes), len(wSystem.windNodes)))
    weightMatrix = np.zeros((len(wSystem.windNodes), len(wSystem.windNodes)))


    #fill connection matrix, this just give 1 if the ndoes are connected, no otherwise
    min = 0
    max = wSystem.width*wSystem.height
    for node in wSystem.windNodes:
        curr = findAbsPos((node.x, node.y), wSystem.width)
        up = findAbsPos((node.x, node.y-1), wSystem.width)
        down = findAbsPos((node.x, node.y+1), wSystem.width)
        right = findAbsPos((node.x+1, node.y), wSystem.width)
        left = findAbsPos((node.x-1, node.y), wSystem.width)

        #Connects nodes that are above, below, right or left of the current node
        if(up >=min and up<max):
            connMatrix[(curr, up)] = 1

        if(down >=min and down<max):
            connMatrix[(curr, down)] = 1

        if(right >=min and right<max):
            connMatrix[(curr, right)] = 1

        if(left >=min and left<max):
            connMatrix[(curr, left)] = 1

    heatDiff = getTempConvDict()

    if(type == 0):
        for node1 in wSystem.windNodes:
            pos1 = findAbsPos((node1.x, node1.y), wSystem.width)
            for node2 in wSystem.windNodes:
                pos2 = findAbsPos((node2.x, node2.y), wSystem.width)
                if(connMatrix[(pos1,pos2)]):
                    val = heatDiff[(node1.temp,node2.temp)] # They are connected
                    weightMatrix[(pos1, pos2)] = val #Add to the weight matrix

        print("Making Weights with Temp Weighted Graph")
    if(type == 1):
        #Not yet testing functional
        for node1 in wSystem.windNodes:
            for node2 in wSystem.windNodes:
                uDiff = (node1.u-node2.u)**2
                vDiff = (node1.v-node2.v)**2
                totalDiff = math.sqrt(uDiff + vDiff)
                corr = 2 - totalDiff
        print("Making Weights with Direction Graph")
    if(type == 2):
        #Not yet testing functional
        for node1 in wSystem.windNodes:
            pos1 = findAbsPos((node1.x, node1.y), wSystem.width)
            for node2 in wSystem.windNodes:
                pos2 = findAbsPos((node2.x, node2.y), wSystem.width)
                val = heatDiff[(node1.temp,node2.temp)]
                weightMatrix[(pos1, pos2)] = val

        print("Making Weights with Temp Graph")

    return weightMatrix


"""
    make_laplacian makes the Laplacian Matrix from the degree (D) and weights (W) matrices

    W: weights matrix of wind data
"""
def make_laplacian( W ):
    D = np.diag(W.sum(axis=1))
    L = D-W
    print("Making Laplacian")
    return L

"""
    graph_solution makes a plot of the results of clustering, breaks up
    eigenvector into 2 or 3 clusters

    X: position values of wind data
    eigvec: calcauted eigenvectors
    clusters: number of expected clusters
    dot_size: dot size of the plot data points
"""
def graph_solution(X, eigvec, clusters,dot_size=130):
    y_spec =eigvec[:,1].copy()
    print(y_spec)
    if(clusters == 2):
        y_spec[y_spec < 0] = 0
        y_spec[y_spec > 0] = 1


    if(clusters == 3):
        index = 0
        for x in y_spec:
            if(x >0.09):
                y_spec[index] = 0
            elif(x< -0.1):
                y_spec[index] = 3
            else:
                y_spec[index] = 5
            index+=1
    print(y_spec)

    #Seperates the tuples in X
    x = [i[0] for i in X]
    y = [i[1] for i in X]
    fig, ax = plt.subplots(figsize=(7,7))
    ax.set_title('Our Results', fontsize=18, fontweight='demi')
    ax.scatter(x, y,c=y_spec ,s=dot_size, cmap='spring')
    print("Graphing Solution")

"""
    graph_kmeans plots the k means solution to the data

    X: position values of wind data
    clusters: number of expected clusters
    dot_size: dot size of the plot data points
"""
def graph_kmeans(X, clusters, dot_size=130):
    km = KMeans(init='k-means++', n_clusters=clusters)
    km_clustering = km.fit(X)
    fig, ax = plt.subplots(figsize=(7,7))
    x = [i[0] for i in X]
    y = [i[1] for i in X]
    ax.set_title('KMeans Results', fontsize=18, fontweight='demi')
    plt.scatter(x, y, c=km_clustering.labels_,  s=dot_size, cmap='rainbow', alpha=0.7)

"""
    graph_skLearn plots teh sklearn's library version of the solution to the data

    X: position values of wind data
    clusters: number of expected clusters
    dot_size: dot size of the plot data points
"""
def graph_skLearn(X, clusters, dot_size=130):
    sc = SpectralClustering(n_clusters=clusters, affinity='nearest_neighbors', random_state=0)
    sc_clustering = sc.fit(X)
    fig, ax = plt.subplots(figsize=(7,7))
    x = [i[0] for i in X]
    y = [i[1] for i in X]
    ax.set_title('Sklearn Clustering Results', fontsize=18, fontweight='demi')
    plt.scatter(x, y, c=sc_clustering.labels_, cmap='rainbow', alpha=0.7, s=dot_size)

if __name__ == '__main__':
    windDataFile = sys.argv[1] #Read user file input
    wSystem = make_wind_data(windDataFile) #Make overall windsystem from data
    xVals= makeXVals(wSystem) #Put the data into x,y position values
    W = make_weights_matrix(wSystem, type=2) #Make weights/connection matrix
    L = make_laplacian(W) #Make Laplacian Matrix
    vals, vecs = find_eigens(L) #Find eigenstuff from Laplacian
    vecs = vecs[:,np.argsort(vals)] #Sort the eigenvectors by the eigenvalues size
    vals = vals[np.argsort(vals)] #Sort the eigenvales by the eigenvalues size
    clusters = wSystem.c
    graph_solution(xVals, vecs, clusters) #Graph our solution
    graph_kmeans(xVals, clusters) #Graph k_means solution
    graph_skLearn(xVals, clusters) #Graph sk_learn's solution
    plt.show()
    print("Completed")
