import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

"""
    getColor covers a discrete temperature into numerical value for plotting
"""
def getColor(tempVal):

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
    #Defaults to 0
    return tempConversion.get(tempVal, 0)

"""
    getImage gets the arrow direction formated for a plot from a filepath

    path: image file path
"""
def getImage(path):
    return OffsetImage(plt.imread(path))

"""
    getDirection finds the nearest direction from u,v values

    u: horizontal direction
    v: vertical direction
"""
def getDirection(u,v):
    directionConversion = {
        (1,0): "E",
        (0,1): "N",
        (-1,0): "W",
        (0,-1): "S",
        (0.7,0.7): "NE",
        (-0.7,0.7): "NW",
        (-0.7,-0.7): "SW",
        (0.7,0.7): "SE",
        (0.4,0.9):"NNE",
        (-0.4,0.9):"NNW",
        (0.9,0.4):"NEE",
        (-0.9,0.4):"NWW",
        (0.4,-0.9):"SSE",
        (-0.4,-0.9):"SSW",
        (0.9,-0.4):"SEE",
        (-0.9,-0.4):"SWW",
    }

    return directionConversion.get((u,v),"E")

"""
    plotAll plots the data with temperature, location, and wind direction indicated in the plot

    nodes: list of wind data nodes
    width: width of data matrix
    height: height of data matrix (should equal width)
"""
def plotAll(nodes, width, height):

    fullList = np.zeros((height, width))

    xList = []
    yList = []
    vals = []
    for node in nodes:
        x = node.x
        y = node.y
        xList.append(x)
        yList.append(y)
        direction = "images/" + getDirection(node.u,node.v) + ".png"
        vals.append((x,y,direction))
        temp = getColor(node.temp)
        fullList[(y,x)] = temp

    fig, ax = plt.subplots(figsize=(7,7))
    ax.set_title('Wind Data Temp and Direction', fontsize=18, fontweight='demi')
    im = ax.imshow(fullList, cmap='plasma')

    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Wind Temps", rotation=-90, va="bottom")

    ax.scatter(xList, yList, s=0.1)

    for x0, y0, path in vals:
        ab = AnnotationBbox(getImage(path), (x0, y0), frameon=False)
        ax.add_artist(ab)
