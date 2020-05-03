# Spectral_Alg
This repo is part of a final project for the student lead Advanced Algorithms at Olin College of Engineering. The code in this repo is a toy example application of spectral algorithms.

## Spectral Algorithms Background
Spectral Algorithms are used in many data exploration applications as a powerful way to find clusters in data. These algorithms use eigenvalues and eigenvectors of the Laplacian Matrix to find the clusters in a set of data. The Laplacian Matrix is made from the degree matrix (a simple matrix that catalogs how many nodes are connected) and the connection matrix (a matrix that shows which nodes are connected to which and with what weights). Ultimately, the Laplacian Matrix is telling us how the nodes are connected and the eigenvalues of the matrix tells us the connectivity of the groups in the data. The eigenvectors tell us the details of which nodes belong in each group.

## Application Background
The United States has very interesting wind patterns. In a very simplistic model, varied temperature wind fronts from the southwest, northwest, and southeast merge in the middle of the US creating frequently creating cyclonic conditions in an area known as “Tornado Valley.” This system can be modeled as a spectral clustering problem. Data from weather stations across the states can report the wind, direction, and temperature to produce a collection of wind data nodes. These nodes can then be converted into a connected graph with weights corresponding to the temperature, location, and direction difference between the nodes. For example, a cold node next to a warm node would render a low weight connection between them.

## Run Instructions (Python installed on your machine is required)
In order to test this application yourself first run the following in your terminal:

```
git clone https://github.com/allisonlynnbasore14/Spectral_Alg.git
```
Next, navigate to the local version of the repo. Once in the folder run:

```
python windApplicaiton.py "data/windData1.txt"
```

You can substitute any data file in for windData1.txt. All data files are included in the /data folder.

## Results
Below are a few plots from the application. The first image is the actual data plotted with temperature designated by color, direction by arrow direction, and location by x,y coordinate. The second image is this implementation's clustering. The colors represent different clusters.

![Plotted Wind Data](https://github.com/allisonlynnbasore14/resultsImages/originalData.png "Plotted Wind Data")

![Results of Spectral Clustering Implementation](https://github.com/allisonlynnbasore14/resultsImages/resultData.png "Results of Spectral Clustering Implementation")

