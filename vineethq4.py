from cProfile import label
from matplotlib import pyplot as plt
import numpy as np
from scipy.spatial import distance
from submodlib_cpp import FeatureBased

from submodlib.functions.facilityLocation import FacilityLocationFunction
from submodlib.functions.disparityMin import DisparityMinFunction
from submodlib.functions.graphCut import GraphCutFunction
from submodlib.functions.disparitySum import DisparitySumFunction
from submodlib.functions.featureBased import FeatureBasedFunction

def points(groundset,repset,func,lambdaval):
    if func == "FacilityLocation":
        FL = FacilityLocationFunction(n=len(groundset),n_rep=len(repset), mode="dense", separate_rep=True,data=np.array(groundset),data_rep=np.array(repset),metric="euclidean")
        greedyList = FL.maximize(budget=10,optimizer='NaiveGreedy', stopIfZeroGain=False, stopIfNegativeGain=False, verbose=False)
    elif func == "DisparitySum":
        DispSum = DisparitySumFunction(n=len(groundset), mode="dense",data=np.array(groundset),metric="euclidean")
        greedyList = DispSum.maximize(budget=10,optimizer='NaiveGreedy', stopIfZeroGain=False, stopIfNegativeGain=False, verbose=False)
    elif func == "DisparityMin":
        DispMin = DisparityMinFunction(n=len(groundset), mode="dense",data=np.array(groundset),metric="euclidean")
        greedyList = DispMin.maximize(budget=10,optimizer='NaiveGreedy', stopIfZeroGain=False, stopIfNegativeGain=False, verbose=False)
    elif func == "GraphCut":
        GC = GraphCutFunction(n=len(groundset),n_rep=len(repset), mode="dense", separate_rep=True,data=np.array(groundset),data_rep=np.array(repset),metric="euclidean",lambdaVal=lambdaval)
        greedyList = GC.maximize(budget=10,optimizer='NaiveGreedy', stopIfZeroGain=False, stopIfNegativeGain=False, verbose=False)
    elif func == "FeatureBased":
        distanceMatrix = distance.cdist(groundset, repset, 'euclidean')
        similarityMatrix = 1-distanceMatrix
        features = []
    for i in range(48):
        #features.append(distanceMatrix[i].tolist())
        features.append(similarityMatrix[i].tolist())

    FB = FeatureBasedFunction(n=len(groundset),features=features, numFeatures=36, sparse=False,mode=FeatureBased.logarithmic)
    greedyList = FB.maximize(budget=10,optimizer='NaiveGreedy', stopIfZeroGain=False, stopIfNegativeGain=False, verbose=False)
    return [i[0] for i in greedyList]

if __name__ == "__main__":
    groundsetfile = open('gset.txt')
    repfile = open('rep.txt')
    groundset = []
    for l in groundsetfile.readlines():
        print(l.strip().split(','))
        groundset.append([float(l.strip().split(',')[0]),float(l.strip().split(',')[1])])
    repset = []
    for l in repfile.readlines():
        repset.append([float(l.strip().split(',')[0]),float(l.strip().split(',')[1])])

    #Plotting all points of groundset and repset
    plt.scatter([i[0] for i in groundset],[i[1] for i in groundset],label="ground set")
    plt.scatter([i[0] for i in repset],[i[1] for i in repset],label="representation set")
    plt.legend()
    plt.savefig('points.png')
    plt.close()

    for s in ["FacilityLoaction","DisparitySum","FeatureBased","DisparityMin"]:
        pts_sel_idx = points(groundset,repset,s,1)
        pts_sel = np.array(groundset)[pts_sel_idx]
        plt.scatter([i[0] for i in pts_sel], [i[1] for i in pts_sel],label=s)
        plt.legend()
        plt.savefig(s+'.png')
        plt.close()

    #GraphCut
    for lambdaval in [0,1,5]:
        pts_sel_idx = points(groundset,repset,"GraphCut",lambdaval)
        pts_sel = np.array(groundset)[pts_sel_idx]
        plt.scatter([i[0] for i in pts_sel], [i[1] for i in pts_sel],label=s+str(lambdaval))
        plt.legend()
        plt.savefig(s+str(lambdaval)+'.png')
        plt.close()

