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

if __name__ == "__main__":
    gsetfile = open("gset.txt", "r")
    repfile  = open("rep.txt", "r")
    gsetlines = gsetfile.readlines()
    replines  = repfile.readlines()
    gset = [i.strip().split(',') for i in gsetlines]
    gset = [[float(i[0]), float(i[1])] for i in gset]
    gsetx = [i[0] for i in gset]
    gsety = [i[1] for i in gset]
    rep = [i.strip().split(',') for i in replines]
    rep = [[float(i[0]), float(i[1])] for i in rep]
    repx = [i[0] for i in rep]
    repy = [i[1] for i in rep]

    # Part-4a 
    plt.scatter(gsetx,gsety,label="ground set")
    plt.scatter(repx,repy,label="representation set")
    plt.legend()
    plt.savefig('4a.png')
    plt.close()

    #Part-4b (FacilityLocation)
    obj1 = FacilityLocationFunction(n=len(gset),n_rep=len(rep), mode="dense", separate_rep=True,data=np.array(gset),data_rep=np.array(rep),metric="euclidean")
    greedyList = obj1.maximize(budget=10,optimizer='NaiveGreedy', stopIfZeroGain=False, stopIfNegativeGain=False, verbose=False)
    print(greedyList)
    points_selected_idx = [i[0] for i in greedyList]
    points_selected = np.array(gset)[points_selected_idx]
    plt.scatter(gsetx,gsety,label="ground set")
    plt.scatter(repx,repy,label="representation set")
    plt.scatter([i[0] for i in points_selected], [i[1] for i in points_selected],label="FacilityLocation")
    plt.legend()
    plt.savefig('FacilityLocation.png')
    plt.close()

    #Part-4c (Disparity Sum Function)
    obj1 = DisparitySumFunction(n=len(gset), mode="dense",data=np.array(gset),metric="euclidean")
    greedyList = obj1.maximize(budget=10,optimizer='NaiveGreedy', stopIfZeroGain=False, stopIfNegativeGain=False, verbose=False)
    print(greedyList)
    points_selected_idx = [i[0] for i in greedyList]
    points_selected = np.array(gset)[points_selected_idx]
    plt.scatter(gsetx,gsety,label="ground set")
    plt.scatter(repx,repy,label="representation set")
    plt.scatter([i[0] for i in points_selected], [i[1] for i in points_selected],label="DisparitySum")
    plt.legend()
    plt.savefig('DisparitySum.png')
    plt.close()

    #Part-4d (Feature Based Function)
    distanceMatrix = distance.cdist(gset, rep, 'euclidean')
    similarityMatrix = 1-distanceMatrix
    features = []
    for i in range(48):
        #features.append(distanceMatrix[i].tolist())
        features.append(similarityMatrix[i].tolist())

    obj1 = FeatureBasedFunction(n=len(gset),features=features, numFeatures=36, sparse=False,mode=FeatureBased.logarithmic)
    #obj1 = FeatureBasedFunction(n=len(gset),n_rep=len(rep), mode="dense", separate_rep=True,data=np.array(gset),data_rep=np.array(rep))
    greedyList = obj1.maximize(budget=10,optimizer='NaiveGreedy', stopIfZeroGain=False, stopIfNegativeGain=False, verbose=False)
    print(greedyList)
    points_selected_idx = [i[0] for i in greedyList]
    points_selected = np.array(gset)[points_selected_idx]
    plt.scatter(gsetx,gsety,label="ground set")
    plt.scatter(repx,repy,label="representation set")
    plt.scatter([i[0] for i in points_selected], [i[1] for i in points_selected],label="FeatureBased")
    plt.legend()
    plt.savefig('FeatureBased.png')
    plt.close()

    #Part-4f (GraphCut)
    for lambdaval in [0,1,5]:
        obj1 = GraphCutFunction(n=len(gset),n_rep=len(rep), mode="dense", separate_rep=True,data=np.array(gset),data_rep=np.array(rep),metric="euclidean",lambdaVal=lambdaval)
        greedyList = obj1.maximize(budget=10,optimizer='NaiveGreedy', stopIfZeroGain=False, stopIfNegativeGain=False, verbose=False)
        print(greedyList)
        points_selected_idx = [i[0] for i in greedyList]
        points_selected = np.array(gset)[points_selected_idx]
        plt.scatter(gsetx,gsety,label="ground set")
        plt.scatter(repx,repy,label="representation set")
        plt.scatter([i[0] for i in points_selected], [i[1] for i in points_selected],label="GraphCut")
        plt.legend()
        plt.savefig('GraphCut.png'+str(lambdaval))
        plt.close()

    #Part-4g (DisparityMin)
    obj1 = DisparityMinFunction(n=len(gset), mode="dense",data=np.array(gset),metric="euclidean")
    greedyList = obj1.maximize(budget=10,optimizer='NaiveGreedy', stopIfZeroGain=False, stopIfNegativeGain=False, verbose=False)
    print(greedyList)
    points_selected_idx = [i[0] for i in greedyList]
    points_selected = np.array(gset)[points_selected_idx]
    plt.scatter(gsetx,gsety,label="ground set")
    plt.scatter(repx,repy,label="representation set")
    plt.scatter([i[0] for i in points_selected], [i[1] for i in points_selected],label="DisparityMin")
    plt.legend()
    plt.savefig('DisparityMin.png')
    plt.close()

