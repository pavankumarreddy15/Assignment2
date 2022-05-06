from cProfile import label
from matplotlib import pyplot as plt

from submodlib.functions.facilityLocation import FacilityLocationFunction
from submodlib.helper import create_kernel
from submodlib.functions.disparityMin import DisparityMinFunction
from submodlib.functions.logDeterminant import LogDeterminantFunction
from submodlib.functions.graphCut import GraphCutFunction

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

    #Part-4b (FacilityLocation)
    obj1 = FacilityLocationFunction(n=len(gset), mode="dense", separate_rep=True,data=gset,data_rep=rep)
    greedyList = obj1.maximize(budget=10,optimizer='NaiveGreedy', stopIfZeroGain=False, stopIfNegativeGain=False, verbose=False)
    print(greedyList)
