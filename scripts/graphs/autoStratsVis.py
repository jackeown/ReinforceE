#######################################################################
# This script is related to scripts/results/omnipotentE.              #
# It takes a prefix like "MPTStrat" and makes a binary heatmap with:  #
#       x-axis: problems sorted by the number of strategies           #
#               that solve that problem                               #
#       y-axis: strategies sorted by the number of problems           #
#               that they solve.                                      #
#       value: 0-1 for solved or not (optionally including proc count #
#######################################################################


import os, sys
sys.path.append(".")

import argparse
from rich.progress import track
from glob import glob
from e_caller import ECallerHistory

from functools import reduce
import matplotlib.pyplot as plt
from collections import defaultdict




def mergeHists(hists):
    """
    Because of 5-fold cross-validation setup, each strategy is spread across
    5 ECallerHistory files. Like so:

    MPTStrat0 -> [MPTStrat00, MPTStrat01, MPTStrat02, MPTStrat03, MPTStrat04]
    MPTStrat15 -> [MPTStrat150, MPTStrat151, MPTStrat152, MPTStrat153, MPTStrat154]
    
    This function regroups them into a single list of ECallerHistories, one per strategy
    """

    whichStrat = lambda run: run[:-1]

    groups = defaultdict(list)
    for run in hists:
        groups[whichStrat(run)].append(hists[run])

    def merge(group):
        group[0].merge(group[1:])
        return group[0]

    merged = [merge(group) for group in groups.values()]
    return merged






def makeHeatmap(args, hists):

    solved = lambda hist, problem: len(hist.history[problem])>0 and hist.history[problem][0]['solved']
    countSolved = lambda hist: len([1 for prob in hist.history if solved(hist, prob)])
    countHists = lambda prob: len([1 for hist in hists if solved(hist, prob)])

    print("Sorting hists by number of solved probs")
    hists = sorted(hists, key=countSolved)

    print("Sorting probs by number of solved hists")
    allProbs = reduce(lambda x,y: x | y, [set(hist.history.keys()) for hist in hists])
    allProbs = sorted(allProbs, key=countHists)

    print("Making heatmap")
    matrix = [[int(not solved(hist, prob)) for prob in allProbs] for hist in hists]


    plt.xlabel("Problems")
    plt.ylabel("Strategies")

    print("Plotting heatmap")
    # aspect ratio makes it more rectangular so it's a better figure...
    plt.imshow(matrix, cmap='Greys', interpolation='nearest', aspect=7)
    os.makedirs("figures/autoStrats", exist_ok=True)
    plt.savefig(f"figures/autoStrats/{args.prefix}.png", dpi=300)


    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('prefix', help="prefix for ECallerHistories to glob for")
    args = parser.parse_args()

    histFiles = glob(f"./ECallerHistory/{args.prefix}[0-9]*")

    name = lambda x: os.path.split(x)[1]
    hists = {name(x):ECallerHistory.load(name(x)) for x in track(histFiles)}

    hists = mergeHists(hists)
    makeHeatmap(args, hists)