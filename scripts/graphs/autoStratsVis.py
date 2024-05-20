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

import random
import argparse
from rich.progress import track
from glob import glob
from e_caller import ECallerHistory
from itertools import cycle

from functools import reduce
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np




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
    hists = sorted(hists, key=countSolved, reverse=True) # best strategies on top

    print("Sorting probs by number of solved hists")
    allProbs = reduce(lambda x,y: x | y, [set(hist.history.keys()) for hist in hists])
    allProbs = sorted(allProbs, key=countHists, reverse=True) # easy problems on left

    print("Making heatmap")
    matrix = [[int(solved(hist, prob)) for prob in allProbs] for hist in hists]
    return matrix



def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def makeDummyHeatmap(numStrats=60, numProbs=2078):
    gainP, gainS = 30, 0.5
    midP, midS = numProbs // 2, numStrats // 2
    bias = lambda i: np.tanh(gainS*(i-midS)/numStrats)*0.1
    p = lambda j: sigmoid(gainP*(j-midP)/numProbs)
    
    original = np.array([[random.random() + bias(i) < p(j) for j in range(numProbs)] for i in range(numStrats)])

    # sort rows by their sums:
    rowsSorted = original[np.argsort(original.sum(axis=1))[::-1]]

    # now sort the columns of rowsSorted by the sum of each column:
    colsSorted = rowsSorted[:, np.argsort(rowsSorted.sum(axis=0))[::-1]]

    return colsSorted







def plotHeatmap(matrix, aspect, extraVlines={}):
    plt.xlabel("Problems (Easy <-> Hard)", fontsize=7)
    plt.ylabel("Strategies (Worst <-> Best)", fontsize=7)

    print("Plotting heatmap")

    rowSums = np.sum(matrix, axis=1)
    rowSumYs = np.arange(len(rowSums))

    plt.imshow(matrix, cmap='Greys', interpolation='nearest', aspect=aspect)
    plt.plot(rowSums, rowSumYs, label="Problems Solved", color='red', alpha=0.7)


    # Vertical lines for showing:
    # 1.) The number of problems solved by no strategy.
    numSolvedByNone = np.sum(np.sum(matrix, axis=0) == 0) 
    plt.vlines(len(matrix[0]) - numSolvedByNone, 0, len(matrix)-1, color='green', alpha=0.7, label="AutoAll",linewidth=1)

    # 2.) The number of problems solved by all strategies.
    numSolvedByAll = np.sum(np.sum(matrix, axis=0) == len(matrix))
    plt.vlines(numSolvedByAll, 0, len(matrix)-1, color='blue', alpha=0.85, label="Solved by all", linewidth=1)

    # 3.) The number of problems solved by the best strategy:
    numSolvedByBest = max(rowSums)
    plt.vlines(numSolvedByBest, 0, len(matrix)-1, color='orange', alpha=0.85, label="Solved by best", linewidth=1)


    otherColors = cycle(['yellow', 'indigo', 'chartreuse', 'magenta', 'cyan', 'pink', 'grey', 'black', 'dodgerblue'])
    for k,v in extraVlines.items():
        plt.vlines(v, 0, len(matrix)-1, color=next(otherColors), alpha=0.85, label=k, linewidth=1)


    plt.tick_params(axis='both', which='minor', labelsize=7)
    plt.tick_params(axis='both', which='major', labelsize=7)


    # make a smaller legend that doesn't overlap with the plot:
    # plt.legend(fontsize=5, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.0)
    # plt.legend(loc='upper right', fontsize=5)

    ax = plt.gca()
    pos = ax.get_position()
    # ax.set_position([pos.x0, pos.y0*1.5, pos.width, pos.height])
    ax.legend(
        fontsize=5,
        loc='upper center', 
        bbox_to_anchor=(0.5, 1.75),
        ncol=3, 
    )


    os.makedirs("figures/autoStrats", exist_ok=True)
    plt.savefig(f"figures/autoStrats/{args.prefix}.png", dpi=300)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('prefix', help="prefix for ECallerHistories to glob for")
    parser.add_argument('--extraVlines', default="", help="e.g. MyMethod1:2435,MyMethod2:2358")
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--fakeHW', help="a:b for a being the number of strategies and b being the number of problems", default="60:8000")
    parser.add_argument('--aspect', type=float, default=20.0)
    args = parser.parse_args()

    extraVlines = {x.split(":")[0]:int(x.split(":")[1]) for x in args.extraVlines.split(",")}

    if args.dry_run:
        numStrats, numProbs = map(int, args.fakeHW.split(":"))
        plotHeatmap(makeDummyHeatmap(numStrats, numProbs), args.aspect, extraVlines=extraVlines)
    else:
        histFiles = glob(f"./ECallerHistory/{args.prefix}[0-9]*")
        name = lambda x: os.path.split(x)[1]
        # hists = {name(x):ECallerHistory.load(name(x), keysToDeleteFromInfos={"stdout", "stderr", "states", "actions", "rewards"}) for x in track(histFiles)}
        # load in parallel:

        print(f"Loading {len(histFiles)} files...")
        from joblib import Parallel, delayed
        hists = Parallel(n_jobs=25,verbose=13)(delayed(ECallerHistory.load)(name(x), keysToDeleteFromInfos={"stdout", "stderr", "states", "actions", "rewards"}) for x in histFiles)
        hists = {name(x):h for x,h in zip(histFiles, hists)}

        hists = mergeHists(hists)
        plotHeatmap(makeHeatmap(args, hists), args.aspect, extraVlines=extraVlines)