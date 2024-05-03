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



def makeHeatmap(args, hists):

    solved = lambda hist, problem: hist.history[problem] and hist.history[problem][0]['solved']
    countSolved = lambda hist: len([1 for prob in hist.history if solved(hist, prob)])
    countHists = lambda prob: len([1 for hist in hists if solved(hist, prob)])

    print("Sorting hists by number of solved probs")
    hists = sorted(hists, key=countSolved)

    print("Sorting probs by number of solved hists")
    allProbs = reduce(lambda x,y: x | y, [set(hist.history.keys()) for hist in hists])
    allProbs = sorted(allProbs, key=countHists)

    print("Making heatmap")
    matrix = [[int(solved(hist, prob)) for prob in allProbs] for hist in hists]


    plt.xlabel("Problems")
    plt.ylabel("Strategies")

    print("Plotting heatmap")
    # aspect ratio makes it more rectangular so it's a better figure...
    plt.imshow(matrix, cmap='Greys', interpolation='nearest', aspect=7)
    os.makedirs("figures/autoStrats", exist_ok=True)
    plt.savefig(f"figures/autoStrats/{args.prefix}.png")


    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('prefix', help="prefix for ECallerHistories to glob for")
    args = parser.parse_args()

    histFiles = glob(f"./ECallerHistory/{args.prefix}[0-9]*")
    hists = [ECallerHistory.load(os.path.split(x)[1]) for x in track(histFiles)]

    makeHeatmap(args, hists)