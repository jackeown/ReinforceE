import sys, os
sys.path.append(".")
from collections import defaultdict

import argparse
from glob import glob
from rich.progress import track

from e_caller import ECallerHistory


def belongsToStrat(runName):
    """runNames are like MPTStrat43 where this means the 3'rd cross-val fold of strategy 4."""
    return runName[:-1]

def mergeHists(hists):
    x = ECallerHistory()
    x.merge(hists)
    return x

def countSolved(hist):
    count = 0
    for prob in hist.history:
        if len(hist.history[prob]) > 0 and hist.history[prob][0]['solved']:
            count += 1
    return count


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=["MPT", "VBT", "SLH"])
    args = parser.parse_args()

    grouped = defaultdict(list)
    for hist in track(glob(f"./ECallerHistory/{args.dataset}Strat[0-9]*")):
        name = os.path.basename(hist)
        hist = ECallerHistory.load(name)
        grouped[belongsToStrat(name)].append(hist)
    
    stratHists = {}
    for group in grouped:
        stratHists[group] = mergeHists(grouped[group])
    
    for strat in sorted(stratHists, key=lambda x: countSolved(stratHists[x]), reverse=True):
        print(strat, countSolved(stratHists[strat]))