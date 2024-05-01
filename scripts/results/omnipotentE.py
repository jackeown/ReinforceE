#######################################################
# This script takes a prefix like "MPTStrat"          #
# and finds all MPTStrat{i} ECallerHistory's          #
# and merges them into a single MPTStratOmnipotent    #
# ECallerHistory and then saves it.                   #
#######################################################
# This is done to see how good MasterWeighted         #
# (and friends) are, relative to if E could magically #
# pick the best strategy for a problem from its set   #
#######################################################

import os, sys
sys.path.append('.')

import argparse

from rich.progress import track

from glob import glob
from e_caller import ECallerHistory



def should_update(hist, prob, merged):
    infos = hist.history[prob]
    merged_infos = merged.history[prob]
    if len(infos) == 0 or not infos[0]['solved']:
        return False
    
    newly_solved = len(merged_infos) == 0 or not merged_infos[0]['solved']
    return newly_solved or (infos[0]['processed_count'] < merged_infos[0]['processed_count'])


def encorporate(hist, merged):
    """add info from hist into merged if it's relevant"""

    # For each problem:
    #   If not has_solution(hist,problem), do nothing
    #   elif solution_worse_or_equal(hist,problem,merged), do nothing
    #   else replace merged.history[problem] with hist.history[problem]

    for prob in hist.history:
        if should_update(hist, prob, merged):
            merged.history[prob] = hist.history[prob]


def mergeECallerHistories(prefix):
    # [0-9]* means it must have at least one digit after prefix. (To not include any past {prefix}Omnipotent)
    toMerge = glob(f"./ECallerHistory/{prefix}[0-9]*")
    merged = ECallerHistory()
    for hist in track(toMerge):
        runName = os.path.split(hist)[1]
        encorporate(ECallerHistory.load(runName), merged)


    # Remove any empty info lists generated accidentally.
    for key in merged.history:
        if len(merged.history[key]) == 0:
            del merged.history[key]

    merged.save(f"{prefix}Omnipotent")



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", type=str, default="MPTStrat")
    args = parser.parse_args()

    mergeECallerHistories(args.prefix)