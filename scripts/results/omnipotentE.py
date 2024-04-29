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

import sys
sys.path.append('.')

from rich.progress import track

from glob import glob
from e_caller import ECallerHistory



def should_update(hist, prob, merged):
    infos = hist.history[prob]
    merged_infos = merged.history[prob]
    if len(infos) == 0 or not infos[0]['solved']:
        return False
    
    newly_solved = not merged_infos[0]['solved']
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
    toMerge = glob(f"./ECallerHistory/{prefix}*")
    merged = ECallerHistory()
    for hist in track(toMerge):
        encorporate(ECallerHistory.load(hist), merged)

    merged.save(f"{prefix}Omnipotent")


