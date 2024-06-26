# This script is for collecting information about the --auto strategies chosen
# for all problems of the MPTPTP2078, VBT, and SLH-29 datasets.
# 
# 1.) First, we run E on all problems of each dataset
#     with their stdout being redirected into a subfolder for that dataset.
# 2.) For each dataset, we then parse in all of these files and look at the distribution of each
#     search parameter.

import re
from rich import print
from rich.progress import track
import IPython
from glob import glob
import subprocess
import os
import argparse
from collections import Counter, defaultdict
import math
from copy import deepcopy
from multiprocessing import Pool
from functools import reduce
import matplotlib.pyplot as plt
from shutil import rmtree

from e_caller import ECallerHistory
from datetime import datetime, timezone



def eliminateComments(stdout, removeHeuristicNames=True):
    stdout = stdout.split("\n")
    stdout = [l for l in stdout if not l.startswith("#")]

    if removeHeuristicNames:
        stdout = [l if not "heuristic_name: " in l else "heuristic_name: Default" for l in stdout]

    stdout = "\n".join(stdout)
    return stdout




failures = []


def run_eprover(prob_info):
    eproverPath, t, prob, stratsPath, schedule = prob_info
    basename = os.path.basename(prob)

    command = [eproverPath, "--auto-schedule" if schedule else "--auto", "--print-strategy", f"--cpu-limit={int(t*0.75)}", prob]

    try:
        p = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=t)
        if p.returncode == 0:
            with open(f"{stratsPath}/{basename}.strat", "w") as f:
                f.write(eliminateComments(p.stdout.decode("utf-8")))
                if len(p.stdout) < 100:
                    return (f"WARNING: {prob} has a small strategy.", None)
        else:
            failures.append(prob)
            return (f"WARNING: {prob} failed with return code {p.returncode}. \n command={' '.join(command)}", prob)
    except subprocess.TimeoutExpired:
        failures.append(prob)
        return (f"WARNING: {prob} timed out: \n command={' '.join(command)}", prob)

    return (None, None)

def collectStrats(probsPath, eproverPath, stratsPath, schedule):
    print(f"Collecting strats from {probsPath}")

    os.makedirs(stratsPath, exist_ok=True)
    probs = glob(f"{probsPath}/*/test/*.p")
    print(f"Running E on {len(probs)} problems")

    t = 60
    print(f"Max time: {t}s * {len(probs)} = {t*len(probs)}s = {t*len(probs)/60}m = {t*len(probs)/60/60}h")

    prob_info_list = [(eproverPath, t, prob, stratsPath, schedule) for prob in probs]
    
    with Pool(20) as pool:
        for warning_message, prob in track(pool.imap_unordered(run_eprover, prob_info_list), total=len(probs)):
            if warning_message:
                print(warning_message)
            if prob:
                failures.append(prob)

def parseStrat(stratFile, schedule):


    def parseKeyVal(k,v):
        if v == "true":
            return True
        elif v == "false":
            return False
        elif v.startswith('"') and v.endswith('"'):
            return v[1:-1]
        else:
            try:
                return int(v)
            except:
                try:
                    return float(v)
                except:
                    return v
                
    def fixHeuristicDef(strat):
        if "heuristic_def" in strat:
            strat["heuristic_def"] = re.findall(r"([0-9]+)[.*](\w+\([^\)]*\))", strat["heuristic_def"])
            strat["heuristic_def"] = tuple(sorted([(int(w),f) for w,f in strat["heuristic_def"]]))


    if schedule:
        # The file contains multiple strategies.
        # Each strategy is preceded by a name and a newline
        strats = []
        with open(stratFile) as f:
            lines = f.readlines()
            strat = {}
            for i in range(len(lines)):
                if lines[i].startswith("strat name:"):
                    if strat:
                        fixHeuristicDef(strat)
                        strats.append(strat)
                    strat = {}
                elif ":" in lines[i]:
                    k,v = lines[i].strip().split(":")
                    strat[k.strip()] = parseKeyVal(k.strip(),v.strip())
        return strats



    else:
        with open(stratFile) as f:
            lines = f.readlines()
            lines = [l for l in lines if not l.startswith("#") and len(l.strip()) > 1]

        strat = {k.strip():v.strip() for k,v in [l.split(":") for l in lines]}
        assert len(strat) == len(lines) # There should be no duplicate keys.

        for k,v in strat.items():
            strat[k] = parseKeyVal(k,v)
        
        fixHeuristicDef(strat)

        return strat


def filterStratFilesByRun(stratFiles, run):
    """Returns only the strat files for problems solved during the given run.
       run can also be a comma separated list of runs to accommodate the 5-fold cross validation.
    """
    if ',' in run:
        runs = run.split(',')
        d = {}
        for r in runs:
            hist = ECallerHistory.load(r, progress=True)
            d.update(hist.history)
    else:
        hist = ECallerHistory.load(run, progress=True)
        d = hist.history
    prob2Solved = {p: any([x['solved'] for x in l]) for p,l in d.items()}
    strat2Prob = lambda s: os.path.basename(s).replace(".strat", "")
    
    return [s for s in stratFiles if strat2Prob(s) in prob2Solved and prob2Solved[strat2Prob(s)]]

def parseStrats(stratsPath, schedule=False, run=None):
    print(f"Parsing {'all' if run is None else run} strategy files from {stratsPath}")
    stratFiles = glob(stratsPath + "/*.strat")
    stratFiles = stratFiles if run is None else filterStratFilesByRun(stratFiles, run)
    isSpecialFile = lambda f: "common" in f or "distilled" in f or "MASTER" in f
    stratFiles = [f for f in stratFiles if not isSpecialFile(f)]
    parsed = [parseStrat(stratFile, schedule) for stratFile in track(stratFiles)]
    stratNames = [os.path.split(stratFile)[1] for stratFile in stratFiles]
    return stratNames, parsed

def summarizeStrats(strats):

    # Flatten list of lists if strats come from schedules
    if type(strats[0]) == list:
        strats = sum(strats, []) 

    # Use counter to keep track of how many times each value appears for each key.
    masterDict = defaultdict(Counter)
    for strat in strats:
        for k,v in strat.items():
            masterDict[k][v] += 1

    return masterDict



def makeMasterHeuristic(counter: Counter, all_ones: bool):
    maxCEFWeight = 20

    master = defaultdict(lambda:0)
    for CEFs, probCount in counter.items():
        for weight, CEF in CEFs:
            master[CEF] += weight * probCount

    # Scale the weights so that the max weight is maxCEFWeight.
    scalingFactor = maxCEFWeight / max(master.values())
    for k,v in master.items():
        master[k] = math.ceil(v*scalingFactor)

    d = sorted([(v,k) for k,v in master.items()], key=lambda x:x[0], reverse=True)
    if all_ones:
        d = [(1,k) for _,k in d]

    return d

def makeMasterStrat(summary, all_ones, instead=None, keepCommon="heuristic"):
    # Make a master strategy that is the most common value for each key, except for heuristic_def
    # where we call makeMasterHeuristic.
    master = {}
    
    assert keepCommon in ["heuristic", "else"]

    if keepCommon == "heuristic":
        for k,counter in summary.items():
            if k == "heuristic_def":
                master[k] = makeMasterHeuristic(counter, all_ones)
            else:
                master[k] = counter.most_common(1)[0][0] if instead is None else instead[k]

    else:
        for k,counter in summary.items():
            if k == "heuristic_def":
                master[k] = makeMasterHeuristic(counter, all_ones) if instead is None else instead[k]
            else:
                master[k] = counter.most_common(1)[0][0]

    return master




def unparse(k,v):
    if isinstance(v, bool):
        return "true" if v else "false"
    elif isinstance(v, str) and v=="" or k =='sine':
        return f'"{v}"'
    else:
        return str(v)
    
def serializeStrat(summary):
    # Need to convert types back to strings (undoing what parseStrat did)
    # Also need to convert heuristic_def back to a string.

    for k,v in list(summary.items()):
        if k == "heuristic_def":
            l = []
            for _, (weight, cef) in enumerate(v):
                l.append(f"{weight}.{cef}")
            summary[k] = '"(' + ",".join(l) + ')"'
        # elif k == "heuristic_name": 
        #     del summary[k]
        else:
            summary[k] = unparse(k,v)
    
    s = "{\n   {\n"
    indentLevel=2
    for k,v in summary.items():
        if k == "no_preproc":
            indentLevel = 1
            s += "   }\n"
        s += indentLevel*"   " + f"{k}:  {v}\n"
    s += "}"

    return s











def plotStrats(strats, dataset, cutoff=5):
    folder = f"figures/stratCuration/{dataset}"
    rmtree(folder, ignore_errors=True)
    os.makedirs(folder)
    sharedKeys = reduce(set.intersection, [set(strat.keys()) for strat in strats])

    valCounts = {}
    for k in sharedKeys:
        valCounts[k] = Counter([strat[k] for strat in strats])
    
    # only keep keys for which one value doesn't dominate
    # (There must be 2 or more values which each represent >cutoff% of the total.)
    keptKeys = [k for k,v in valCounts.items() if len(v) > 1 and v.most_common(2)[1][1] > len(strats)*(cutoff/100)]

    # Make a histogram with value counts for each key
    # and save it to f"{folder}/{key}.png"
    for k in keptKeys:
        values = [str(strat[k]) for strat in strats]
        plt.title(f"Strategy Distribution for '{k}'")

        # angle labels
        plt.xticks(rotation=60, ha='right')
        plt.hist(values)

        # make sure there's space for the labels
        plt.tight_layout()

        plt.savefig(f"{folder}/{k}.png")
        plt.close()

def stratSortKey(which):

    if which in [0,1]:
        return lambda nameNStrat: nameNStrat[0]
    
    def slhKey(nameNStrat):
        prob = nameNStrat[0].split(".strat")[0]
        probPath = f"/home/jack/Desktop/ATP/GCS/SLH-29/Problems/{prob}"
        with open(probPath) as f:
            lines = f.readlines()
        
        datetimeStr = lines[2].split("%")[1].strip()
        dt_object = datetime.strptime(datetimeStr, '%Y-%m-%d %H:%M:%S.%f')
        return dt_object.replace(tzinfo=timezone.utc).timestamp()

    return slhKey





if __name__ == "__main__":

    parser= argparse.ArgumentParser()
    parser.add_argument("which", type=int)
    parser.add_argument("--makeStrats", action="store_true")
    parser.add_argument("--makeMaster", action="store_true")
    parser.add_argument("--makeMasterSuccess", default="", help="creates MASTERSuccess.strat and MASTERSuccess_RoundRobin.strat including only problems solved in the named run.")
    parser.add_argument("--makeCommonHeuristic", action="store_true", help="makes a variant of each strat with the merged heuristic")
    parser.add_argument("--makeCommonElse", action="store_true", help="makes a variant of each strat with all params equal to that of the master except for heuristic_def")
    parser.add_argument("--makeIncremental", action="store_true", help="makes a variant of each strat that is the merging of all strats that came before in the problem ordering. (for simulating Sledgehammer basically)")
    parser.add_argument("--makePlots", action="store_true")
    parser.add_argument("--ipython", action="store_true")
    parser.add_argument("--schedule", action="store_true", help="use --auto-schedule instead of --auto (and assume multiple strategies in each problem's strat file.)")
    args = parser.parse_args()

    foldPath, ePath, stratPath = [
        ("/home/jack/Desktop/ATP/GCS/MPTPTP2078/Bushy/Folds", "eprover", "strats/MPTPTP2078"),
        ("/home/jack/Desktop/ATP/GCS/VBT/Folds", "eprover", "strats/VBT"),
        ("/home/jack/Desktop/ATP/GCS/SLH-29/Folds", "eprover-ho", "strats/SLH-29")
    ][args.which]

    if args.schedule:
        stratPath = f"schedules/{stratPath}"

    if args.makeStrats:
        collectStrats(foldPath, ePath, stratPath, args.schedule)

    stratNames, strats = parseStrats(stratPath, args.schedule)

    # Make mini master strats for each problem's 
    # strategies from its schedule
    if args.schedule:
        os.makedirs(f"{stratPath}/merged", exist_ok=True)
        for name, toSummarize in zip(stratNames, strats):
            summary = summarizeStrats(toSummarize)
            with open(f"{stratPath}/merged/{name}", "w") as f:
                f.write(serializeStrat(makeMasterStrat(summary, all_ones=False)))
            
            allOnesName = name.split(".strat")[0] + "_allOnes.strat"
            with open(f"{stratPath}/merged/{allOnesName}", "w") as f:
                f.write(serializeStrat(makeMasterStrat(summary, all_ones=True)))



    summary = summarizeStrats(strats)

    if args.makeCommonHeuristic:
        for name, strat in zip(stratNames, strats):
            commonHeuristicStrat = makeMasterStrat(summary, all_ones=False, instead=strat, keepCommon="heuristic")
            os.makedirs(f"{stratPath}/commonHeuristic", exist_ok=True)
            with open(f"{stratPath}/commonHeuristic/{name}", "w") as f:
                f.write(serializeStrat(commonHeuristicStrat))

    if args.makeCommonElse:
        for name, strat in zip(stratNames, strats):
            commonElseStrat = makeMasterStrat(summary, all_ones=False, instead=strat, keepCommon="else")
            os.makedirs(f"{stratPath}/commonElse", exist_ok=True)
            with open(f"{stratPath}/commonElse/{name}", "w") as f:
                f.write(serializeStrat(commonElseStrat))

    if args.makePlots:
        plotStrats(strats, stratPath.split("/")[-1])

    if args.makeMaster:
        with open(f"{stratPath}/MASTER.strat", "w") as f:
            print("Writing master strategy to", f.name)
            f.write(serializeStrat(makeMasterStrat(deepcopy(summary), all_ones=False)))

        with open(f"{stratPath}/MASTER_RoundRobin.strat", "w") as f:
            print("Writing master strategy (all_ones) to", f.name)
            f.write(serializeStrat(makeMasterStrat(deepcopy(summary), all_ones=True)))

    if args.makeMasterSuccess:
        successStratNames, successStrats = parseStrats(stratPath, run=args.makeMasterSuccess)
        successSummary = summarizeStrats(successStrats)
        # IPython.embed()

        with open(f"{stratPath}/MASTERSuccess.strat", "w") as f:
            print("Writing MASTERSuccess.strat to ", f.name)
            f.write(serializeStrat(makeMasterStrat(deepcopy(successSummary), all_ones=False)))

        with open(f"{stratPath}/MASTERSuccess_RoundRobin.strat", "w") as f:
            print("Writing MASTERSuccess_RoundRobin.strat to ", f.name)
            f.write(serializeStrat(makeMasterStrat(deepcopy(successSummary), all_ones=True)))
    
    if args.makeIncremental:
        os.makedirs(f"{stratPath}/incremental", exist_ok=True)
        
        # Sort strats:
        stratNames, strats = zip(*sorted(zip(stratNames, strats), key=stratSortKey(args.which)))
        
        # Incrementally go through them making new versions:
        soFar = []
        for name, strat in track(list(zip(stratNames, strats))):
            soFar.append(strat)
            newStrat = makeMasterStrat(summarizeStrats(soFar), all_ones=False)
            with open(f"{stratPath}/incremental/{name}", "w") as f:
                f.write(serializeStrat(newStrat))

    # print(summary)

    if args.ipython:
        IPython.embed()