import subprocess
import argparse
import psutil
from time import sleep
import os, sys, time
from glob import glob
from collections import defaultdict

import select


def getStratCounts(stratPath):
        strats = glob(f"{stratPath}/*.p.strat")
        stratContents = defaultdict(list)
        for strat in strats:
                with open(strat) as f:
                        stratContents[f.read()].append(strat)
        return stratContents

stratPath = "/home/jack/Desktop/ReinforceE/strats"
stratPaths = {
        'mpt': f"{stratPath}/MPTPTP2078",
        'vbt': f"{stratPath}/VBT",
        'slh': f"{stratPath}/SLH-29"
}


uniqueStrats = {}
for ds, stratPath in stratPaths.items():
        strats = getStratCounts(stratPath)
        total = sum([len(x) for x in strats.values()])
        unique = len(strats)
        print(f"{total} strats total for {ds}, {unique} unique", sorted((len(x) for x in strats.values()), reverse=True))
        uniqueStrats[ds] = [l[0] for l in strats.values()]




MPTPPath =                      "--folds_path=/home/jack/Desktop/ATP/GCS/MPTPTP2078/Bushy/Folds"
MPT_CPU_LIMIT = 60
MPT_CPU_LIMIT_STR =            f"--soft_cpu_limit={MPT_CPU_LIMIT   } --cpu_limit={MPT_CPU_LIMIT+5}"

SLHPath =                       "--folds_path=/home/jack/Desktop/ATP/GCS/SLH-29/Folds"
SLH_CPU_LIMIT = 60
SLH_CPU_LIMIT_STR =            f"--soft_cpu_limit={SLH_CPU_LIMIT} --cpu_limit={SLH_CPU_LIMIT+5}"

VBTPath =                       "--folds_path=/home/jack/Desktop/ATP/GCS/VBT/Folds"
VBT_CPU_LIMIT = 60
VBT_CPU_LIMIT_STR =            f"--soft_cpu_limit={VBT_CPU_LIMIT} --cpu_limit={VBT_CPU_LIMIT+5}"

common_flags = f"--num_workers=8 --entropy_weight=0.001 --critic_weight=0.1 --max_blame=50000 --n_layers=2 --n_units=80 --epochs=1 --batch_size=5 --ppo_batch_size=128 --LAMBDA=0.987 --discount_factor=0.998 --test_num=1"



experiment_formats = [
    # MPTPTP2078
    lambda i: f'python tmux_magic_main.py --main_args="{common_flags} {MPT_CPU_LIMIT_STR} --auto --policy_type=none --eprover_path=eprover --strat_file={uniqueStrats["mpt"][i]}" {MPTPPath} MPTStrat{i} --test',
    
    # VBT
    lambda i: f'python tmux_magic_main.py --main_args="{common_flags} {VBT_CPU_LIMIT_STR} --auto --policy_type=none --eprover_path=eprover --strat_file={uniqueStrats["vbt"][i]}" {VBTPath} VBTStrat{i} --test',

    # SLH
    lambda i: f'python tmux_magic_main.py --main_args="{common_flags} {SLH_CPU_LIMIT_STR} --auto --policy_type=none --eprover_path=eprover-ho --strat_file={uniqueStrats["slh"][i]}" {SLHPath} SLHStrat{i} --test',
]


MPTExperiments = [experiment_formats[0](i) for i in range(len(uniqueStrats['mpt']))]
VBTExperiments = [experiment_formats[1](i) for i in range(len(uniqueStrats['vbt']))]
SLHExperiments = [experiment_formats[2](i) for i in range(len(uniqueStrats['slh']))]
experiments_to_run = SLHExperiments[15:] # VBTExperiments[28:] + SLHExperiments

# Check that CPU is not too busy
def too_busy(percent):
    cpu_utilizations = []
    for i in range(10):
        cpu_utilizations.append(psutil.cpu_percent())
        sleep(2)

    cpu_utilization = max(cpu_utilizations)

    return cpu_utilization > percent


def runExperiment(exp):
    print("Running experiment: {}".format(exp))
    subprocess.call(exp, shell=True)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu_threshold", type=float, default=30.0)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    
    if args.dry_run:
        for exp in experiments_to_run:
            print(exp)
            print("\n\n")
    else:
        t1 = time.time()
        n = len(experiments_to_run)

        i=0
        while len(experiments_to_run):
            while too_busy(args.cpu_threshold):
                
                # if user enters anything, run the next experiment
                select.select([sys.stdin], [], [], 0.0)
                if sys.stdin in select.select([sys.stdin], [], [], 0.0)[0]:
                    line = input()
                    break

                sleep(100)


            runExperiment(experiments_to_run.pop(0))
            print("Remaining experiments: {}".format(len(experiments_to_run)))
            elapsedHours = (time.time() - t1) / 3600
            expRate = i / elapsedHours # i experiments completed so far in elapsedHours time
            safediv = lambda a, b: -1 if b == 0 else a / b
            hoursRemaining = safediv(len(experiments_to_run), expRate)
            print(f"Elapsed: {elapsedHours}, ETA: {hoursRemaining} hours ({hoursRemaining / 24} days)")
            i += 1
            sleep(500)

        

