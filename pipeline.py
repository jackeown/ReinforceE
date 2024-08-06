import subprocess
import argparse
import psutil
from time import sleep
import os
import sys
import time

import select


MPTPPath =                      "--folds_path=/home/jack/Desktop/ATP/GCS/MPTPTP2078/Bushy/Folds"
MPTPStratFile =                 "--strat_file=/home/jack/Desktop/ReinforceE/strats/MPTPTP2078/MASTER.strat"
MPTPStratFileRR =               "--strat_file=/home/jack/Desktop/ReinforceE/strats/MPTPTP2078/MASTER_RoundRobin.strat"
MPTPSuccessStratFile =          "--strat_file=/home/jack/Desktop/ReinforceE/strats/MPTPTP2078/MASTERSuccess.strat"
MPTPSuccessStratFileRR =        "--strat_file=/home/jack/Desktop/ReinforceE/strats/MPTPTP2078/MASTERSuccess_RoundRobin.strat"
MPTPSchedStrats =               "--strat_file=/home/jack/Desktop/ReinforceE/schedules/strats/MPTPTP2078/merged"

MPT_CommonHeuristic_StratDir =  "--strat_file=/home/jack/Desktop/ReinforceE/strats/MPTPTP2078/commonHeuristic"
MPT_CommonElse_StratDir =       "--strat_file=/home/jack/Desktop/ReinforceE/strats/MPTPTP2078/commonElse"
MPT_Incremental_StratDir =      "--strat_file=/home/jack/Desktop/ReinforceE/strats/MPTPTP2078/incremental"

MPT_CPU_LIMIT = 60
MPT_CPU_LIMIT_STR =            f"--soft_cpu_limit={MPT_CPU_LIMIT   } --cpu_limit={MPT_CPU_LIMIT+5}"
MPT_CPU_LIMIT_MODEL_STR =      f"--soft_cpu_limit={MPT_CPU_LIMIT+15} --cpu_limit={MPT_CPU_LIMIT+20}"

SLHPath =                       "--folds_path=/home/jack/Desktop/ATP/GCS/SLH-29/Folds"
SLHStratFile =                  "--strat_file=/home/jack/Desktop/ReinforceE/strats/SLH-29/MASTER.strat"
SLHStratFileRR =                "--strat_file=/home/jack/Desktop/ReinforceE/strats/SLH-29/MASTER_RoundRobin.strat"
SLHSuccessStratFile =           "--strat_file=/home/jack/Desktop/ReinforceE/strats/SLH-29/MASTERSuccess.strat"
SLHSuccessStratFileRR =         "--strat_file=/home/jack/Desktop/ReinforceE/strats/SLH-29/MASTERSuccess_RoundRobin.strat"
SLHSchedStrats =                "--strat_file=/home/jack/Desktop/ReinforceE/schedules/strats/SLH-29/merged"

SLH_CommonHeuristic_StratDir =  "--strat_file=/home/jack/Desktop/ReinforceE/strats/SLH-29/commonHeuristic"
SLH_CommonElse_StratDir =       "--strat_file=/home/jack/Desktop/ReinforceE/strats/SLH-29/commonElse"
SLH_Incremental_StratDir =      "--strat_file=/home/jack/Desktop/ReinforceE/strats/SLH-29/incremental"

SLH_CPU_LIMIT = 60
SLH_CPU_LIMIT_STR =            f"--soft_cpu_limit={SLH_CPU_LIMIT} --cpu_limit={SLH_CPU_LIMIT+5}"
SLH_CPU_LIMIT_MODEL_STR =      f"--soft_cpu_limit={SLH_CPU_LIMIT+15} --cpu_limit={SLH_CPU_LIMIT+20}"

VBTPath =                       "--folds_path=/home/jack/Desktop/ATP/GCS/VBT/Folds"
VBTStratFile =                  "--strat_file=/home/jack/Desktop/ReinforceE/strats/VBT/MASTER.strat"
VBTStratFileRR =                "--strat_file=/home/jack/Desktop/ReinforceE/strats/VBT/MASTER_RoundRobin.strat"
VBTSuccessStratFile =           "--strat_file=/home/jack/Desktop/ReinforceE/strats/VBT/MASTERSuccess.strat"
VBTSuccessStratFileRR =         "--strat_file=/home/jack/Desktop/ReinforceE/strats/VBT/MASTERSuccess_RoundRobin.strat"
VBTSchedStrats =                "--strat_file=/home/jack/Desktop/ReinforceE/schedules/strats/VBT/merged"

VBT_CommonHeuristic_StratDir =  "--strat_file=/home/jack/Desktop/ReinforceE/strats/VBT/commonHeuristic"
VBT_CommonElse_StratDir =       "--strat_file=/home/jack/Desktop/ReinforceE/strats/VBT/commonElse"
VBT_Incremental_StratDir =      "--strat_file=/home/jack/Desktop/ReinforceE/strats/VBT/incremental"

VBT_CPU_LIMIT = 60
VBT_CPU_LIMIT_STR =            f"--soft_cpu_limit={VBT_CPU_LIMIT} --cpu_limit={VBT_CPU_LIMIT+5}"
VBT_CPU_LIMIT_MODEL_STR =      f"--soft_cpu_limit={VBT_CPU_LIMIT+15} --cpu_limit={VBT_CPU_LIMIT+20}"


mem_size=20
# mem_size=1
state_dim = mem_size*5
# old settings...
# common_flags = f"--num_workers=6 --entropy_weight=2e-6 --max_blame=6000 --lr=3e-6 --n_layers=3 --n_units=100 --epochs=1 --batch_size=8 --ppo_batch_size=512 --LAMBDA=0.96 --state_dim={state_dim}"

# modelled after local testing on MPTPTP2078
common_flags = f"--num_workers=2 --entropy_weight=0.001 --critic_weight=0.1 --max_blame=50000 --n_layers=2 --n_units=80 --epochs=1 --batch_size=5 --ppo_batch_size=128 --LAMBDA=0.987 --discount_factor=0.998 --state_dim={state_dim} "
common_flags += f"--max_train_steps=4500"

TMP_CPU_LIMIT_MODEL_STR = "--soft_cpu_limit=5 --cpu_limit=10"

train_experiments = [
    # Neural Nets
    # f'python tmux_magic_main.py --main_args="{common_flags} --lr=8e-4 {SLH_CPU_LIMIT_MODEL_STR} --policy_type=nn --eprover_path=eprover_RL-ho_HIST_1 {SLHStratFile} --auto" {SLHPath} SLHNN1Hist',
    # f'python tmux_magic_main.py --main_args="{common_flags} --lr=8e-4 {MPT_CPU_LIMIT_MODEL_STR} --policy_type=nn --eprover_path=eprover_RL_HIST_1 {MPTPStratFile} --auto" {MPTPPath} MPTNN1Hist',
    # f'python tmux_magic_main.py --main_args="{common_flags} --lr=8e-4 {VBT_CPU_LIMIT_MODEL_STR} --policy_type=nn --eprover_path=eprover_RL_HIST_1 {VBTStratFile} --auto" {VBTPath} VBTNN1Hist',

    # f'python tmux_magic_main.py --main_args="{common_flags} --lr=4e-4 {MPT_CPU_LIMIT_MODEL_STR} --policy_type=nn --eprover_path=eprover_RL_HIST {MPTPStratFile} --auto" {MPTPPath} MPTNN20Hist',


    # Constant Categorical Distribution
    # f'python tmux_magic_main.py --main_args="{common_flags} --lr=2e-5 {VBT_CPU_LIMIT_MODEL_STR} --policy_type=constcat --eprover_path=eprover_RL_HIST_1 {VBTStratFile} --auto" {VBTPath} VBTConstCat1Hist',
    # f'python tmux_magic_main.py --main_args="{common_flags} --lr=2e-5 {SLH_CPU_LIMIT_MODEL_STR} --policy_type=constcat --eprover_path=eprover_RL-ho_HIST_1 {SLHStratFile} --auto" {SLHPath} SLHConstCat1Hist',
    # f'python tmux_magic_main.py --main_args="{common_flags} --lr=2e-5 {MPT_CPU_LIMIT_MODEL_STR} --policy_type=constcat --eprover_path=eprover_RL_HIST_1 {MPTPStratFile} --auto" {MPTPPath} MPTConstCat1Hist',



]

test_experiments = [
    # VBT Experiments...
    # f'python tmux_magic_main.py --main_args="{common_flags} {VBT_CPU_LIMIT_STR} --auto --policy_type=none --eprover_path=eprover {VBTStratFile}" {VBTPath} VBTRoundRobin --test',
    # f'python tmux_magic_main.py --main_args="{common_flags} {VBT_CPU_LIMIT_STR} --auto --policy_type=none --eprover_path=eprover {VBTStratFileRR}" {VBTPath} VBTRoundRobinAllOnes --test',
    # f'python tmux_magic_main.py --main_args="{common_flags} {VBT_CPU_LIMIT_STR} --auto --policy_type=none --eprover_path=eprover --test_num=1" {VBTPath} VBTAuto --test',
    f'python tmux_magic_main.py --main_args="{common_flags} {VBT_CPU_LIMIT_STR} --auto_sched --policy_type=none --eprover_path=eprover --test_num=1" {VBTPath} VBTAutoSched --test',
    # f'python tmux_magic_main.py --main_args="{common_flags} {VBT_CPU_LIMIT_MODEL_STR} --auto --policy_type=nn --eprover_path=eprover_RL_HIST_1 {VBTStratFile}" {VBTPath} VBTNN1Hist --test',
    # f'python tmux_magic_main.py --main_args="{common_flags} {VBT_CPU_LIMIT_MODEL_STR} --auto --policy_type=constcat --eprover_path=eprover_RL_HIST_1 {VBTStratFile}" {VBTPath} VBTConstCat1Hist --test',

    # MPTPTP2078 Experiments...
    # f'python tmux_magic_main.py --main_args="{common_flags} {MPT_CPU_LIMIT_STR} --auto --policy_type=none --eprover_path=eprover {MPTPStratFile}" {MPTPPath} MPTRoundRobin --test',
    # f'python tmux_magic_main.py --main_args="{common_flags} {MPT_CPU_LIMIT_STR} --auto --policy_type=none --eprover_path=eprover {MPTPStratFileRR}" {MPTPPath} MPTRoundRobinAllOnes --test',
    # f'python tmux_magic_main.py --main_args="{common_flags} {MPT_CPU_LIMIT_STR} --auto --policy_type=none --eprover_path=eprover --test_num=1" {MPTPPath} MPTAutoAudit --test',
    f'python tmux_magic_main.py --main_args="{common_flags} {MPT_CPU_LIMIT_STR} --auto_sched --policy_type=none --eprover_path=eprover --test_num=1" {MPTPPath} MPTAutoSched --test',
    # f'python tmux_magic_main.py --main_args="{common_flags} {MPT_CPU_LIMIT_MODEL_STR} --auto --policy_type=constcat --eprover_path=eprover_RL_HIST_1 {MPTPStratFile}" {MPTPPath} MPTConstCat1Hist --test',
    # f'python tmux_magic_main.py --main_args="{common_flags} {MPT_CPU_LIMIT_MODEL_STR} --auto --policy_type=nn --eprover_path=eprover_RL_HIST_1 {MPTPStratFile}" {MPTPPath} MPTNN1Hist --test',

    # f'python tmux_magic_main.py --main_args="{common_flags} {MPT_CPU_LIMIT_MODEL_STR} --auto --policy_type=nn --eprover_path=eprover_RL_HIST {MPTPStratFile}" {MPTPPath} MPTNN20Hist --test',

    # # SLH Experiments...
    # f'python tmux_magic_main.py --main_args="{common_flags} {SLH_CPU_LIMIT_STR} --auto --policy_type=none --eprover_path=eprover-ho {SLHStratFile}" {SLHPath} SLHRoundRobin --test',
    # f'python tmux_magic_main.py --main_args="{common_flags} {SLH_CPU_LIMIT_STR} --auto --policy_type=none --eprover_path=eprover-ho {SLHStratFileRR}" {SLHPath} SLHRoundRobinAllOnes --test',
    # f'python tmux_magic_main.py --main_args="{common_flags} {SLH_CPU_LIMIT_STR} --auto --policy_type=none --eprover_path=eprover-ho --test_num=1" {SLHPath} SLHAuto --test',
    # f'python tmux_magic_main.py --main_args="{common_flags} {SLH_CPU_LIMIT_STR} --auto_sched --policy_type=none --eprover_path=eprover-ho --test_num=1" {SLHPath} SLHAutoSched --test',
    # f'python tmux_magic_main.py --main_args="{common_flags} {SLH_CPU_LIMIT_MODEL_STR} --auto --policy_type=constcat --eprover_path=eprover_RL-ho_HIST_1 {SLHStratFile}" {SLHPath} SLHConstCat1Hist --test',
    # f'python tmux_magic_main.py --main_args="{common_flags} {SLH_CPU_LIMIT_MODEL_STR} --auto --policy_type=nn --eprover_path=eprover_RL-ho_HIST_1 {SLHStratFile}" {SLHPath} SLHNN1Hist --test',   

    # # CommonHeuristic Experiments...
    # f'python tmux_magic_main.py --main_args="{common_flags} --test_num=1 {VBT_CPU_LIMIT_STR} --auto {VBT_CommonHeuristic_StratDir} --policy_type=none --eprover_path=eprover" {VBTPath} VBTCommonHeuristic --test',
    # f'python tmux_magic_main.py --main_args="{common_flags} --test_num=1 {MPT_CPU_LIMIT_STR} --auto {MPT_CommonHeuristic_StratDir} --policy_type=none --eprover_path=eprover" {MPTPPath} MPTCommonHeuristic --test',
    # f'python tmux_magic_main.py --main_args="{common_flags} --test_num=1 {SLH_CPU_LIMIT_STR} --auto {SLH_CommonHeuristic_StratDir} --policy_type=none --eprover_path=eprover-ho" {SLHPath} SLHCommonHeuristic --test',

    # # CommonElse Experiments...
    # f'python tmux_magic_main.py --main_args="{common_flags} --test_num=1 {VBT_CPU_LIMIT_STR} --auto {VBT_CommonElse_StratDir} --policy_type=none --eprover_path=eprover" {VBTPath} VBTCommonElse --test',
    # f'python tmux_magic_main.py --main_args="{common_flags} --test_num=1 {MPT_CPU_LIMIT_STR} --auto {MPT_CommonElse_StratDir} --policy_type=none --eprover_path=eprover" {MPTPPath} MPTCommonElse --test',
    # f'python tmux_magic_main.py --main_args="{common_flags} --test_num=1 {SLH_CPU_LIMIT_STR} --auto {SLH_CommonElse_StratDir} --policy_type=none --eprover_path=eprover-ho" {SLHPath} SLHCommonElse --test',

    # # MASTERSuccess Experiments...
    # f'python tmux_magic_main.py --main_args="{common_flags} --test_num=1 {MPT_CPU_LIMIT_STR} --auto --policy_type=none --eprover_path=eprover {MPTPSuccessStratFile}" {MPTPPath} MPTSuccessRoundRobin --test',
    # f'python tmux_magic_main.py --main_args="{common_flags} --test_num=1 {MPT_CPU_LIMIT_STR} --auto --policy_type=none --eprover_path=eprover {MPTPSuccessStratFileRR}" {MPTPPath} MPTSuccessRoundRobinAllOnes --test',

    # f'python tmux_magic_main.py --main_args="{common_flags} --test_num=1 {SLH_CPU_LIMIT_STR} --auto --policy_type=none --eprover_path=eprover-ho {SLHSuccessStratFile}" {SLHPath} SLHSuccessRoundRobin --test',
    # f'python tmux_magic_main.py --main_args="{common_flags} --test_num=1 {SLH_CPU_LIMIT_STR} --auto --policy_type=none --eprover_path=eprover-ho {SLHSuccessStratFileRR}" {SLHPath} SLHSuccessRoundRobinAllOnes --test',

    # f'python tmux_magic_main.py --main_args="{common_flags} --test_num=1 {VBT_CPU_LIMIT_STR} --auto --policy_type=none --eprover_path=eprover {VBTSuccessStratFile}" {VBTPath} VBTSuccessRoundRobin --test',
    # f'python tmux_magic_main.py --main_args="{common_flags} --test_num=1 {VBT_CPU_LIMIT_STR} --auto --policy_type=none --eprover_path=eprover {VBTSuccessStratFileRR}" {VBTPath} VBTSuccessRoundRobinAllOnes --test',

    # MasterIncremental Experiments...
    # f'python tmux_magic_main.py --main_args="{common_flags} --test_num=1 {VBT_CPU_LIMIT_STR} --auto --policy_type=none --eprover_path=eprover {VBT_Incremental_StratDir}" {VBTPath} VBTRoundRobinIncremental --test',
    # f'python tmux_magic_main.py --main_args="{common_flags} --test_num=1 {MPT_CPU_LIMIT_STR} --auto --policy_type=none --eprover_path=eprover {MPT_Incremental_StratDir}" {MPTPPath} MPTRoundRobinIncremental --test',
    # f'python tmux_magic_main.py --main_args="{common_flags} --test_num=1 {SLH_CPU_LIMIT_STR} --auto --policy_type=none --eprover_path=eprover-ho {SLH_Incremental_StratDir}" {SLHPath} SLHRoundRobinIncremental --test',


    # MasterWeightedRoundRobin Experiments...
    # f'python tmux_magic_main.py --main_args="{common_flags} --test_num=1 {MPT_CPU_LIMIT_STR} --auto --policy_type=none --eprover_path=eprover_fair_round_robin {MPTPStratFile}" {MPTPPath} MPTMasterRR --test',
    # f'python tmux_magic_main.py --main_args="{common_flags} --test_num=1 {VBT_CPU_LIMIT_STR} --auto --policy_type=none --eprover_path=eprover_fair_round_robin {VBTStratFile}" {VBTPath} VBTMasterRR --test',
    # f'python tmux_magic_main.py --main_args="{common_flags} --test_num=1 {SLH_CPU_LIMIT_STR} --auto --policy_type=none --eprover_path=/home/jack/eprover/eprover_fair_round_robin/PROVER/eprover-ho {SLHStratFile}" {SLHPath} SLHMasterRR --test',


    # ScheduleMerging Experiments...
    # f'python tmux_magic_main.py --main_args="{common_flags} --test_num=1 {MPT_CPU_LIMIT_STR} --auto --policy_type=none --eprover_path=eprover {MPTPSchedStrats}" {MPTPPath} MPTSchedMerge --test',
    # f'python tmux_magic_main.py --main_args="{common_flags} --test_num=1 {VBT_CPU_LIMIT_STR} --auto --policy_type=none --eprover_path=eprover {VBTSchedStrats}" {VBTPath} VBTSchedMerge --test',

    # f'python tmux_magic_main.py --main_args="{common_flags} --test_num=1 {SLH_CPU_LIMIT_STR} --auto --policy_type=none --eprover_path=eprover-ho {SLH_Incremental_StratDir}" {SLHPath} SLHRoundRobinIncremental --test',



]




# model_prefixes = ["models/VBTConstCat1Hist", "models/SLHConstCat1Hist", "models/MPTConstCat1Hist"]
# master_strats = ["strats/VBT/MASTER.strat", "strats/SLH-29/MASTER.strat", "strats/MPTPTP2078/MASTER.strat"]
# strat_path_prefixes = ["strats/VBT/distilledVBT", "strats/SLH-29/distilledSLH", "strats/MPTPTP2078/distilledMPT"]

# # gains = [2,5,10]
# gains = [5]
# for gain in gains:
#     for model_prefix, master_strat, strat_path_prefix in zip(model_prefixes, master_strats, strat_path_prefixes):
#         for i in range(5):
#             command = f"python scripts/others/distill.py {model_prefix}{i}.pt {master_strat} {strat_path_prefix}_gain{gain}_{i}.strat --gain={gain}"
#             subprocess.call(command, shell=True)


# # one for VBT,SLH, and MPT
# distill_experiments = []
# for gain in gains:
#     distill_experiments += [
#         f'python tmux_magic_main.py --main_args="{common_flags} --test_num=1 {VBT_CPU_LIMIT_STR} --policy_type=none --eprover_path=eprover --strat_file=strats/VBT/distilledVBT_gain{gain}_0.strat --auto" {VBTPath} VBTConstCatDistilled_gain{gain}_ --update_strat_file_suffix --test',
#         f'python tmux_magic_main.py --main_args="{common_flags} --test_num=1 {SLH_CPU_LIMIT_STR} --policy_type=none --eprover_path=eprover-ho --strat_file=strats/SLH-29/distilledSLH_gain{gain}_0.strat --auto" {SLHPath} SLHConstCatDistilled_gain{gain}_ --update_strat_file_suffix --test',
#         f'python tmux_magic_main.py --main_args="{common_flags} --test_num=1 {MPT_CPU_LIMIT_STR} --policy_type=none --eprover_path=eprover --strat_file=strats/MPTPTP2078/distilledMPT_gain{gain}_0.strat --auto" {MPTPPath} MPTConstCatDistilled_gain{gain}_ --update_strat_file_suffix --test',
#     ]

# experiments_to_run = train_experiments + test_experiments + distill_experiments
# experiments_to_run = train_experiments
experiments_to_run = test_experiments
# experiments_to_run = distill_experiments

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
    parser.add_argument("--cpu_threshold", type=float, default=20.0)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    if args.dry_run:
        for exp in experiments_to_run:
            print(exp)
            print("\n\n")
    else:
        while len(experiments_to_run):
            while too_busy(args.cpu_threshold):
                
                # if user enters anything, run the next experiment
                select.select([sys.stdin], [], [], 0.0)
                if sys.stdin in select.select([sys.stdin], [], [], 0.0)[0]:
                    line = input()
                    break

                sleep(100)


            runExperiment(experiments_to_run.pop(0))
            sleep(500)

        

