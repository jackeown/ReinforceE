#!/bin/bash

# This is just a simple script to make the tables / figures in my IWIL 2024 paper:
#########################
# Experiments included: #
#########################
# MasterAllOnes = RoundRobinAllOnes
# MasterWeighted = RoundRobin
# MasterWeightedRR = MasterRR
# MasterSuccess = SuccessRoundRobin
# CommonHeuristic = CommonHeuristic
# CommonElse = CommonElse
# Auto = Auto
# AutoSched = AutoSched
# AutoAll = StratOmnipotent


# define datasets "MPT", "VBT", "SLH" in bash array for loop:
datasets=("MPT" "VBT" "SLH")

# define experiments similarly:
for ds in "${datasets[@]}"; do
    echo "Making ${dataset} tables / figures"
    ~/.pyenv/shims/python scripts/results/compareSolved.py --cv --ijait \
        ${ds}RoundRobinAllOnes ${ds}RoundRobin ${ds}SuccessRoundRobin \
        ${ds}MasterRR \
        ${ds}CommonHeuristic ${ds}CommonElse \
        ${ds}Auto ${ds}AutoSched ${ds}AutoAll > ~/Desktop/ReinforceE/latexTables/${ds}_ijait.tex
done
echo "Finished!"

