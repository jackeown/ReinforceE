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
# NeuralNet = NN1Hist
# ConstCat = ConstCat1Hist
# Distilled = ConstCatDistilled_gain5_


# define datasets "MPT", "VBT", "SLH" in bash array for loop:
datasets=("MPT" "VBT" "SLH")

# define experiments similarly:
for ds in "${datasets[@]}"; do
    echo "Making ${dataset} tables / figures"
    ~/.pyenv/shims/python scripts/results/compareSolved.py --cv --ijait \
        ${ds}RoundRobinAllOnes ${ds}RoundRobin ${ds}SuccessRoundRobin \
        ${ds}CommonHeuristic ${ds}CommonElse \
        ${ds}Auto ${ds}AutoSched ${ds}AutoAll \
        ${ds}NN1Hist ${ds}ConstCat1Hist ${ds}ConstCatDistilled_gain5_ > ~/Desktop/ReinforceE/latexTables/${ds}_ijait.tex
done
echo "Finished!"

