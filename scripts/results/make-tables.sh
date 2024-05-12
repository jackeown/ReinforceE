#!/bin/bash

# This is just a simple script to make the tables / figures in my dissertation:
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
        ${ds}NN1Hist ${ds}ConstCat1Hist ${ds}ConstCatDistilled_gain5_ > ~/Desktop/ReinforceE/latexTables/${ds}.tex
done
echo "Finished!"




echo "Making MPT ConstCat heatmaps"
~/.pyenv/shims/python scripts/graphs/policyHeatmap.py MPTConstCat1Hist0 MPTConstCat1Hist0 --problem=MPT0557+1.p --dataset=MPT
~/.pyenv/shims/python scripts/graphs/policyHeatmap.py MPTConstCat1Hist1 MPTConstCat1Hist1 --problem=MPT1964+1.p --dataset=MPT

echo "Making MPT NN heatmaps"
~/.pyenv/shims/python scripts/graphs/policyHeatmap.py MPTNN1Hist0 MPTNN1Hist0 --problem=MPT1592+1.p --dataset=MPT
~/.pyenv/shims/python scripts/graphs/policyHeatmap.py MPTNN1Hist0 MPTNN1Hist0 --problem=MPT0557+1.p --dataset=MPT

echo "Making more MPT NN heatmaps"
~/.pyenv/shims/python scripts/graphs/policyHeatmap.py MPTNN1Hist1 MPTNN1Hist1 --problem=MPT1539+1.p --dataset=MPT
~/.pyenv/shims/python scripts/graphs/policyHeatmap.py MPTNN1Hist1 MPTNN1Hist1 --problem=MPT1964+1.p --dataset=MPT

echo "Finished!";