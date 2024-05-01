#!/bin/bash

# This is just a simple script to make the tables / figures in my dissertation:

echo "Making MPT tables / figures"
~/.pyenv/shims/python scripts/results/compareSolved.py MPTAuto MPTRoundRobin MPTRoundRobinAllOnes MPTNN1Hist MPTConstCat1Hist MPTConstCatDistilled_gain5_  --cv

echo "Making MPT ConstCat heatmaps"
~/.pyenv/shims/python scripts/graphs/policyHeatmap.py MPTConstCat1Hist0 MPTConstCat1Hist0 --problem=MPT0557+1.p --dataset=MPT
~/.pyenv/shims/python scripts/graphs/policyHeatmap.py MPTConstCat1Hist1 MPTConstCat1Hist1 --problem=MPT1964+1.p --dataset=MPT

echo "Making MPT NN heatmaps"
~/.pyenv/shims/python scripts/graphs/policyHeatmap.py MPTNN1Hist0 MPTNN1Hist0 --problem=MPT1592+1.p --dataset=MPT
~/.pyenv/shims/python scripts/graphs/policyHeatmap.py MPTNN1Hist0 MPTNN1Hist0 --problem=MPT0557+1.p --dataset=MPT

echo "Making more MPT NN heatmaps"
~/.pyenv/shims/python scripts/graphs/policyHeatmap.py MPTNN1Hist1 MPTNN1Hist1 --problem=MPT1539+1.p --dataset=MPT
~/.pyenv/shims/python scripts/graphs/policyHeatmap.py MPTNN1Hist1 MPTNN1Hist1 --problem=MPT1964+1.p --dataset=MPT


echo "Making VBT tables / figures"
~/.pyenv/shims/python scripts/results/compareSolved.py VBTAuto VBTRoundRobin VBTRoundRobinAllOnes VBTNN1Hist VBTConstCat1Hist VBTConstCatDistilled_gain5_  --cv

echo "Making SLH tables / figures"
~/.pyenv/shims/python scripts/results/compareSolved.py SLHAuto SLHRoundRobin SLHRoundRobinAllOnes SLHNN1Hist SLHConstCat1Hist SLHConstCatDistilled_gain5_  --cv

echo "Finished!"

