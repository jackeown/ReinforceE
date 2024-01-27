import argparse
from glob import glob
import torch
from rich.progress import track
import sys, os
import time
sys.path.append('.')

# Demonstrates the correlation between the 5 cross validation fold constcat policies over the course of training.


def loadPolicy(path):
    policy = torch.load(path)
    return torch.softmax(policy(torch.tensor([[1,2,3,4,5.]])), dim=1).reshape(-1)

def loadPolicies(paths):
    # paths are to files like 00047.pt
    # sort by filename number
    paths = sorted(paths, key=lambda x: int(os.path.basename(x).split(".")[0]))

    policies = [loadPolicy(path) for path in track(paths)]
    # remove any not modified/created in the last n days:
    n = 5
    policies = [policy for path,policy in zip(paths, policies) if os.path.getmtime(path) > time.time() - n*24*60*60]
    return torch.stack(policies)

def getCorrelation(policies):
    return torch.corrcoef(policies)


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("prefix", help="a prefix for helping glob the model history files")
    args = parser.parse_args()

    folds = glob(f"model_histories/{args.prefix}*_train")
    assert len(folds) == 5, "There should be 5 folds"

    # shape: fold x time x CEF probability 
    policies = [loadPolicies(glob(f"{fold}/*.pt")) for fold in folds]

    # by zipping, each dists is a list of 5 policies at a given moment
    for dists in zip(*policies):
        # print(dists)
        # print("-"*80)
        print(getCorrelation(torch.stack(dists)))





