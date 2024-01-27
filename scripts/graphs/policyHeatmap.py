# This script takes in as input:
#       1. a policy name ("MPTNN1" for instance)
#       2. a run name (to get trajectories for visualization)
#       3. Optionally a problem name ("MPT0001+1.p" for instance)
#          (Without a problem name, it will select one based on the following:)
#          1. The problem was solved
#          2. The proof was found in less than N processed clauses.
import argparse
import os, sys
sys.path.append(".")
from e_caller import ECallerHistory
from helpers import normalizeState
import random
import torch
import matplotlib.pyplot as plt



def selectProblem(hist, min_proc_len=1500, max_proc_len=3000, seed=0):
    probs = list(hist.history.keys())
    random.seed(seed)
    random.shuffle(probs)

    for problem in probs:
        l = len(hist.history[problem][0]['states'])
        if hist.history[problem] and hist.history[problem][0]['solved'] and l > min_proc_len and l < max_proc_len:
            print("problem: ", problem)
            return problem
    

def makePolicyHeatmap(policy, states):
    return torch.softmax(policy(normalizeState(states).float()).detach(), dim=1).T

def escapePath(s):
    return s.replace("/", "_")



if __name__ == "__main__":
    parser = argparse.ArgumentParser("Creates a heatmap of the policy's CEF preferences over the course of a proof search")
    parser.add_argument("policy", help="The policy name (i.e MPTNN1)")
    parser.add_argument("run", help="The run name (under ECallerHistory)")
    parser.add_argument("--problem", help="The problem name from the run that you want to visualize")
    parser.add_argument("--dataset", choices=["MPT","VBT","SLH"], required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    heatmap_dir = f"figures/heatmaps/{args.dataset}"
    os.makedirs(heatmap_dir, exist_ok=True)


    # 1. Get the policy:
    try:
        print("Assuming policy saved under models...")
        policy = torch.load(f"models/{args.policy}.pt")
    except:
        print("Failed...Assuming policy is a path instead...")
        policy = torch.load(args.policy)

    # 2. Get the run data:
    hist = ECallerHistory.load(args.run)

    # 3. Select problem if not provided:
    problem = args.problem if args.problem is not None else selectProblem(hist, seed=args.seed)
    if problem is None:
        print("Could not select a problem")
        exit()

    # 4. Make and save the heatmap:
    heatmap = makePolicyHeatmap(policy, hist.history[problem][0]['states'])

    plt.imshow(heatmap, aspect='auto', interpolation='none', cmap='gist_ncar', origin='lower')
    plt.colorbar()
    plt.xlabel("Time Step")
    plt.ylabel("CEF")
    # plt.title(f"Heatmap of \"{args.policy}\" on \"{problem}\"")
    plt.savefig(f"{heatmap_dir}/{escapePath(args.policy)}_{problem}.png", dpi=600, bbox_inches='tight', pad_inches=0)