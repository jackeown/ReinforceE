# This file tries to plot how much higher the critic rates problems it solves.

import argparse
import os, sys
sys.path.append(".")
from e_caller import ECallerHistory
from helpers import normalizeState
import random
import torch
import matplotlib.pyplot as plt
from rich import print
from rich.progress import track
import IPython



def getRandomProblems(hist, numProblems=10, seed=0, criteria=lambda x: True):
    random.seed(seed)
    probs = list(hist.history.keys())
    random.shuffle(probs)
    selected = []

    for p in probs:
        if len(hist.history[p]) > 0 and criteria(hist.history[p][0]):
            selected.append(p)
        if len(selected) == numProblems:
            break

    return selected


def getCriticEvaluation(model, hist, problem, seed=0):
    info = hist.history[problem][0]
    
    try:
        return model.critic(normalizeState(info['states']).float()).detach().T
    except:
        IPython.embed()



# Higher order...
def hasAtLeastNStates(n):
    return lambda info: len(info['states']) >= n



def randomColor(solved=False):
    """
    Creates a random light color if solved and a dark color if not.
    For use with matplotlib.pyplot.
    """

    low, high = (0.5,1.0) if solved else (0.0,0.5)
    # return (random.uniform(low, high), random.uniform(low, high), random.uniform(low, high))
    return 'green' if solved else 'red'


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Creates a plot of the critic evaluations over the course of a proof search")
    parser.add_argument("model", help="The model name(s) (i.e MPTNN1)")
    parser.add_argument("run", help="The run name (under ECallerHistory)")
    parser.add_argument("--problems", help="The problem names from the run that you want to visualize (comma separated)")
    parser.add_argument("--numProblems", type=int, default=30)
    parser.add_argument("--dataset", choices=["MPT","VBT","SLH"], required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--legend", action="store_true")
    parser.add_argument("--opacity", type=float, default=0.5)
    args = parser.parse_args()


    max_len = 2_000

    print(f"Loading History '{args.run}'")
    hist = ECallerHistory.load(args.run)

    print(f"Loading Model '{args.model}'")
    model = torch.load(f"models/{args.model}.pt")

    if "problem" in args:
        problems = args.problem.split(",")
    else:
        print(f"Selecting {args.numProblems} random problems")
        problems = getRandomProblems(hist, numProblems=args.numProblems, seed=args.seed, criteria=hasAtLeastNStates(max_len))

    
    solvedEvals = []
    unsolvedEvals = []
    track = lambda x, *args, **kwargs: x
    for problem in track(problems, description=f"Getting Critic Evaluations for {len(problems)} problems"):
        evals = getCriticEvaluation(model, hist, problem, seed=args.seed)[0][:max_len]
        if hist.history[problem][0]['solved']:
            solvedEvals.append(evals)
        else:
            unsolvedEvals.append(evals)
    
    averageSolvedEvals = sum(solvedEvals) / len(solvedEvals)
    averageUnsolvedEvals = sum(unsolvedEvals) / len(unsolvedEvals)
    diff = averageSolvedEvals - averageUnsolvedEvals
    plt.plot(diff, color="black", alpha=args.opacity)
    plt.plot([0,max_len],[0,0], color='grey', alpha=args.opacity)
    
    # plt.plot(averageSolvedEvals, label="Solved", color="green", alpha=args.opacity)
    # plt.plot(averageUnsolvedEvals, label="Unsolved", color="red", alpha=args.opacity)

    plt.xlabel("Clauses Processed")
    plt.ylabel("critic(solved)-critic(unsolved)")
    plt.xlim(0, max_len)
    # plt.ylim(0,0.3)

    if args.legend:
        plt.legend()

    print(f"Saving to figures/critic/{args.dataset}/{args.run}.png")
    os.makedirs(f"figures/critic/{args.dataset}", exist_ok=True)
    plt.savefig(f"figures/critic/{args.dataset}/{args.run}_solved_vs_unsolved.png", dpi=500)


    print("Problems Solved: ", [problem for problem in problems if hist.history[problem][0]['solved']])
    print("Problems Failed: ", [problem for problem in problems if not hist.history[problem][0]['solved']])
    
