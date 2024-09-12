# This file will plot the critic evaluations for a 
# given model and problem over the course of its proof attempt.
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
    return model.critic(normalizeState(info['states']).float()).detach().T



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
    parser.add_argument("--opacity", type=float, default=0.4)
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
        problems = getRandomProblems(hist, numProblems=args.numProblems, seed=args.seed, criteria=hasAtLeastNStates(400))

    
    # sort problems so all solved ones come last:
    problems.sort(key=lambda x: hist.history[x][0]['solved'], reverse=False)

    successEvals, failEvals = [],[]
    for problem in track(problems, description=f"Getting Critic Evaluations for {len(problems)} problems"):
        evals = getCriticEvaluation(model, hist, problem, seed=args.seed)
        print(evals.shape, end=' and ')
        print(evals.min(), evals.max())
        solved = hist.history[problem][0]['solved']
        plt.plot(evals[0][:max_len], label=problem, color=randomColor(solved), alpha=args.opacity)
        if solved:
            # need to pad to max_len with the last value:
            if len(evals[0]) < max_len:
                evalsPadded = torch.cat((evals[0], torch.tensor([evals[0][-1]]).repeat(max_len - len(evals[0]))))
            else:
                evalsPadded = evals[0][:max_len]
            successEvals.append(evalsPadded)
        else:
            if len(evals[0]) < max_len:
                evalsPadded = torch.cat((evals[0], torch.tensor([evals[0][-1]]).repeat(max_len - len(evals[0]))))
            else:
                evalsPadded = evals[0][:max_len]
            failEvals.append(evalsPadded)

    averageSuccessEvals = sum(successEvals) / len(successEvals)
    averageFailEvals = sum(failEvals) / len(failEvals)
    plt.plot(averageSuccessEvals, label="Solved", color="green", alpha=1)
    plt.plot(averageFailEvals, label="Unsolved", color="red", alpha=1)

    
    plt.xlabel("Clauses Processed")
    plt.ylabel("Critic Evaluation")
    plt.xlim(0, max_len)
    plt.ylim(-1000, 100)

    if args.legend:
        plt.legend()

    print(f"Saving to figures/critic/{args.dataset}/{args.run}.png")
    os.makedirs(f"figures/critic/{args.dataset}", exist_ok=True)
    plt.savefig(f"figures/critic/{args.dataset}/{args.run}.png", dpi=500)


    print("Problems Solved: ", [problem for problem in problems if hist.history[problem][0]['solved']])
    print("Problems Failed: ", [problem for problem in problems if not hist.history[problem][0]['solved']])
    
