import sys,os
sys.path.append('.')
import argparse
from helpers import normalizeState
import torch
from e_caller import ECallerHistory
import random
import matplotlib.pyplot as plt

def getPolicyPreferences(model, states):
    preferencesOverTime = []
    states = normalizeState(states).to(torch.float)
    for state in states:
        preferencesOverTime.append([x.item() for x in torch.softmax(model(state.reshape(1,-1)), dim=1).flatten()])
    return preferencesOverTime


def makeNNPlot(model, info, mode=1):
    # mode 0 is stackplots, mode 1 is a heatmap
    info["states"] = info["states"][:3000]
    policies = getPolicyPreferences(model, info['states'])
    time_series = list(zip(*policies))

    if mode == 0:
        xs = list(range(len(info['states'])))
        plt.stackplot(xs, *time_series)
        plt.xlabel("Time Step")
        plt.ylabel("CEF selection probabilities")
        plt.xlim(0,len(xs))
        plt.savefig(f"figures/newActorPlots_{info['problem']}.png")
    elif mode == 1:
        plt.figure()
        plt.imshow(time_series, aspect='auto', interpolation='none', cmap='hot', origin='lower')
        plt.xlabel("Time Step")
        plt.ylabel("CEF")
        plt.savefig(f"figures/newActorPlots_heatmap_{os.path.split(args.policy)[-1]}_{info['problem']}.png")
    else:
        raise ValueError("Invalid mode")


def makeConstCatPlot(model, info, mode=1):
    makeNNPlot(model, info, mode)



def selectionCriteria(problem, hist):
    if problem is None or len(hist.history[problem]) == 0:
        return False
    
    goldilocks = len(hist.history[problem][0]['states'])
    return goldilocks > 800 and hist.history[problem][0]['solved']






if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("policy", help="The path to the policy to inspect")
    parser.add_argument("eCallerHistory", help="The name of the ECallerHistory to use")
    parser.add_argument("--problemName", default=None)
    parser.add_argument("--constant", action="store_true")
    parser.add_argument("--mode", default=1, type=int)
    args = parser.parse_args()

    policy = torch.load(args.policy)
    hist = ECallerHistory.load(args.eCallerHistory)

    problem = args.problemName

    while not selectionCriteria(problem, hist):
        problem = random.choice(list(hist.history.keys()))
    
    infos = hist.history[problem]
    info = infos[0]

    if args.constant:
         makeConstCatPlot(policy, info, args.mode)
    else:
         makeNNPlot(policy, info, args.mode)



