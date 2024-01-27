# I'm just paranoid maybe...(I believe) I saw some old_log_probs were very low (-15)
# This indicates that the sampled actions were very low probability USUALLY!!!
# That doesn't make sense...

# To debug this, I'm saving a bunch of info in ./debugging/lossStuff_*

import torch
from glob import glob
import os
from rich import print
from random import choice

def toProbs(x, round=False):
    if round:
        return (torch.exp(torch.tensor(x))*100).round()
    else:
        return torch.exp(torch.tensor(x))

def mean(l):
    return sum(l) / len(l)


if __name__ == "__main__":

    stuffs = sorted(glob("debugging/lossStuff_*"))
    if len(stuffs) == 0:
        print("No stuffs found")
        exit(1)
    
    early = torch.load(stuffs[0])
    mid = torch.load(stuffs[len(stuffs)//2])
    late = torch.load(stuffs[-1])

    print("This is all from randomly sampled batches...")

    # early = choice(early)
    # mid = choice(mid)
    # late = choice(late)

    n = 80

    for name,x in zip(["early", "mid", "late"], [early, mid, late]):
        avgAdv = mean([x['adv'].mean() for x in x])
        avgOldProbs = mean([toProbs(x['old_log_probs']).mean() for x in x])
        avgProbs = mean([toProbs(x['log_probs']).mean() for x in x])

        print(f"== {name} ==")
        print(f"{name} avgAdv: {avgAdv:.4f}")
        print(f"{name} avgOldProbs: {avgOldProbs:.4f}")
        print(f"{name} avgProbs: {avgProbs:.4f}")
        print("#"*n)


        # print(name)
        # print("Avg Advantage: ", x['adv'].mean())
        # print("old_probs: ", toProbs(x['old_log_probs']))
        # print("Average: ", toProbs(x['old_log_probs']).mean())
        # print("probs: ", toProbs(x['log_probs']))
        # print("Average: ", toProbs(x['log_probs']).mean())
        # print("#"*n)


