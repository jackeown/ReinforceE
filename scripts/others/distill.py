import torch
import argparse
import re
import sys
sys.path.append(".")
from rich import print


def getStrategy(filename):
    with open(filename) as f:
        return f.readlines()

def getCEFs(heuristic_def_line: str):
    key,val = heuristic_def_line.split(":")
    cefs = re.findall(r"([0-9]+)[.*](\w+\([^\)]*\))", val)
    cefs = [(int(weight),cef) for weight,cef in cefs]
    return cefs

def modifyCEFs(cefs, policy):
    newCEFs = []

    assert len(cefs) == len(policy), f"len(cefs)={len(cefs)} != len(policy)={len(policy)}"

    for newWeight, oldCEF in zip(policy, cefs):
        newCEFs.append([newWeight, oldCEF[1]])
    return newCEFs

def getPolicy(filename):
    policy = torch.load(filename)
    policy = torch.softmax(policy(torch.tensor([[0,0,0,0]])), dim=1)[0]
    return [int(x.item()) for x in (policy*100*args.gain).round()]


def modifyStrategy(stratLines, policy):
    newStratLines = []
    for line in stratLines:
        if "heuristic_def" not in line:
            newStratLines.append(line)
        else:
            cefs = modifyCEFs(getCEFs(line), policy)
            heuristic = ",".join([f"{cef[0]}.{cef[1]}" for cef in cefs])
            newStratLines.append(f'   heuristic_def: "({heuristic})"\n')

    return newStratLines


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("policy")
    parser.add_argument("base_strategy")
    parser.add_argument("output_strategy")
    parser.add_argument("--gain", type=float, default=5.0)
    args = parser.parse_args()

    # summary of operation being run for when it's done in a script
    print(f"python scripts/others/distill.py {args.policy} {args.base_strategy} {args.output_strategy} --gain={args.gain}")

    stratLines = getStrategy(args.base_strategy)
    policy = getPolicy(args.policy)
    newStratLines = modifyStrategy(stratLines, policy)

    with open(args.output_strategy, "w") as f:
        f.writelines(newStratLines)