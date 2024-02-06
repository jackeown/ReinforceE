# To find out when to stop training, I look back upon a training run and...
# 1. Look at which problems were solved in each "epoch" (not perfect due to some problems not giving "info"s during some runs.
import sys
sys.path.append('.')
from collections import Counter, defaultdict
from e_caller import ECallerHistory
from rich.progress import track
import IPython
import matplotlib.pyplot as plt

run = "testy3_again3_train"
hist = ECallerHistory.load(run, keysToDeleteFromInfos=['stdout', 'stderr', 'states', 'actions', 'rewards'], progress=True)


numInfos = [len(l) for k,l in hist.history.items()]
n = min(numInfos)




def getEpochTimes(hist):
    numInfos = [len(l) for k,l in hist.history.items()]
    n = max(numInfos)
    epochTimes = {}
    for i in track(range(n), description="Computing Epoch Times"):
        infoTimes = [l[i]['timestamp'] for k,l in hist.history.items() if len(l) > i]
        epochTimes[i] = min(infoTimes)
    return epochTimes

def getEpochInfos(hist, epochTimes):
    timestampToEpoch = lambda t: max([epoch for epoch,epochStart in epochTimes.items() if t > epochStart], default=-1)
    allInfos = [info for infos in hist.history.values() for info in infos]
    epochInfos = defaultdict(list)
    for info in allInfos:
        epochInfos[timestampToEpoch(info['timestamp'])].append(info)
    return epochInfos

def getProbsSolved(infos):
    return {info['problem'] for info in infos if info['solved']}

def getAvgProcCount(infos, probSet=None):
    if probSet is None:
        probSet = set([info['problem'] for info in infos if info['solved']])

    procCounts = [info['processed_count'] for info in infos if info['solved'] and info['problem'] in probSet]
    return sum(procCounts) / len(procCounts)

epochTimes = getEpochTimes(hist)
epochInfos = getEpochInfos(hist, epochTimes)



solvedEver = set()

ys = defaultdict(list)

# I want to get the problems consistently solved...
keepEpoch = lambda i: i > 0 and i < len(epochInfos) - 2
probSet = set.intersection(*[getProbsSolved(epochInfos[k]) for k in epochInfos if keepEpoch(k)])
print(len(probSet))

for k in sorted(epochInfos):
    solvedThisTime = getProbsSolved(epochInfos[k])
    avgProcCount = getAvgProcCount(epochInfos[k], probSet=probSet)
    missedProbs = solvedEver - solvedThisTime
    newProbs = solvedThisTime - solvedEver
    solvedEver = solvedThisTime | solvedEver

    print("#"*100)
    print(f"Epoch: {k}")
    print(f"Avg Processed Count: {avgProcCount}")
    print(f"Solved this time: {len(solvedThisTime)}")
    print(f"Solved Ever:      {len(solvedEver)}")
    print(f"Newly Solved:     {len(newProbs)}")
    print(f"Missed Probs:     {len(missedProbs)}")
    
    ys['avgProcCount'].append(avgProcCount)
    ys['solvedThisTime'].append(len(solvedThisTime))
    ys['solvedEver'].append(len(solvedEver))
    ys['newProbs'].append(len(newProbs))
    ys['missedProbs'].append(len(missedProbs))

for key in ys:
    plt.plot(ys[key][1:-2], label=key)

plt.legend()
plt.show()

IPython.embed()


# solvedEver = set()
# for i in range(n):
#     solvedThisTime = {prob for prob, l in hist.history.items() if l[i]['solved']}
#     missedProbs = solvedEver - solvedThisTime
#     newProbs = solvedThisTime - solvedEver
#     solvedEver = solvedThisTime | solvedEver

#     print("#"*100)
#     print(f"Solved this time: {len(solvedThisTime)}")
#     print(f"Solved Ever:      {len(solvedEver)}")
#     print(f"Newly Solved:     {len(newProbs)}")
#     print(f"Missed Probs:     {len(missedProbs)}")
#     # print(f"Number of problems solved in epochs <= {i}: {len(solvedEver)} (new: {len(newProbs)}, missed: {len(missedProbs)})")
    