import torch
import numpy as np
import sys, os
sys.path.append('.')
from e_caller import ECallerHistory
from helpers import normalizeState
import random
from glob import glob
import IPython

import seaborn as sns
import matplotlib.pyplot as plt
from rich.progress import track
from rich import print


run = 'testhist2c'
# policy = torch.load(f'models/{run}.pt')
# policy = torch.load("dummy.pt")
# p = lambda x: torch.softmax(policy(normalizeState(x.float())), dim=1)



import datetime

def formatTime(epoch_time):
    # Convert epoch time to datetime object
    return datetime.datetime.fromtimestamp(epoch_time).strftime('%Y-%m-%d %H:%M:%S')


def getCorrelations(infos, policy):
    """How are the past actions with the new action probabilities related?"""

    p = lambda x: torch.softmax(policy(normalizeState(x.float())), dim=1)

    states = []
    pastActions = []
    for info in random.choices(infos, k=40):
        s = torch.from_numpy(info['states']).reshape(-1,100)
        states.append(s)
        pastActions.append(policy.preprocess(normalizeState(s)))

    states = torch.cat(states,dim=0)
    pastActions = torch.cat(pastActions,dim=0)[:,:-5]

    return pastActions, p(states)

    IPython.embed()





def correlation_heatmap(A, B, index=0):
    """
    Takes two PyTorch tensors A and B and creates a heatmap for the correlation of the columns of A with the columns of B.
    """

    plt.figure(figsize=(20,15))

    if A.shape[0] != B.shape[0]:
        raise ValueError("A and B must have the same number of rows")

    # Convert tensors to numpy for correlation computation
    A_np = A.detach().numpy()
    B_np = B.detach().numpy()

    # Number of columns in A and B
    n_cols_A = A.shape[1]
    n_cols_B = B.shape[1]

    # Initialize the correlation matrix
    corr_matrix = np.zeros((n_cols_A, n_cols_B))

    # Compute the correlation for each pair of columns
    for i in range(n_cols_A):
        for j in range(n_cols_B):
            corr_matrix[i, j] = np.corrcoef(A_np[:, i], B_np[:, j])[0, 1]


    # Add bottom row to signify average B column values:
    avgDist = np.mean(B_np, axis=0)
    
    # remapping avgDist to fit the correlation matrix vmin/vmax
    # (0,0.08) -> (-0.4,0.4)
    remapped = (avgDist)*(0.8/0.06) - 0.4
    corr_matrix = np.concatenate((corr_matrix, remapped.reshape(1, -1)), axis=0)

    # Create a heatmap
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', annot_kws={"size": 6}, vmin=-0.4, vmax=0.4) # need to set low and high values for consistent frames
    plt.xlabel('Columns of B')
    plt.ylabel('Columns of A')
    plt.title('Correlation Matrix Heatmap')

    # Save the figure
    os.makedirs('figures/correlation_heatmaps', exist_ok=True)
    plt.savefig(f'figures/correlation_heatmaps/{index:04}.png', dpi=300, bbox_inches='tight')
    plt.clf()
    plt.close()

    return corr_matrix



# h = hists[0]
# h.merge(hists[1:])


num_hists = len(glob(f'ECallerHistory/{run}_train/*.history'))
# step = max(1, num_hists // 10)

begin = max(num_hists+1 - 10, 1)
end = num_hists+1
hists = ((n,ECallerHistory.load(f'{run}_train', num=n, keysToDeleteFromInfos=['stdout','stderr', 'actions', 'rewards'])) for n in track(list(range(begin,end))))


# All hists for one policy:

# for i,h in hists:
#     infos = sum(list(h.history.values()), [])
#     if len(infos) == 0:
#         continue

#     timestamps = [info['timestamp'] for info in infos]
#     earliest, latest = min(timestamps), max(timestamps)
#     print(f"earliest: {formatTime(earliest)}, latest: {formatTime(latest)}, diff: {latest-earliest:.2f} for {i}.history")
#     correlation_heatmap(*getCorrelations(infos), i)

# Entire policy history for the latest hists:

infos = sum([sum(list(h.history.values()), []) for _,h in hists], [])
hists = None

getNum = lambda s: int(os.path.split(s)[1].split('.')[0])
policies = [(getNum(f),torch.load(f)) for f in glob(f'model_histories/{run}/*.pt')]
policies = sorted(policies, key=lambda x: x[0])

# step = 1
step = max(1, len(policies) // 200)
policies = policies[::step]

mats = []
for i,policy in track(policies):
    mats.append(correlation_heatmap(*getCorrelations(infos, policy), i))

IPython.embed()