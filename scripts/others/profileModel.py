import sys
sys.path.append('.')
import random
import torch
from policy_grad import PolicyNet, optimize_step_ppo, calculateReturnsAndAdvantageEstimate, select_action
from my_profiler import Profiler
from helpers import Episode
import IPython
from rich.progress import track
import numpy as np

NUM_ACTIONS = 48
MAX_EPISODE_LENGTH = 7000

def getFakeState(dim):
    X = torch.randn(1, dim).clamp(0,NUM_ACTIONS/50.0)
    return X

def getFakeEpisode(stateDim, numActions, n=None, success=None, policy=None, profiler=None):
    n = random.randint(2,20) if n is None else n
    states = [getFakeState(stateDim) for _ in range(n)]
    
    if policy is None:
        actions = [random.randint(0,numActions-1) for _ in range(n)]
    else:
        actions = []
        for s in states:
            if profiler is not None:
                with profiler.profile("select_action"):
                    actions.append(select_action(policy, s).item())
            else:
                actions.append(select_action(policy, s).item())

    success = random.randint(0,1) if success is None else success
    partOfProof = lambda p: random.random() < p
    rewards = [1 if partOfProof(0.1) else 0 for _ in range(n)] if success else [0 for _ in range(n)]
    return states,actions,rewards


def profileStuff(stateDim, numActions, nTrials=5):
    profiler = Profiler()
    policy = PolicyNet(stateDim, 100, numActions, 2)
    opt = torch.optim.Adam(policy.parameters(), lr = 0.0001)

    for i in track(range(nTrials)):
        with profiler.profile("getFakeEpisode"):
            states, actions, rewards = getFakeEpisode(stateDim, numActions, n=MAX_EPISODE_LENGTH, success=True, policy=policy, profiler=profiler)
        
        states = np.concatenate([s.numpy() for s in states])
        actions = np.array(actions)
        rewards = np.array(rewards)
        # IPython.embed()

        episode = Episode(f'prob {i}', states, actions, rewards, policy)
        with profiler.profile("calculateReturnsAndAdvantageEstimate"):
            value_target, advantage, log_probs, values = calculateReturnsAndAdvantageEstimate(policy, episode)

        with profiler.profile("optimize_step_ppo"):
            rollout_buffer = [states, actions, rewards, advantage, log_probs, values]

            lossStuff = optimize_step_ppo(opt, policy, 
                                        rollout_buffer=rollout_buffer,
                                        batch_size=128, 
                                        critic_weight=0.1, 
                                        entropy_weight=0.001, 
                                        max_grad_norm=0.5, 
                                        epochs=1)

    return profiler


if __name__ == "__main__":

    # Profile the following for various state sizes:
    # Forward pass, calculateReturnsAndAdvantageEstimate, optimize_step_ppo, select_action
    
    result1 = profileStuff(5, NUM_ACTIONS, nTrials=100)
    result20 = profileStuff(100, NUM_ACTIONS, nTrials=100)

    result1.report()
    result20.report()

    IPython.embed()

