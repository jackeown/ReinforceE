import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import math
import IPython
import numpy as np
from collections import deque
import sys
from helpers import mean, normalizeState, saveToDir
from rich import print as pprint

class DummyProfiler:

    @staticmethod
    def profile(s):
        return DummyContextManager()

class DummyContextManager:

    def __init__(self):
        pass

    def __enter__(self):
        pass

    def __exit__(self, a,b,c):
        pass



# This is a remapping of rewards:
# every successful proof with n steps will have returns (from the start state) equal to...
# return = 100 - n*penalty
# penalty = 1e-3:
#     1000*1e-3 is 10.
#     a proof with 1,000 steps therefore has return of 90
#     and a proof with 10,000 steps has of return = 0.
#     an unsuccessful proof with n steps would have return of -n*penalty

def adjustRewards(reward):
    return (reward*10.0 - 3e-4) / 2.0


# Gamma is the discount factor
# Lambda is the Generalized Advantage Estimation parameter
def calculateReturnsAndAdvantageEstimate(policy, episode, GAMMA=0.999, LAMBDA=0.95, lunarLander=False):

    if lunarLander:
        states = torch.from_numpy(episode.states).to(torch.float)
    else:
        states = normalizeState(torch.from_numpy(episode.states).to(torch.float))

    try:
        actions = torch.from_numpy(episode.actions).to(torch.long) # Important line (has bug?)
    except Exception as e:
        print(e)
        print(f"episode.actions: {episode.actions}")
        sys.exit(1)

    soft = torch.softmax(policy(states), dim=1)
    log_probs = torch.distributions.Categorical(soft).log_prob(actions).detach().numpy()
    values = policy.critic(states).reshape(-1).detach().numpy()    
    reward = adjustRewards(episode.rewards)

    values_shifted = np.zeros_like(values)
    values_shifted[:-1] = values[1:]

    n = len(values)

    advantage = np.zeros(n + 1)
    returns = np.zeros(n + 1)

    not_last_timestep = 0
    for t in reversed(range(n)):
        returns[t] = reward[t] + GAMMA * returns[t + 1]
        td_err = reward[t] + (GAMMA * values_shifted[t] * not_last_timestep) - values[t]
        advantage[t] = td_err + (GAMMA * LAMBDA * advantage[t + 1] * not_last_timestep)
        not_last_timestep = 1

    biased_target = advantage[:n] + np.squeeze(values)
    noisy_target = returns[:n]
    
    how_much_bias = 0.65
    # value_target = biased_target
    value_target = how_much_bias*biased_target + (1-how_much_bias)*noisy_target

    return value_target, advantage[:n], log_probs, values











def calculateReturns(policy_net, episode, off_policy, profiler, numpy=False, discount_factor=0.99):

    with profiler.profile("calculateReturns"):
        returns = deque()
        total = 0

        # returns = rewards when rewards are all 0
        if all(x == 0.0 for x in episode.rewards):
            return episode.rewards

        for s,a,r in reversed(list(zip(episode.states, episode.actions, episode.rewards))):

            importance = 1
            if off_policy:
                likelihood1 = torch.softmax(policy_net(s), dim=1)[0][a].detach().item()
                likelihood2 = torch.softmax(episode.policy_net(s), dim=1)[0][a].detach().item()
                try:
                    importance = (likelihood1 / likelihood2)
                except ZeroDivisionError:
                    importance = 1e-100
            
                if importance < 0.001 or importance > 99.9:
                    print(f"Ridiculous importance: {importance}")
                    return None

            total = importance * (r.item() + total * discount_factor)
            returns.appendleft(total)
        
        if numpy:
            returns = np.array(returns)
        else:
            returns = torch.tensor(returns)

        # Return normalization...
        if hasattr(policy_net, "critic"):
            return returns
        else:
            sys.exit()
            return (returns - returns.mean()) / (returns.std() + 1e-7)


def getGradNorm(policy_net, log=True, maxGradNorm=0.5):
    total_norm = 0.0
    for p in policy_net.parameters():
        if p.grad is not None:
            param_norm = p.grad.detach().data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    
    if log:
        print(f"Gradient Norm: {total_norm}")

    if maxGradNorm > 0:
        gradNorm = torch.nn.utils.clip_grad_norm_(policy_net.parameters(), maxGradNorm)
        if gradNorm > maxGradNorm and log:
            print(f"Gradient Clipping: (norm = {gradNorm})")

    return total_norm



def batch_generator(rollout_buffer, batch_size):
    random.shuffle(rollout_buffer)
    batch = []
    i = 0
    while i < len(rollout_buffer):
        batch.append(rollout_buffer[i])
        if len(batch) == batch_size:
            states,actions,returns,adv,log_probs,vals = list(zip(*batch))
            to_yield = [
                torch.stack(states).reshape(batch_size,-1),
                torch.stack(actions).reshape(batch_size),
                torch.stack(returns).reshape(batch_size),
                torch.stack(adv).reshape(batch_size),
                torch.stack(log_probs).reshape(batch_size),
                torch.stack(vals).reshape(batch_size),
            ]
            # print(to_yield)
            yield to_yield
            batch = []
        i += 1









def optimize_step_ppo(opt, policy_net, rollout_buffer, batch_size=128, critic_weight=0.1, entropy_weight=0.001, max_grad_norm=0.5, epochs=10):

    debugging = []
    saved_info = []
    for _ in range(epochs):
        for S, A, VAL_TARGET, ADV, old_log_probs, old_values in batch_generator(rollout_buffer, batch_size):
            # Get action log probabilities.
            dist = torch.distributions.Categorical(torch.softmax(policy_net(S),dim=1))
            log_probs = dist.log_prob(A)

            # ratio between old and new policy (should be one at the first iteration)
            ratio = torch.exp(log_probs - old_log_probs)


            # Actor Objective/Loss
            actor_objective_unclipped = ADV * ratio
            actor_objective_clipped = ADV * torch.clamp(ratio, 0.85, 1.15)
            actor_loss = -torch.min(actor_objective_unclipped, actor_objective_clipped).mean()


            # Critic Loss
            values = policy_net.critic(S).reshape(-1)
            critic_loss = critic_weight * F.huber_loss(values, VAL_TARGET).mean()
            

            # Entropy Loss and Total Loss
            entropy_loss = -entropy_weight * dist.entropy().mean()


            # Compute L1 loss component
            l1_weight = 1e-4
            l1_parameters = []
            for parameter in policy_net.parameters():
                l1_parameters.append(parameter.view(-1))
            l1_loss = l1_weight * torch.abs(torch.cat(l1_parameters)).mean()

            # Total Loss
            loss = actor_loss + critic_loss + entropy_loss + l1_loss

            # Perform Gradient Step
            opt.zero_grad()
            loss.backward()
            gradNorm = getGradNorm(policy_net, log=False, maxGradNorm=max_grad_norm)
            opt.step()


            debugging.append({
                "values": values.detach(),
                "values_target": VAL_TARGET.detach(),
                "dist": dist,
                "log_probs": log_probs.detach(),
                "old_log_probs": old_log_probs.detach(),
                "adv": ADV.detach(),
                "ratio": ratio.detach(),
                "actor_objective_unclipped": actor_objective_unclipped.detach(),
                "actor_objective_clipped": actor_objective_clipped.detach(),
                "actor_loss": actor_loss.detach(),
                "critic_loss": critic_loss.detach(),
                "entropy_loss": entropy_loss.detach(),
                "loss": loss.detach(),
            })


            # Logging Info for Dashboard/ECallerHistory
            saved_info.append([
                loss.item(),actor_loss.item(),critic_loss.item(),entropy_loss.item(), 
                gradNorm, ADV.mean().item()
            ])

    saveToDir("debugging", debugging, "lossStuff")

    # Return mean of each loss/metric as a dictionary
    return {k:mean(v) for k,v in zip(
        ["total","actor","critic","entropy", "grad_norm", "advantage"],
        zip(*saved_info)
    )}



def optimize_step(opt, policy_net, states, actions, returns, profiler=None, batch_size=64, critic_weight=0.1, entropy_weight=0.001, max_grad_norm=0.5):
    opt.zero_grad()

    if profiler is None:
        profiler = DummyProfiler
    
    with profiler.profile("forward pass"):
        dist = torch.distributions.Categorical(torch.softmax(policy_net(states), dim=1))
        entropy = dist.entropy()

        if hasattr(policy_net, "critic"):
            values = policy_net.critic(states).reshape(-1)
            advantage = returns - values

            critic_loss = F.mse_loss(values, returns, reduction="mean")
            # critic_loss = F.smooth_l1_loss(values, returns, reduction="mean")
            actor_loss = (-dist.log_prob(actions) * advantage.detach()).mean()
            entropy_loss = (-entropy).mean()

            critic_loss *= critic_weight
            entropy_loss *= entropy_weight

            loss = actor_loss + critic_loss + entropy_loss
        else:
            loss = -dist.log_prob(actions) * (returns)
            loss = loss.sum()

    with profiler.profile("backward pass"):
        loss.backward()
        gradNorm = getGradNorm(policy_net, log=False, maxGradNorm=max_grad_norm)
        opt.step()
    
    return {
        'total': loss.item(),
        'actor': actor_loss.item() if hasattr(policy_net, "critic") else loss.item(), 
        'critic': critic_loss.item() if hasattr(policy_net, "critic") else None, 
        'entropy': entropy_loss.item() if hasattr(policy_net, "critic") else entropy.item(),
        'grad_norm': gradNorm,
        'episode_length': len(actions),
        'solved': int(any(returns > 1e-6)),
        'advantage': advantage.mean().item()
    }
    




def select_action(policy_net, state):

    # if the number of given clause selections is crazy high,
    # we have little training data and it may be better to simply act randomly.
    # if state[0][0] > 2.0:
    #     return torch.randint(0,state.shape[1], (state.shape[0],1))

    out = policy_net(state)
    dist = torch.softmax(out, dim=1)
    return torch.distributions.Categorical(dist).sample().view(-1,1)




class Critic(nn.Module):

    def __init__(self, inDim, midDim, outDim):
        super().__init__()
        self.l1 = nn.Linear(inDim, midDim)
        self.l2 = nn.Linear(midDim, midDim)
        self.l3 = nn.Linear(midDim, outDim)

    def forward(self, x):
        a1 = F.relu(self.l1(x))
        a2 = F.relu(self.l2(a1))
        # return self.l3(a2)
        # return torch.sigmoid(self.l3(a2))
        return self.l3(a2)


class PolicyNetAttn(nn.Module):
    def __init__(self, stateDim, outDim, numDists):
        super().__init__()
        self.default = nn.Parameter(torch.randn(outDim))
        self.defaultWeight = nn.Parameter(torch.tensor(0.5))
        self.keys = nn.Parameter(torch.randn(numDists, stateDim))
        self.values = nn.Parameter(torch.randn(numDists, outDim))
        

    def forward(self, states):
        affinities = torch.softmax(states @ self.keys.T, 1)
        queryResult = (affinities.T * self.values).sum(dim=0).reshape(1,-1)
        weightedQueryResult = (1.0 - self.defaultWeight) * queryResult
        weightedDefault = self.defaultWeight * self.default
        return weightedQueryResult + weightedDefault


class PolicyNetUniform(nn.Module):
    def __init__(self, outDim):
        super().__init__()
        self.x = nn.Parameter(torch.ones(1,1))
        self.outDim = outDim
        
    def forward(self, states):
        return torch.ones(states.shape[0], self.outDim) * self.x


class PolicyNetVeryBad(nn.Module):
    """Confidently predicts class one always..."""
    def __init__(self, outDim):
        super().__init__()
        self.x = nn.Parameter(torch.ones(1,1))
        self.outDim = outDim
        
    def forward(self, states):
        stupid_one = (torch.sigmoid(self.x) / 1000) + 1
        out = torch.zeros(states.shape[0], self.outDim)
        out[:,0] = 100
        return out * stupid_one


class PolicyNetConstCategorical(nn.Module):
    def __init__(self, inDim, outDim):
        super().__init__()
        self.dist = nn.Parameter(torch.ones(outDim))
        self.critic = Critic(inDim, 100, 1)
        
    def forward(self, states):
        return self.dist.repeat(states.shape[0], 1)*50.0
    

class PolicyNetRoundRobin(nn.Module):
    def __init__(self, inDim, outDim):
        super().__init__()
        self.n = 0
        self.critic = Critic(inDim, 100, 1)
        self.outDim = outDim
        
    def forward(self, states):
        self.n += 1
        chosen = self.n % self.outDim
        return torch.tensor([100. if i == chosen else 0. for i in range(self.outDim)]).reshape(1,-1)



class PolicyNet(nn.Module):

    def __init__(self, inDim, midDim, outDim, numHidden, dropoutProb=0.03):
        super().__init__()

        # only the history of actions as a sum of scaled one-hots: see preprocess method
        self.stateSize = inDim
        self.numActions = outDim

        newIndim = inDim if inDim <= 5 else 5 + self.numActions + ((inDim - 5) // 5)*4
        # newIndim = 5 + outDim

        self.inputDropout = nn.Dropout(dropoutProb)
        self.input = nn.Linear(newIndim, midDim)
        self.normalizations = nn.ModuleList([nn.LayerNorm(midDim) for _ in range(numHidden)])
        self.hidden = nn.ModuleList([nn.Linear(midDim, midDim) for _ in range(numHidden)])
        self.dropouts = nn.ModuleList([nn.Dropout(dropoutProb) for _ in range(numHidden)])
        self.output = nn.Linear(midDim, outDim)

        self.hook = nn.Linear(newIndim, outDim)

        self.critic = None
        # self.c = Critic(newIndim, 100, 1)
        # self.critic = lambda s, x: c(s.preprocess(x))

    # def critic(self, x):
    #     if x.shape[-1] > 5:
    #         x = self.preprocess(x)
    #     return self.c(x)
        

    def preprocess(self, s, verbose=False):
        reshaped = (len(s.shape) == 1)
        if reshaped:
            state = s.reshape(1,-1)
        else:
            state = s

        # the actions are at index 5n + 4 and the last 5 features are the *current* state
        actionFeatures = state[:, 4:-5:5]

        actionIndices = (actionFeatures * 40).long()

        # Mask for valid actions (action index should not be -1)
        valid_actions_mask = (actionIndices != -1)

        # Temporarily replace -1 with 0 for one-hot encoding
        actionIndices[actionIndices == -1] = 0

        # 1.) Get one-hot vectors, masked where actionIndices are -1
        oneHots = F.one_hot(actionIndices, num_classes=self.numActions)
        oneHots[~valid_actions_mask] = 0

        # 2.) Sum one-hot vectors with weights 1/(i^2)
        weights = 1 / torch.pow(torch.arange(oneHots.shape[1], 0, -1), 2).view(1, -1, 1).to(state.device)
        oneHotsWeighted = oneHots * weights
        oneHotsSummed = oneHotsWeighted.sum(dim=1)

        # 3.) Add back actual current state (last 5 features)
        current_state = state[:, -5:]
        # pSizes = state[:, 0:-5:5]
        # uSizes = state[:, 1:-5:5]
        # pWeights = state[:, 2:-5:5]
        # uWeights = state[:, 3:-5:5]
        # newState = torch.cat([oneHotsSummed, current_state, pSizes, uSizes, pWeights, uWeights], dim=1)

        # 3.) Add back actual current state (last 5 features)
        # More efficient approach maybe?:
        histMinusActions = state.view(state.shape[0],-1,5)[:,:-1,:-1] # first -1 means don't include current state, and second -1 means don't include actions
        newState = torch.cat([oneHotsSummed, histMinusActions.reshape(state.shape[0],-1), current_state], dim=1)

        if verbose:
            pprint("Action Features", actionFeatures)
            pprint("Action Indices", actionIndices)
            pprint("Valid actions mask", valid_actions_mask)
            pprint("One Hots", oneHots)
            pprint("One Hots Summed", oneHotsSummed)
            pprint("New State", newState)

        return newState[0] if reshaped else newState



    def forward(self, states):

        if states.shape[-1] > 5:
            states = self.preprocess(states)

        # print(states)

        first = self.input(self.inputDropout(states))
        # first = self.input(states)
        x = first
        for layer, normalization, drop in zip(self.hidden, self.normalizations, self.dropouts):
        # for layer, normalization in zip(self.hidden, self.normalizations):
            x = normalization(layer(x))
            x = F.relu(x)
            x = drop(x)
            # x = F.leaky_relu(x, self.sqrt5) + first

        action_dist = self.output(x) + self.hook(states)
        return action_dist
















############################################
#     DQN Stuff below here...              #
############################################





def calculate_returns_dqn(policy, episode, GAMMA=0.999, LAMBDA=0.95, lunarLander=False):
    with torch.no_grad():
        # states = normalizeState(torch.from_numpy(episode.states).to(torch.float))
        # stateVals = policy(states)

        reward = adjustRewards(episode.rewards)
        n = len(reward)
        noisyReturns = np.zeros(n+1)
        # biasedReturns = np.zeros(n+1)

        for t in reversed(range(n)):
            noisyReturns[t] = reward[t] + GAMMA * noisyReturns[t + 1]

            # futureVal = stateVals[t+1] if t < n - 1 else 0
            # biasedReturns[t] = reward[t] + GAMMA * futureVal

        bias = 0.97
        # returns = noisyReturns[:n] * (1 - bias) + bias * biasedReturns[:n]
        returns = noisyReturns[:n]

        # return returns, returns, returns, returns # to keep compatibility with how it's called in main.py
        return reward, returns, reward, reward # rewards are needed, not returns for TD learning...





def batch_generator_dqn(rollout_buffer, batch_size):
    # random.shuffle(rollout_buffer)

    SHUFF = list(range(len(rollout_buffer)))
    random.shuffle(SHUFF)

    batch = []
    i = 0
    while i < len(rollout_buffer):
        batch.append(rollout_buffer[SHUFF[i]])
        if len(batch) == batch_size:
            states,actions,returns,adv,log_probs,vals,next_states = list(zip(*batch))
            to_yield = [
                torch.stack(states).reshape(batch_size,-1),
                torch.stack(actions).reshape(batch_size),
                torch.stack(returns).reshape(batch_size),
                torch.stack(adv).reshape(batch_size),
                torch.stack(log_probs).reshape(batch_size),
                torch.stack(vals).reshape(batch_size),
                torch.stack(next_states).reshape(batch_size,-1)
            ]
            # print(to_yield)
            yield to_yield
            batch = []
        i += 1



# from ChatGPT
def soft_update(target_net, online_net, tau=0.001):
    for target_param, online_param in zip(target_net.parameters(), online_net.parameters()):
        target_param.data.copy_(tau * online_param.data + (1.0 - tau) * target_param.data)


def optimize_step_dqn(opt, policy_net, rollout_buffer, batch_size=128, critic_weight=0.1, entropy_weight=0.001, max_grad_norm=0.5, epochs=10):
    
    debugging = []
    saved_info = []
    for _ in range(epochs):
        for i, (S, A, R, Returns,_,_,S_next) in enumerate(batch_generator_dqn(rollout_buffer, batch_size)):

            Q_vals = policy_net(S).gather(1, A.unsqueeze(1)).squeeze(1)
            
            # if S_next[i] is dummy, then Q_vals_next[i] should be 0
            dummies = ((S_next - 3.14159).abs() < 0.001).all(dim=1)

            Q_vals_next = torch.zeros_like(Q_vals)
            actions_next = policy_net(S_next[~dummies]).max(1)[1] # argmax actions of policy
            Q_vals_next[~dummies] = policy_net(S_next[~dummies], q2=True).gather(1, actions_next.unsqueeze(1)).squeeze(1) # values of those actions

            target = R + 0.998 * Q_vals_next
            bias = 0.975
            target = Returns * (1-bias) + target*bias
            loss = F.smooth_l1_loss(Q_vals, target.detach())

            opt.zero_grad()
            loss.backward()
            norm = getGradNorm(policy_net, log=False, maxGradNorm=max_grad_norm)
            if norm > max_grad_norm:
                print(f"Gradient Clipping: (norm = {norm})")
            opt.step()

            # Logging Info for Dashboard/ECallerHistory
            saved_info.append([
                loss.item(), loss.item(), loss.item(), loss.item(), 
                0.1337, 0.1337
            ])

            # Swap Q networks
            if i % 80 == 0:
                soft_update(policy_net.q2, policy_net.q)
            # elif i % 100 == 99:
            #     policy_net.q, policy_net.q2 = policy_net.q2, policy_net.q

    # Return mean of each loss/metric as a dictionary
    return {k:mean(v) for k,v in zip(
        ["total","actor","critic","entropy", "grad_norm", "advantage"],
        zip(*saved_info)
    )}



def nthRoot(n,x):
    return x ** (1/n)

def select_action_dqn(policy_net, state, epsilon=0.064):
    
    # x such that 0.5*x^100000 = 0.05
    # x = nthRoot(100000,0.05/0.5) = nthRoot(100000,0.1) 
    # select_action_dqn.epsilon *= nthRoot(1_000_000, 0.05 / 0.5)

    with torch.no_grad():
        # randomly sample actions sometimes
        if random.random() < epsilon:
            return torch.tensor(random.randrange(policy_net.numActions))

        # distributional
        Qs = policy_net(state).flatten()
        # Qs2 = policy_net(state, q2=True).flatten()
        gain = 48
        soft = F.softmax(Qs*gain, dim=-1)
        dist = torch.distributions.Categorical(probs=soft)
        action = dist.sample()
        largestProb = soft.max().item()

        if random.random() < 0.0002:
            print(f"largestProb: {largestProb}")
            print(f"probs (min, mean, max): {soft.min().item():.2}, {soft.mean().item():.2}, {largestProb:.2}")
            print(f"Qs (min, mean, max): {Qs.min().item():.2}, {Qs.mean().item():.2}, {Qs.max().item():.2}")
            Qs2 = policy_net(state, q2=True).flatten()
            diff = (Qs - Qs2).abs()
            print(f"Diff (min, mean, max): {diff.min().item():.2}, {diff.mean().item():.2}, {diff.max().item():.2}")

        return action



class FF(nn.Module):

    def __init__(self, inDim, midDim, outDim, n_layers):
        super().__init__()
        self.input = nn.Linear(inDim, midDim)
        self.hidden = nn.ModuleList([nn.Linear(midDim, midDim) for _ in range(n_layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(midDim) for _ in range(n_layers)])
        self.output = nn.Linear(midDim, outDim)
        self.hook = nn.Linear(inDim, outDim)
    
    def forward(self, x):
        begin = x
        x = F.relu(self.input(x))
        for layer, norm in zip(self.hidden, self.norms):
            x = norm(layer(x))
            x = F.relu(x)
        return self.output(x) + self.hook(begin)
    

# From ChatGPT
class DuelingFF(nn.Module):
    def __init__(self, inDim, midDim, outDim, n_layers):
        super().__init__()
        self.input = nn.Linear(inDim, midDim)
        self.hidden = nn.ModuleList([nn.Linear(midDim, midDim) for _ in range(n_layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(midDim) for _ in range(n_layers)])

        # Output layers for state value and advantages
        self.value_output = nn.Linear(midDim, 1)  # State value V(s)
        self.value_hook = nn.Linear(inDim, 1)

        self.adv_output = nn.Linear(midDim, outDim)  # Advantages A(s, a)
        self.adv_hook = nn.Linear(inDim, outDim)  # Keep your original hook

    def forward(self, x):
        begin = x
        x = F.relu(self.input(x))
        for layer, norm in zip(self.hidden, self.norms):
            x = norm(layer(x))
            x = F.relu(x)

        # Compute state value and advantages
        value = self.value_output(x) + self.value_hook(begin)
        adv = self.adv_output(x) + self.adv_hook(begin)

        # Combine using the dueling network formula: Q = V + (A - mean(A))
        adv_mean = adv.mean(1, keepdim=True)  # Compute mean advantage for subtracting
        q = value + (adv - adv_mean)

        return q


class PolicyNetDQN(nn.Module):

    def __init__(self, inDim, midDim, outDim, numHidden, dropoutProb=0.01):
        super().__init__()

        # only the history of actions as a sum of scaled one-hots: see preprocess method
        self.stateSize = inDim
        self.numActions = outDim

        # TODO: toggle for theorem proving (not lunar lander)
        newIndim = inDim if inDim <= 5 else 5 + self.numActions + ((inDim - 5) // 5)*4
        # newIndim = inDim

        # self.q = FF(newIndim, midDim, outDim, numHidden)
        # self.q2 = FF(newIndim, midDim, outDim, numHidden)

        self.q = DuelingFF(newIndim, midDim, outDim, numHidden)
        self.q2 = DuelingFF(newIndim, midDim, outDim, numHidden)



    def preprocess(self, s, verbose=False):
        reshaped = (len(s.shape) == 1)
        if reshaped:
            state = s.reshape(1,-1)
        else:
            state = s

        # the actions are at index 5n + 4 and the last 5 features are the *current* state
        actionFeatures = state[:, 4:-5:5]

        actionIndices = (actionFeatures * 40).long()

        # Mask for valid actions (action index should not be -1)
        valid_actions_mask = (actionIndices != -1)

        # Temporarily replace -1 with 0 for one-hot encoding
        actionIndices[actionIndices == -1] = 0

        # 1.) Get one-hot vectors, masked where actionIndices are -1
        oneHots = F.one_hot(actionIndices, num_classes=self.numActions)
        oneHots[~valid_actions_mask] = 0

        # 2.) Sum one-hot vectors with weights 1/(i^2)
        weights = 1 / torch.pow(torch.arange(oneHots.shape[1], 0, -1), 2).view(1, -1, 1).to(state.device)
        oneHotsWeighted = oneHots * weights
        oneHotsSummed = oneHotsWeighted.sum(dim=1)

        # 3.) Add back actual current state (last 5 features)
        current_state = state[:, -5:]
        # pSizes = state[:, 0:-5:5]
        # uSizes = state[:, 1:-5:5]
        # pWeights = state[:, 2:-5:5]
        # uWeights = state[:, 3:-5:5]
        # newState = torch.cat([oneHotsSummed, current_state, pSizes, uSizes, pWeights, uWeights], dim=1)

        # 3.) Add back actual current state (last 5 features)
        # More efficient approach maybe?:
        histMinusActions = state.view(state.shape[0],-1,5)[:,:-1,:-1] # first -1 means don't include current state, and second -1 means don't include actions
        newState = torch.cat([oneHotsSummed, histMinusActions.reshape(state.shape[0],-1), current_state], dim=1)

        if verbose:
            pprint("Action Features", actionFeatures)
            pprint("Action Indices", actionIndices)
            pprint("Valid actions mask", valid_actions_mask)
            pprint("One Hots", oneHots)
            pprint("One Hots Summed", oneHotsSummed)
            pprint("New State", newState)

        return newState[0] if reshaped else newState

    def forward(self, states, q2=False):

        # TODO: enable for theorem proving (not lunar lander)
        if states.shape[-1] > 5:
            states = self.preprocess(states)

        return self.q2(states) if q2 else self.q(states)



