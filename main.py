
from collections import defaultdict, deque
import os, re, struct, sys, shutil
import argparse
import subprocess, threading
import functools
import random
# import multiprocessing as mp
import torch.multiprocessing as mp 
import subprocess
from time import time, sleep
from glob import glob
import resource
from typing import Iterable, List

import torch
import numpy as np

print_dumb = print
from rich import print
from rich.progress import track
import rich_dashboard

from datetime import datetime
import IPython


from my_profiler import Profiler
from helpers import Episode, mean, initPipe, normalizeState, start_logging_process, saveToDir
from e_caller import ECallerHistory, extractStatus, extractProcessedCount, extractOverhead, getStates, getActions, getRewards, getGivenClauses
from policy_grad import PolicyNet, PolicyNetDQN, PolicyNetConstCategorical, PolicyNetUniform, PolicyNetAttn, DummyProfiler
# from policy_grad import optimize_step_ppo, select_action, calculateReturnsAndAdvantageEstimate
from policy_grad import optimize_step_dqn as optimize_step_ppo, \
                        select_action_dqn as select_action, \
                        calculate_returns_dqn as calculateReturnsAndAdvantageEstimate



dummyProfiler = DummyProfiler()

def clonePolicy(module):
    if module is None:
        return None
    
    copy,_ = getPolicy() # get a new instance
    copy.load_state_dict(module.state_dict()) # copy weights and stuff
    return copy

def createInfoFromE(stdout, stderr, prob, t1, t2, policy):

    info = {}

    info['policy'] = policy.state_dict() if policy is not None else None
    info["status"] = extractStatus(stdout)
    if info['status'] == "No SZS Status":
        print(f"No szs status? sus: {prob}")
        print_dumb("stdout: ",stdout[:1000] + "..."*100 + stdout[-1000:])
        print_dumb("stderr: ",stderr)


    if "Segmentation fault" in stderr:
        print(f"SEGFAULT for {prob}")


    info["solved"] = ("Unsatisfiable" in info["status"]) or ("Theorem" in info["status"])

    info['states'] = np.array(getStates(stdout))
    info['actions'] = np.array(getActions(stdout))
    # print("Getting rewards...")
    info['rewards'] = np.array(getRewards(stdout, len(info['states'])))
    # print("Finished getting rewards")

    # info['configName'] = getConfigName(stderr, yell=False)
    # info['cefs'] = getConfig(stdout, stderr, info['configName'], yell=False)



    try:
        info['args'] = args
    except NameError:
        info['args'] = None


    info["time"] = t2 - t1
    info["timestamp"] = t1
    info["processed_count"] = len(getGivenClauses(stdout))
    info["processed_count_reported"] = extractProcessedCount(stdout) if info["solved"] else float('inf')
    info["problem"] = prob
    info["probId"] = int(prob[-8:-4])
    info["statePipeTime"], info["actionPipeTime"], info["rewardPipeTime"], info["prepTime"] = extractOverhead(stdout)


    
    limit = 5_000
    # limit = 100_000_000
    info['stdout'] = "\n".join(stdout.split("\n")[-limit:])
    info['stderr'] = "\n".join(stderr.split("\n")[-limit:])

    return info


###############################################################################
#                                                                             #
#                              PIPE HELPERS                                   #
#                                                                             #
###############################################################################


# State history is like this:
# (|p|,|u|,pweight,uweight,action)
# (|p|,|u|,pweight,uweight,action)
# (|p|,|u|,pweight,uweight,action)
# ...
# (|p|,|u|,pweight,uweight,t)
# num_feats = HISTORY_SIZE*5

def recvState(StatePipe, sync_num, state_dim):
    numStates = (state_dim / 5)
    assert numStates == int(numStates)
    numStates = int(numStates)

    syncNumSize = 4 # an int
    
    # (|p|,|u|,pweight,uweight,action OR |p|,|u|,pweight,uweight,t)
    histStateSize = (numStates-1)*(8*2 + 3*4) # 2 size_t, 2 floats, and one int (action)
    currStateSize = 8*2 + 2*4 + 8 # 2 size_t, 2 floats, and a size_t (t)

    stateSize = syncNumSize + histStateSize + currStateSize

    bytez = b''
    while len(bytez) < stateSize:
        new = os.read(StatePipe, stateSize-len(bytez))
        if len(new) == 0:
            return None
        bytez += new

    formatString = "=i" + "qqffi" * (numStates-1) + "qqffq"
    stuff = struct.unpack(formatString, bytez)
    sync_num_remote = stuff[0]
    assert sync_num_remote == sync_num, f"{sync_num_remote} != {sync_num}"

    state = torch.tensor(stuff[1:]).reshape(1, -1)
    state = normalizeState(state)

    # [sync_num_remote, ever_processed, processed, unprocessed, processed_weight, unprocessed_weight] = struct.unpack("=iqqqff", bytez)
    # assert (sync_num_remote == sync_num), f"{sync_num_remote} != {sync_num}"

    # state = torch.tensor([ever_processed, processed, unprocessed, processed_weight, unprocessed_weight]).reshape(1, -1)
    # state = normalizeState(state)

    # saveToDir("debugging", state, "state")

    return state

def sendAction(action, pipe, sync_num):
    bytes_written = 0
    to_write = struct.pack("=ii", sync_num, action.item())
    while bytes_written < len(to_write):
        bytes_written += os.write(pipe, to_write[bytes_written:])

def recvReward(pipe, sync_num):
    rewardSize = 4 + 4
    bytez = b''
    while len(bytez) < rewardSize:
        new = os.read(pipe, rewardSize - len(bytez))
        if len(new) == 0:
            return None
        bytez += new

    [sync_num_remote, reward] = struct.unpack("=if", bytez)
    assert sync_num_remote == sync_num, f"{sync_num_remote} != {sync_num}"

    assert reward in [0.0, 1.0]
    reward = 1.0 if reward == 1.0 else 0.0
    return torch.tensor(reward).reshape(1)


###############################################################################
#                                                                             #
#                             END PIPE HELPERS                                #
#                                                                             #
###############################################################################






def communicateWithE(policy, workerId, stateDim):
    sync_num = 0

    StatePipe = initPipe(f"/tmp/StatePipe{workerId}", send=False, log=False)
    ActionPipe = initPipe(f"/tmp/ActionPipe{workerId}", send=True, log=False)
    RewardPipe = initPipe(f"/tmp/RewardPipe{workerId}", send=False, log=False)

    while True:
        try:
            state = recvState(StatePipe, sync_num, stateDim)

            if state is None:
                os.close(StatePipe)
                os.close(ActionPipe)
                os.close(RewardPipe)
                return

            action = select_action(policy, state)
            sendAction(action, ActionPipe, sync_num)
            recvReward(RewardPipe, sync_num)

            sync_num += 1

        except OSError as e:
            print("OSError. Probably pipe closed")
            break

    # Clean up leftover pipes:
    os.close(StatePipe)
    os.close(ActionPipe)
    os.close(RewardPipe)



def ensureProcessDeath(p):
    try:
        p.wait(timeout=1)  # where some_timeout is an appropriate duration in seconds
    except subprocess.TimeoutExpired:
        p.terminate()  # Try to terminate the process gracefully
        time.sleep(2)  # Give it some time to terminate
        if p.poll() is None:  # Check if the process has really terminated
            p.kill()  # Kill it if it hasn't

# For reading E's stdout/stderr while we interact with E in another thread
def read_output(process, result):
    stdout, stderr = [], []

    def gather_output(pipe):
        for line in iter(pipe.readline, b''):
            yield line

    for line in gather_output(process.stdout):
        stdout.append(line.decode())
    for line in gather_output(process.stderr):
        stderr.append(line.decode())
    ensureProcessDeath(process)

    result.append(("".join(stdout), "".join(stderr)))

def makeEnv(workerId):
    env = os.environ.copy()
    env["E_RL_STATEPIPE_PATH"] = f"/tmp/StatePipe{workerId}"
    env["E_RL_ACTIONPIPE_PATH"] = f"/tmp/ActionPipe{workerId}"
    env["E_RL_REWARDPIPE_PATH"] = f"/tmp/RewardPipe{workerId}"

    if not os.path.exists(env['E_RL_STATEPIPE_PATH']):
        os.mkfifo(env['E_RL_STATEPIPE_PATH'])
    if not os.path.exists(env['E_RL_ACTIONPIPE_PATH']):
        os.mkfifo(env['E_RL_ACTIONPIPE_PATH'])
    if not os.path.exists(env['E_RL_REWARDPIPE_PATH']):
        os.mkfifo(env['E_RL_REWARDPIPE_PATH'])

    return env





# Helper from GPT4:
def communicate_with_timeout(policy, workerId, state_dim, subprocess, timeout):
    """
    Executes communicateWithE in a separate thread and waits for it to complete with a timeout.
    If the function exceeds the timeout, terminates the subprocess and returns False indicating a timeout occurred.
    
    Args:
        policy: The policy object to pass to communicateWithE.
        workerId: The worker ID to pass to communicateWithE.
        state_dim: The state dimension to pass to communicateWithE.
        subprocess: The subprocess.Popen object representing the running E process.
        timeout: The timeout duration in seconds.
    
    Returns:
        A boolean indicating whether the function completed before the timeout.
    """
    def target():
        communicateWithE(policy, workerId, state_dim)

    thread = threading.Thread(target=target)
    thread.start()
    thread.join(timeout=timeout)
    if thread.is_alive():
        print("Timeout occurred, terminating subprocess and aborting communicateWithE.")
        try:
            subprocess.terminate()  # Ensure the subprocess is terminated
        except ProcessLookupError:
            print("eprover Subprocess already terminated.")

        thread.join()  # Wait for the thread to clean up
        return False  # Indicate that a timeout occurred
    return True  # Indicate successful completion








def runE(policy, eproverPath, problemPath, state_dim=5, soft_cpu_limit=1, cpu_limit=5, auto=False, auto_sched=False, create_info=True, verbose=False, dryRun=False, strat_file=None):
    problemName = os.path.split(problemPath)[1]
    workerId = random.randint(0,1_000_000_000)

    NORMAL_FLAGS = f"-l1 --proof-object --print-statistics --training-examples=3 --soft-cpu-limit={soft_cpu_limit} --cpu-limit={cpu_limit}"
    # NORMAL_FLAGS = ""

    command_args = [eproverPath, *NORMAL_FLAGS.split()]
    if verbose:
        command_args.append("-v")

    if strat_file is not None:
        # if strat_file is a folder, then we want to use the strategy files in that folder.
        # use the first one that matches the problem name
        if os.path.isdir(strat_file):
            strat_file = os.path.join(strat_file, f"{problemName}.strat")

            if not os.path.exists(strat_file):
                strat_file = None
                print("Couldn't find file: ", strat_file)

        command_args.append(f"--parse-strategy={strat_file}")
    else:
        pass


    if auto:
            command_args.append("--auto")
    elif auto_sched:
            command_args.append("--auto-schedule")
    
    command_args.append(problemPath)

    print("Eprover command: " + " ".join(command_args))

    if dryRun:
        return " ".join(command_args)

    # if verbose and not create_info:
    #     p = subprocess.Popen(command_args, env=makeEnv(workerId)) # calls E
    #     if policy is not None:
    #         # communicateWithE(policy, workerId, state_dim)
    #         execute_with_timeout(policy, workerId, state_dim, p, int(cpu_limit*1.5))
    #     p.wait()
    #     return

    # Run E itself with the args specified above.
    t1 = time()
    p = subprocess.Popen(command_args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=makeEnv(workerId))

    # Thread needed to read E's stdout/stderr incrementally
    result = []
    thread = threading.Thread(target=read_output, args=(p,result))
    thread.start()

    # Interact with E via named pipes using "policy" to map states to actions.
    if policy is not None:
        timed_out = not communicate_with_timeout(policy, workerId, state_dim, p, int(cpu_limit*1.5))

    # Finish reading E's stdout/stderr
    thread.join()
    stdout,stderr = result[0]
    t2 = time()

    # Make sure E is double dead...
    p.terminate()

    if verbose:
        print_dumb("About to print stdout/stderr...")
        print_dumb(stdout,stderr)
        print_dumb("The above was stdout / stderr...")

    # clean up pipes:
    os.remove(f"/tmp/StatePipe{workerId}")
    os.remove(f"/tmp/ActionPipe{workerId}")
    os.remove(f"/tmp/RewardPipe{workerId}")

    # Extract important info from E's stdout/stderr
    if create_info:
        if verbose:
            print_dumb("About to create info...")
        info = createInfoFromE(stdout,stderr, problemName, t1, t2, clonePolicy(policy))
        if verbose:
            print_dumb("Created info!")
        return info
    else:
        return stdout,stderr





#################### Alternative Environment for testing RL #############################
def createInfoFromLunarLander(states, actions, rewards, policy):
    info = {}
    info['solved'] = True
    info['problem'] = "LunarLander"
    info['states'] = np.array(states)
    info['actions'] = np.array(actions)
    info['rewards'] = np.array(rewards) / 200
    info['stdout'] = ''
    info['stderr'] = ''
    info['policy'] = policy.state_dict() if policy is not None else None
    return info

def runLunarLander(policy, problem):
    import gymnasium as gym
    env = gym.make("LunarLander-v2")
    observation, info = env.reset(seed=random.randint(0,1_000_000))
    states, actions, rewards = [],[],[]
    for _ in range(10000):
        sleep(0.00001)
        states.append(observation)
        torch_obs = torch.from_numpy(observation).to(torch.float).reshape([1,-1])
        action = select_action(policy, torch_obs).item()
        actions.append(action)

        observation, reward, terminated, truncated, info = env.step(action)
        rewards.append(reward)

        if terminated or truncated:
            env.close()
            return createInfoFromLunarLander(states, actions, rewards, policy)

#######################################################################################


def getOpt(params, lr):

    if args.opt_type == 'rmsprop':
        opt = torch.optim.RMSprop(params, lr = lr)
    elif args.opt_type == 'adam':
        opt = torch.optim.Adam(params, lr = lr)
    elif args.opt_type == 'sgd':
        opt = torch.optim.SGD(params, lr = lr, nesterov=True, momentum=0.9)
    elif args.opt_type == 'adagrad':
        opt = torch.optim.Adagrad(params, lr = lr)
    else:
        print(f"No Optimizer Being Used!!!: args.opt_type = {args.opt_type}")
        sys.exit(1)
    

    return opt


def getPolicy(log=False):

    print_if_log = lambda x: print(x) if log else None

    if args.policy_type == "none":
        print_if_log("Model is None")
        return None, None
    
    if args.load_model:
        print_if_log("loading model...")
        policy = torch.load(args.model_path)
        if not args.test:
            opt = getOpt(policy.parameters(), args.lr)
            opt.load_state_dict(torch.load(args.opt_path))
        else:
            opt = None
    else:
        print_if_log("not loading model...")

        if args.policy_type == "constcat":
            policy = PolicyNetConstCategorical(args.state_dim, args.CEF_no)
        elif args.policy_type == "nn":
            policy = PolicyNet(args.state_dim, args.n_units, args.CEF_no, args.n_layers)
        elif args.policy_type == "uniform":
            policy = PolicyNetUniform(args.CEF_no)
        elif args.policy_type == "attn":
            policy = PolicyNetAttn(args.state_dim, args.CEF_no, args.n_units)
        elif args.policy_type == "dqn":
            policy = PolicyNetDQN(args.state_dim, args.n_units, args.CEF_no, args.n_layers)
        else:
            print(f"No Policy Being Used!!!: args.policy_type = {args.policy_type}")

        opt = getOpt(policy.parameters(), args.lr)

    return policy, opt



def sendInfoAndEpisode(proc, profiler, info_queue, episode_queue, message_queue, episode_read):
    info = proc.get()

    policyModule, garbageOpt = getPolicy()
    policyModule.load_state_dict(info['policy'])
    policy = info['policy']
    info['policy'] = None
    info_queue.put(info)

    ep = Episode(info['problem'], info['states'], info['actions'], info['rewards'], policyModule)

    if len(ep.states) == 0:
        message_queue.put("len(ep.states) == 0!")
        with open("errors/zero_states.txt", "w+") as f:
            f.write(str(info) + "\n\n\n")

    if len(set([len(ep.states), len(ep.actions), len(ep.rewards)])) != 1:
        message = f"State / Action / Reward sizes Differ!: ({len(ep.states)},{len(ep.actions)},{len(ep.rewards)})"
        print(message)
        message_queue.put(message)

        min_len = min(len(ep.states), len(ep.actions), len(ep.rewards))
        ep = Episode(ep.problem, ep.states[:min_len], ep.actions[:min_len], ep.rewards[:min_len], ep.policy_net)

    # Only send episode to trainer if it is legit.
    else:
        # returns = calculateReturns(policy, ep, False, dummyProfiler, numpy=True, discount_factor=args.discount_factor)
        print("Calculating returns...")
        with profiler.profile("<g> calculateReturns"):
            returnsAdvProbsAndVals = calculateReturnsAndAdvantageEstimate(policyModule, ep, GAMMA=args.discount_factor, LAMBDA=args.LAMBDA, lunarLander=args.lunar_lander)
        modified_ep = Episode(ep.problem, ep.states, ep.actions, returnsAdvProbsAndVals, policy)
        print("After Calculating returns...")

        saveToDir("./debugging", modified_ep, prefix="ep")

        modified_ep = applyMaxBlame(modified_ep, args.max_blame)

        episode_read.value = False
        print("Sending episode...")
        episode_queue.put((info['solved'], modified_ep))

        while not episode_read.value:
            sleep(0.01)
    
    # probName and whether or not the problem was solved in presaturation interreduction.
    return (info['problem'], len(ep.states) == 0 and info['solved'])


# This is because I thought there was something that required
# a process sending certain data over an mp.Queue
# to remain alive until the data was received.
# def sendInfoAndEpisodeDetached(profiler, *args):
#     with profiler.profile("<g> sendInfoAndEpisodeDetached"):
#         proc = mp.Process(target=sendInfoAndEpisode, args=args)
#         proc.start()


def sendUnsentEpisodes(profiler, processes, sentCount, episode_queue, info_queue, message_queue, episode_read, onlyDone=False):

    if onlyDone:
        with profiler.profile("<g> send done but unsent episodes"):
            readiness = ",".join([str(1 if proc.ready() else 0) for proc in processes[sentCount:]])
            message_queue.put(f"{len(processes)} {sentCount} ({readiness})")
            for proc in processes[sentCount:]:
                if proc.ready():
                    sendInfoAndEpisode(proc, profiler, info_queue, episode_queue, message_queue, episode_read)
                    processes[sentCount] = None
                    sentCount += 1
                else:
                    # sleep(0.01) # Testing removing this...If it causes Issues, I'll put it back, but I noticed the gatherer spends a lot of time here...
                    break # important to quit so that sentCount truly represents a cutoff point in processes.
                    
    else:
        while sentCount < len(processes):
            sendInfoAndEpisode(processes[sentCount], profiler, info_queue, episode_queue, message_queue, episode_read)
            sentCount += 1
    
    return sentCount



def makeRunner(policy, args):
    if args.lunar_lander:
        runner = functools.partial(runLunarLander, policy)
    else:
        runner = functools.partial(runE, policy, args.eprover_path, state_dim=args.state_dim, strat_file=args.strat_file, soft_cpu_limit=args.soft_cpu_limit, cpu_limit=args.cpu_limit, auto=args.auto, verbose=args.verbose)
    return runner
    


# Instead of this function taking in the "unsentProcs" as was previously done,
# I need it to take in the full processes array to be able to mutate it.
# Therefore, processes, and sentCount must be sent in...
def waitForLearner(profiler, episode_queue, message_queue, processes, sentCount):
    with profiler.profile("<g> waiting for learner (rate limiting)"):
        unsentProcs = processes[sentCount:]
        firstUnready = lambda l: len(l) > 0 and not l[0].ready()
        numUnready = lambda l: len([x for x in l if not x.ready()])
        criteria = lambda l: numUnready(l) > 15 or (firstUnready(l) and len(unsentProcs)>15)

        sleepTimes = [0 for x in unsentProcs]
        dt = 0.5
        sleepCount = 1
        while episode_queue.qsize() > 10 or criteria(unsentProcs):
            sleepCount += 1
            unready_procs = [x for x in unsentProcs if not x.ready()]
            message = f"Sleeping for {dt} seconds for the {sleepCount}th time...Waiting...{len(unready_procs)} unready but enqueued episodes and {episode_queue.qsize()} episodes in queue."
            message += f"\n Unsent Procs: {''.join(['1' if x.ready() else '0' for x in unsentProcs])}"
            message += f"\n sleepTimes: {sleepTimes}"
            message_queue.put(message)
            # if random.random() < dt:
            #     print(message)
            sleep(dt)
            
            for i,proc in enumerate(unsentProcs):
                if not proc.ready():
                    sleepTimes[i] += 1
                else:
                    sleepTimes[i] = 0

            for i in range(len(sleepTimes) - 1, -1, -1):  # Iterate in reverse to avoid index errors
                if sleepTimes[i] > 2*args.cpu_limit*(1/dt) and i < args.num_workers:
                    message = f"We've waited 120 seconds for process {sentCount + i}. Aborting."
                    message_queue.put(message)
                    # print(message)
                    
                    # Remove ith element from unsentProcs
                    del unsentProcs[i]
                    del processes[sentCount + i]
                    del sleepTimes[i]  # Also remove the corresponding sleep time entry
                    
                    message = f"Process {sentCount + i} removed. Remaining processes: {len(unsentProcs)}"
                    message_queue.put(message)
                    # print(message)

                    




def getLatestPolicyFromTrainer(profiler, runner, policy, policy_queue):
    with profiler.profile("<g> receiving latest policy"):
        while policy_queue.qsize():
            policy = policy_queue.get()
            runner = makeRunner(policy, args)

    return runner, policy


def gather_episodes_process(policy_queue, problems, episode_queue, info_queue, message_queue, presat_info_queue, profiler_queue, stop_event, episode_read):
    policy = policy_queue.get()
    runner = makeRunner(policy, args)
    random.seed(args.seed)

    i = 0
    profiler = Profiler()
    while not stop_event.value:
        sentCount = 0
        with mp.Pool(args.num_workers) as p:
            processes = []
            probsSolvedPresat = set()

            random.shuffle(problems)

            for prob in problems:
                runner, policy = getLatestPolicyFromTrainer(profiler, runner, policy, policy_queue)
                message_queue.put(f"Adding {prob} to pool.")
                processes.append(p.apply_async(runner, (prob,)))
                waitForLearner(profiler, episode_queue, message_queue, processes, sentCount)
                sentCount = sendUnsentEpisodes(profiler, processes, sentCount, episode_queue, info_queue, message_queue, episode_read, onlyDone=True)
                profiler_queue.put(profiler.copy());profiler.reset()

            # After all problems have been attempted, send all remaining episodes.
            sentCount = sendUnsentEpisodes(profiler, processes, sentCount, episode_queue, info_queue, message_queue, episode_read, onlyDone=True) # onlyDone=False is better, but breaks things.
            
        profiler_queue.put(profiler.copy());profiler.reset()
        presat_info_queue.put(probsSolvedPresat)
        message_queue.put(f"Finished the {i}th proof attempt of all Problems. On to the {i+1}th...")
        message_queue.put("Checking if sleeping for 10 seconds helps...")
        sleep(10)
        i += 1


def keepTraining(everSolved, batches_processed, patience=5*2078, max_train_steps=1e6, keep_training_queue=None):

    print("Checking if training should be stopped...")

    if batches_processed > max_train_steps:
        return False

    if len(everSolved) > len(keepTraining.prevEverSolved):
        keepTraining.attemptsSinceNewSolution = 0
        keepTraining.prevEverSolved = set(everSolved)
        return True
    else:
        keepTraining.attemptsSinceNewSolution += 1
        keep_training_queue.put((len(keepTraining.prevEverSolved), keepTraining.attemptsSinceNewSolution, patience, batches_processed))
        return keepTraining.attemptsSinceNewSolution < patience

keepTraining.attemptsSinceNewSolution = 0
keepTraining.prevEverSolved = set()



def applyMaxBlame(ep, max_blame):

    if len(ep.rewards) == 0:
        return ep

    if isinstance(ep.rewards, tuple):
        if any(x != 0 for x in ep.rewards[0]):
            return ep
    elif any(x != 0 for x in ep.rewards):
        return ep

    return Episode(ep.problem, ep.states[:max_blame], ep.actions[:max_blame], ep.rewards[:max_blame], ep.policy_net)


def waitForEpisode(profiler, episode_queue, message_queue, episode_read):
    with profiler.profile("<t> waiting for episodes"):
        i=0
        while episode_queue.qsize() < 1:
            i+=1
            if i%10 == 0:
                print("trainer Waiting for episode", end='', flush=True)
                message_queue.put("Trainer waiting for episodes...")
                
            sleep(0.1)

    solved, ep = episode_queue.get()
    episode_read.value = True

    if solved and ep.problem not in train_policy_process.everSolved:
        message_queue.put(f"[cyan]{ep.problem}[/cyan] solved for the first time!")
        train_policy_process.everSolved.add(ep.problem)

    return ep


def assertNumeric(ep, message_queue):
    isNumeric = lambda x: np.issubdtype(x.dtype,np.number)
    allNumeric = all(isNumeric(x) for x in [ep.states, ep.actions, ep.rewards[0]])
    if not allNumeric:
        message_queue.put("Non numeric states, actions, or rewards encountered!")
        message_queue.put(",".join([str(x.dtype) for x in [ep.states, ep.actions, ep.rewards]]))
    return allNumeric


def addEpisodeToBatch(profiler, ep, batch, message_queue):
    if len(ep.states) > 0 and assertNumeric(ep, message_queue):
        with profiler.profile("<t> Numpy -> torch"):
            states = torch.from_numpy(ep.states).to(torch.float)
            actions = torch.from_numpy(ep.actions).to(torch.long)
            returns = torch.from_numpy(ep.rewards[0]).to(torch.float) # ep.rewards are actually (returns,advantages,log_probs,values)...see episode_queue.put
            advantages = torch.from_numpy(ep.rewards[1]).to(torch.float)
            log_probs = torch.from_numpy(ep.rewards[2]).to(torch.float)
            values = torch.from_numpy(ep.rewards[3]).to(torch.float)

        with profiler.profile("<t> Normalizing State"):
            if not args.lunar_lander:
                states = normalizeState(states)

        toAppend = [states, actions, returns, advantages, log_probs, values]
        if args.policy_type == "dqn":
            next_states = states[1:]
            next_states = torch.cat([next_states, 3.14159*torch.ones(1, states.shape[1])], dim=0) # a fake next state that will be detectable during training.
            toAppend.append(next_states)

        batch.append(toAppend)




def update_rollout_buffer(buff, newStuff):
    print(f"Current rollout buffer LENGTH {len(buff)}")
    while len(buff) > 1_000_000:
        buff.popleft()
    
    buff.extend(newStuff)


def train_policy_process(policy, opt, episode_queue, policy_queue, info_queue, keep_training_queue, message_queue, profiler_queue, stop_event, episode_read):

    policy_queue.put(clonePolicy(policy))

    batch = []
    model_iteration = 0
    profiler = Profiler()
    i = 0
    batches_processed = 0

    assert len(args.run.strip())
    model_history_dir = f"model_histories/{args.run}"
    shutil.rmtree(model_history_dir, ignore_errors=True)
    os.makedirs(model_history_dir, exist_ok=True)

    # need both for DQN with a target net.
    rollout_buffers = [deque(),deque()]
    rollout_buffer = rollout_buffers[0]

    while keepTraining(train_policy_process.everSolved, batches_processed,  args.train_patience, args.max_train_steps, keep_training_queue):
        print(f"train main loop {i}")
        i += 1
        message_queue.put(f"train main loop {i}")

        with profiler.profile("<t> entire while body"):

            print("About to wait for an episode...")
            addEpisodeToBatch(profiler, waitForEpisode(profiler, episode_queue, message_queue, episode_read), batch, message_queue)
            print(f"Got the episode... added to batch. Currently at {len(batch)} episodes")

            if len(batch) >= args.batch_size:
                print(f"About to optimize step for {len(batch)} episodes")
                with profiler.profile("<t> Optimize_step"):

                    # Batch is a list of (states, actions, returns, advantages, log_probs, values) tuples
                    # unzipped is then (all_states, all_actions, all_returns, all_advantages, all_log_probs, all_values)
                    # list(zip(*unzipped)) is then the same as batch, but the the original components catted into tensors.

                    unzipped = [torch.cat(X, dim=0) for X in zip(*batch)]

                    if args.policy_type == "dqn":
                        update_rollout_buffer(rollout_buffer, list(zip(*unzipped)))
                    else:
                        rollout_buffer = list(zip(*unzipped))

                    batches_processed += 1
                    info_queue.put((
                        opt, 
                        clonePolicy(policy),
                        optimize_step_ppo(opt, policy, rollout_buffer, args.ppo_batch_size, args.critic_weight, args.entropy_weight, args.max_grad_norm, args.epochs)
                    ))

                print("After optimize step...")

                model_iteration += 1
                torch.save(policy, f"{model_history_dir}/{model_iteration:05d}.pt")

                policy_queue.put(clonePolicy(policy))
                batch = []
        
        profiler_queue.put(profiler.copy())
        profiler.reset()
    

    print("###################################\n"*20)
    print(f"Finished Training: total batches processed / train steps: {batches_processed}")

    stop_event.value = True
train_policy_process.everSolved = set()



def logTraining(loss):
    print(loss)
    if logTraining.i < 15:
        for mode in loss:
            logTraining.runningLoss[mode].append(loss[mode])
    else:
        for mode in loss:
            if isinstance(logTraining.runningLoss[mode], Iterable):
                logTraining.runningLoss[mode] = mean(logTraining.runningLoss[mode]) if len(logTraining.runningLoss[mode]) > 0 else 0
            else:
                logTraining.runningLoss[mode] = 0.99*logTraining.runningLoss[mode] + 0.01*loss[mode]

        dashboard.updateLoss(loss, logTraining.runningLoss)

    logTraining.i += 1
logTraining.runningLoss = defaultdict(list)
logTraining.i = 0



def TrainPolicy(problems, args):
    policy, opt = getPolicy(log=True)
    history = ECallerHistory(args=args, delete=f"{args.run}_train")

    # Primary Stuff (necessary for learning to make sense)
    episode_queue = mp.Queue()
    policy_queue = mp.Queue()
    stop_event = mp.Value('b', False)
    episode_read = mp.Value('b', False)

    # Stuff used only for dashboard
    keep_training_queue = mp.Queue()
    message_queue = mp.Queue()
    gather_info_queue = mp.Queue()
    train_info_queue = mp.Queue()
    presat_info_queue = mp.Queue()
    profiler_queue = mp.Queue()

    # Start the process that calls E
    gatherProc = mp.Process(
        target = gather_episodes_process,
        args = (policy_queue, problems, episode_queue, gather_info_queue, message_queue, presat_info_queue, profiler_queue, stop_event, episode_read)
    )
    print("Starting gatherer...")
    gatherProc.start()


    # Start the process that trains the model
    trainProc = mp.Process(
        target = train_policy_process,
        args = (policy, opt, episode_queue, policy_queue, train_info_queue, keep_training_queue, message_queue, profiler_queue, stop_event, episode_read)
    )
    print("Starting trainer...")
    trainProc.start()


    # Update Dashboard during training.
    # Also save policy and history occasionally.
    numSuccess = 0
    numFailure = 0
    while not stop_event.value:
        # print("AAA\n"*100)
        if gather_info_queue.qsize() > 0:
            info = gather_info_queue.get()
            if info['solved']:
                numSuccess += 1
                if args.lunar_lander:
                    dashboard.procCounts.append(1)
                else:
                    dashboard.procCounts.append(info['processed_count'])
            else:
                numFailure += 1

            history.addInfo(info)
            dashboard.registerProofAttemptReward(info['rewards'].sum())
            
            if not args.lunar_lander:
                dashboard.registerProofAttemptSuccess(info['solved'])

        if presat_info_queue.qsize() > 0:
            dashboard.updatePresatInfo(presat_info_queue.get())

        while train_info_queue.qsize() > 0:
            # print("Line 825 ..." + "#"*1000)
            worked = False
            try:
                opt, policy, loss = train_info_queue.get()
                worked = True
            except Exception as e:
                message_queue.put("Failed to train_info_queue.get()...trying again...")
            if worked:
                logTraining(loss)

        if keep_training_queue.qsize() > 0:
            probsEverSolved, attemptsSinceSolution, patience, batchesProcessed = keep_training_queue.get()
            dashboard.updateProbsEverSolved(probsEverSolved, attemptsSinceSolution, patience, batchesProcessed, numSuccess, numFailure)
        if message_queue.qsize() > 0:
            dashboard.addMessage(message_queue.get())


        sizes = {
            "Episode Queue: ": episode_queue.qsize(),
            "Policy Queue: ": policy_queue.qsize(),
            "Gather_Info Queue: ": gather_info_queue.qsize(),
            "Train Info Queue: ": train_info_queue.qsize(),
            "KeepTraining Queue: ": keep_training_queue.qsize(),
            "Message Queue: ": message_queue.qsize(),
            "Presat info Queue: ": presat_info_queue.qsize(),
            "Profiler Queue: ": profiler_queue.qsize()
        }

        # print(sizes)
        dashboard.updateQueueInfo(sizes)

        while profiler_queue.qsize() > 0:
            dashboard.updateProfiler(profiler_queue.get())

        qs = [episode_queue, policy_queue, gather_info_queue, train_info_queue, keep_training_queue]
        max_queue_size = max(x.qsize() for x in qs)
        if max_queue_size < 5 and time() - TrainPolicy.lastSaved > 60:
            gatherState = "running" if gatherProc.is_alive() else "dead"
            trainState = "running" if trainProc.is_alive() else "dead"
            dashboard.addMessage(f"Saving Policy and History... (gatherer {gatherState}, trainer {trainState})")

            dead = {name for name,proc in zip(['gather','train'],[gatherProc, trainProc]) if not proc.is_alive()}
            
            history.save(f"{args.run}_train", None, eager=True if dead else False)
            torch.save(policy, args.model_path)
            torch.save(opt.state_dict(), args.opt_path)
            TrainPolicy.lastSaved = time()
            
            if dead:
                print(f"The following processes died: {dead}")
                print(("#"*100 + '\n')*15)
                sys.exit()


        dashboard.render(save=f"./dashboards/{args.run}.txt")

    gatherProc.join(timeout=5)
    gatherProc.kill()
    print("Finished gatherer...")

    trainProc.join(timeout=5)
    trainProc.kill()
    print("Finished trainer...")


    history.save(f"{args.run}_train", None, eager=True)
    torch.save(policy, args.model_path)
    torch.save(opt.state_dict(), args.opt_path)

    print("TrainPolicy done.")

    return history
TrainPolicy.lastSaved = 0


def EvaluatePolicy(policy, problems, args):
    history = ECallerHistory(args=args, delete=args.run)

    if policy is not None:
        policy.eval() # for dropout
    runner = makeRunner(policy, args)
    # if args.lunar_lander:
    #     runner = functools.partial(runLunarLander, policy)
    # else:
    #     runner = functools.partial(runE, policy, args.eprover_path, state_dim=args.state_dim, strat_file=args.strat_file, auto=args.auto, auto_sched=args.auto_sched, soft_cpu_limit=args.soft_cpu_limit, cpu_limit=args.cpu_limit)



    t1 = time()
    print("Evaluating...")
    probsSolved = 0
    for i in range(args.test_num):
        print(f"Solving all {len(problems)} problems for the {i}th time...")
        with mp.Pool(args.num_workers) as p:
            for j, info in enumerate(p.imap(runner, problems)):
                print(f"solving problem {j} of {len(problems)} for the {i}th / {args.test_num} time...")
                probsLeft = (len(problems) - j) + (args.test_num - i - 1) * len(problems)
                speed = (time() - t1) / (j+1+i*len(problems)) # seconds per problem
                print(f"Hours so far: {(time() - t1) / 60 / 60:.2f} hours")
                print(f"Estimated hours left: {probsLeft * speed / 60 / 60:.2f} hours")

                if info['solved']:
                    print('+', end='')
                    probsSolved += 1
                else:
                    print('-', end='')
                
                print(f"{probsSolved} / {j+1 + i*len(problems)} ({probsSolved / (j+1 + i*len(problems)):.2%})") # what percent solved?

                history.addInfo(info)
                history.save(f"{args.run}", None, eager=False)



    print(f"Number of Problems Solved (average over {args.test_num} runs): {probsSolved / args.test_num} / {len(problems)}")
    return history


def extractCEF_no(path):
    with open(path) as f:
        s = [l for l in f.readlines() if "heuristic_def" in l][0]
    return len(re.findall(r'([0-9])+[\.\*](\w+\([^\)]+\))', s))


if __name__ == "__main__":

    torch.set_num_interop_threads(8)
    torch.set_num_threads(8)

    # I commented the next line out because I was getting a shared memory manager timeout error
    # mp.set_sharing_strategy('file_system')

    # Both forkserver and spawn break things :(
    # mp.set_start_method('forkserver')

    # soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    # resource.setrlimit(resource.RLIMIT_NOFILE, (100_000,100_000))

    parser = argparse.ArgumentParser()
    parser.add_argument("--problems", default=os.path.expanduser("~/Desktop/ATP/GCS/MPTPTP2078/Bushy/Problems/"), help="path to where problems are stored")
    parser.add_argument("--run", default="test")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--test_num", type=int, default=10)
    parser.add_argument("--verbose", action="store_true", help='used in runE')
    parser.add_argument("--lunar_lander", action="store_true")
    parser.add_argument("--auto", action="store_true")
    parser.add_argument("--auto_sched", action="store_true")

    parser.add_argument("--lr", type=float, default=5e-5, help="learning rate for optimizer")
    parser.add_argument("--batch_size", type=int, default=4, help="number of episodes before training")
    parser.add_argument("--ppo_batch_size", type=int, default=128, help="Batch size for PPO updates")
    parser.add_argument("--n_units", type=int, default=100, help="Number of units per hidden layer in the policy")
    parser.add_argument("--n_layers", type=int, default=2, help="Number of hidden NN layers in the policy")
    parser.add_argument("--discount_factor", type=float, default=0.998, help="discount factor for RL")
    parser.add_argument("--LAMBDA", type=float, default=0.95, help="PPO discount for interpolating between full returns and TD estimate")
    parser.add_argument("--epochs", type=int, default=1, help="Epochs for each PPO training phase")
    parser.add_argument("--critic_weight", type=float, default=0.4)
    parser.add_argument("--entropy_weight", type=float, default=4e-5)
    parser.add_argument("--max_grad_norm", type=float, default=4.0)
    parser.add_argument("--max_blame", type=int, default=90_000, help="Maximum number of given clause selections to punish for a failed proof attempt...")
    
    parser.add_argument("--load_model", action="store_true")
    parser.add_argument("--model_path", default="latest_model.pt")
    parser.add_argument("--opt_path", default="latest_model_opt.pt")
    parser.add_argument("--opt_type", default="adam", choices=["adam", "sgd", "rmsprop", "adagrad"])
    parser.add_argument("--policy_type", default="nn", choices=["nn", "constcat", "none", "uniform", "attn", "dqn"])
    
    parser.add_argument("--state_dim", type=int, default=5)
    parser.add_argument("--CEF_no", type=int, default=-1, help="Number of cefs to use. -1 to load from strat_file")
    # parser.add_argument("--cef_file", default="cefs_auto.txt")
    parser.add_argument("--strat_file")

    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--eprover_path", default="eprover")
    parser.add_argument("--eproverScheduleFile", default=os.path.expanduser("~/eprover/HEURISTICS/schedule.vars"))
    parser.add_argument("--soft_cpu_limit", type=int, default=1)
    parser.add_argument("--cpu_limit", type=int, default=2)

    parser.add_argument("--train_patience", type=int, default=10*2078, help="How many proof attempts to wait for another solved problem before stopping training.")
    parser.add_argument("--max_train_steps", default=4500, type=int, help="Maximum number of PPO train batches to train on.") # 4500 ~= 11*2078 / 5 batches: decided from how many MPT epochs before no new probs solved
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    problems = glob(f"{args.problems}/*")

    # This dashboard should only be updated by TrainPolicy or things it calls synchronously
    dashboard = rich_dashboard.DashBoard(f"Experiment Information for \"{args.run}\"", entropy_weight=args.entropy_weight, args=args)


    if args.lunar_lander:
        args.CEF_no = 4 # number of actions in lunar lander gym env: 4

    if args.CEF_no == -1:

        if args.strat_file is None:
            pass
        elif os.path.isdir(args.strat_file):
            print("Ignoring CEF_no because strat_file is a directory.")
        else:
            args.CEF_no = extractCEF_no(args.strat_file)


    month = datetime.now().timetuple().tm_mon
    day = datetime.now().timetuple().tm_yday
    p = start_logging_process(f"logs/{args.run}.{month}.{day}.log")

    if args.test:
        args.load_model = True
        policy, opt = getPolicy(log=True)
        history = EvaluatePolicy(policy, problems, args)
        history.save(args.run, None, eager=True)
        p.terminate()
        IPython.embed()
    else:
        history = TrainPolicy(problems, args)
        p.terminate()
        sys.exit()
        # IPython.embed()

