# This is meant to be run alongside a call to eprover to do the
# interaction manually for debugging.
# Should be run from above scripts directory

import sys, os
sys.path.append('.')

from main import recvState, sendAction, recvReward, initPipe, select_action
from policy_grad import PolicyNet
import torch
import argparse
from time import sleep






if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--policy")
    parser.add_argument("--state_dim", default=5, type=int)
    parser.add_argument("numCefs", type=int)
    parser.add_argument("--pause", type=float, default=0.0)
    args = parser.parse_args()

    if args.policy:
        policy = torch.load(args.policy)
    else:
        policy = PolicyNet(args.state_dim, 100, args.numCefs, 2)


    # Create Named Pipes
    workerId = "1" # this is the default in the C code
    statePath = f"/tmp/StatePipe{workerId}"
    actionPath = f"/tmp/ActionPipe{workerId}"
    rewardPath = f"/tmp/RewardPipe{workerId}"

    if not os.path.exists(statePath):
        os.mkfifo(statePath)
    if not os.path.exists(actionPath):
        os.mkfifo(actionPath)
    if not os.path.exists(rewardPath):
        os.mkfifo(rewardPath)

    StatePipe = initPipe(statePath, send=False, log=False)
    ActionPipe = initPipe(actionPath, send=True, log=False)
    RewardPipe = initPipe(rewardPath, send=False, log=False)


    # Interact with E
    sync_num = 0
    while True:
        try:
            state = recvState(StatePipe, sync_num, args.state_dim)
            print(f"Received state: {state}")

            if state is None:
                os.close(StatePipe)
                os.close(ActionPipe)
                os.close(RewardPipe)
                print("Received state is None")
                break
            
            if args.pause > 0:
                sleep(args.pause)
                # input("Press Enter to continue...")

            print("Selecting action...")
            action = select_action(policy, state)
            print(f"Selected action: {action}")

            print(f"Sending action {action}...")
            sendAction(action, ActionPipe, sync_num)
            print(f"Sent action: {action}")

            recvReward(RewardPipe, sync_num)

            sync_num += 1

        except OSError as e:
            print("OSError. Probably pipe closed")
            break

    # Clean up leftover pipes:
    os.close(StatePipe)
    os.close(ActionPipe)
    os.close(RewardPipe)
