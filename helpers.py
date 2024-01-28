import os, struct, select
import torch
import time
from time import sleep
import psutil
from multiprocessing import Process
from glob import glob

# Named Pipe Helpers ##########################################################


def saveToDir(path, thing, prefix=''):
    print("Fake SaveToDir")
    return # for disabling 
    if prefix != '':
        prefix = prefix + '_'

    os.makedirs(path, exist_ok=True)

    existing = glob(f"{path}/{prefix}*.pt") #prefix1.pt prefix2.pt prefix3.pt, etc
    n = max([int(x.split(prefix)[-1].split('.')[0]) for x in existing] + [0])+1

    torch.save(thing, f"{path}/{prefix}{n:05d}.pt")



def sleepUntilHasData(pipe, timeout):
    print(" (sthd: ", end='')
    r = []
    t1 = time.time()
    while len(r) == 0 and (time.time() - t1) < timeout:
        print('.', end='')
        r, w, e, = select.select([pipe],[],[], 1)

    print(") ")
    return len(r) > 0
    

def read(pipe, n, timeout=300):
    print(f"Trying to read {n} bytes")
    t1 = time.time()
    bytez = b''
    while len(bytez) < n and (time.time() - t1) < timeout:
        if len(bytez) > 0:
            print(f"Read so far ({len(bytez)}/{n}): ", bytez)
        if sleepUntilHasData(pipe, timeout):
            newBytes = os.read(pipe, n-len(bytez))
            bytez += newBytes
            if len(newBytes) == 0 and len(bytez) == 0:
                sleep(1)
                # print("os.read returned b''")
                # break
    
    print(f"read {len(bytez)} bytes")
    if len(bytez) < n:
        return None
    
    return bytez

def write(pipe, bytez, timeout=30):
    num_written = 0
    t1 = time.time()
    while num_written < len(bytez) and time.time() - t1 < timeout:
        num_written += os.write(pipe, bytez[num_written:])
    
    if num_written == len(bytez):
        return True
    return False

def initPipe(pipePath, send=False, log=True):
    if log:
        print(f"Initializing Pipe ({pipePath})")

    mode = os.O_WRONLY if send else os.O_RDONLY

    retval = os.open(pipePath, mode)
    
    if log:
        print(f"Finished Initializing Pipe ({pipePath})")
    return retval


def normalizeState(state):
    # GCSCount, ProcessedSize, UnprocessedSize, ProcessedWeight, UnprocessedWeight, Action
    tMax = 40_000
    pSizeMax = 9_000
    uSizeMax = 1_300_000
    pWeightMax = 100
    uWeightMax = 200
    aMax = 40
    histDivideBy = torch.tensor([pSizeMax, uSizeMax, pWeightMax, uWeightMax, aMax], dtype=torch.float)
    currDivideBy = torch.tensor([pSizeMax, uSizeMax, pWeightMax, uWeightMax, tMax], dtype=torch.float)

    memSize = state.shape[1] // 5
    if memSize > 1:
        toDivideBy = torch.cat([
            histDivideBy.repeat(1, memSize-1)[0],
            currDivideBy
        ], dim=0)
    else:
        toDivideBy = currDivideBy

    s = state / toDivideBy
    return s

# def recvState(StatePipe, sync_num, CEF_no):
#     stateSize = 4 + 3*8 + 2*4
#     bytez = read(StatePipe, stateSize)
#     if bytez is None:
#         return None, False

#     [sync_num_remote, ever_processed, processed, unprocessed, processed_weight, unprocessed_weight] = struct.unpack("=iqqqff", bytez)
#     episode_begin = (sync_num_remote == 0 and sync_num > 0)
#     assert (episode_begin or sync_num_remote == sync_num), f"{sync_num_remote} != {sync_num}"
    

#     tensor = torch.tensor([ever_processed, processed, unprocessed, processed_weight, unprocessed_weight]).reshape(1, -1)
    
#     # "Normalize" using log as per Geoff's suggestion.
#     tensor = normalizeState(tensor)

#     return tensor, episode_begin


def sendAction(action, pipe, sync_num):
    return write(pipe, struct.pack("=ii", sync_num, action.item()))
    


def recvReward(pipe, sync_num):
    rewardSize = 4 + 4
    bytez = read(pipe, rewardSize)
    if bytez is None:
        return None

    [sync_num_remote, reward] = struct.unpack("=if", bytez)
    assert sync_num_remote == sync_num, f"{sync_num_remote} != {sync_num}"

    assert reward in [0.0, 1.0]
    reward = 1.0 if reward == 1.0 else 0.0
    return torch.tensor(reward).reshape(1)



def sendProbId(pipe, id):
    write(pipe, struct.pack("=i", id))


def recvProbId(pipe):
    chars = read(pipe, 4)
    if chars is None:
        return None
        
    [id] = struct.unpack("=i", chars)
    return id




# Episode Helpers #############################################################

from collections import namedtuple
Episode = namedtuple('Episode', ['problem', 'states', 'actions', 'rewards', 'policy_net'])




# Generic Helpers #############################################################

def mean(l):
    return sum(l) / len(l)






def log_resources(log_file="resources.log", interval=4):

    os.makedirs(os.path.split(log_file)[0], exist_ok=True)
    if os.path.exists(log_file):
        os.remove(log_file)

    while True:
        print("...")
        cpu_percent = psutil.cpu_percent()
        memory_info = psutil.virtual_memory()
        shared_memory_info = psutil.virtual_memory()._asdict()["shared"]
        shared_memory_gb = shared_memory_info / 1_000_000_000
        
        log_msg = f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}, CPU: {cpu_percent}%, RAM: {memory_info.percent}%, Shared RAM: {shared_memory_gb}GB\n"
        with open(log_file, "a") as file:
            file.write(log_msg)
        
        sleep(interval)



def start_logging_process(path, interval=4):
    print("starting logging...")
    p = Process(target=log_resources, args=(path, interval))
    p.start()
    return p

