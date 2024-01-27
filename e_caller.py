from time import time, sleep
from glob import glob
from collections import defaultdict
import re, os, sys, shutil

from rich.progress import track
old_print = print
from rich import print

import IPython
import torch
from helpers import mean
import termplotlib as tpl


def parseFolds(s):
    if s==None:
        return None

    try:
        folds = [int(x) for x in s.split(",")]
        return folds
    except:
        print("Failed to parse folds")
        exit()

def mkfifos(num):
    try:
        rw = 0o600
        os.mkfifo(f"/tmp/ProbIdPipe{num}", rw)
        os.mkfifo(f"/tmp/StatePipe{num}", rw)
        os.mkfifo(f"/tmp/ActionPipe{num}", rw)
        os.mkfifo(f"/tmp/RewardPipe{num}", rw)
        print("Made Named Pipes")
    except:
        print("Named Pipes Already Exist")

def extractStatus(stdout):
    try:
        return re.search("SZS status (.*)", stdout)[1]
    except:
        return "No SZS Status"

def extractProcessedCount(stdout):
    try:
        return int(re.search("# Processed clauses\s+: (.*)", stdout)[1])
    except:
        return "Error extracting processed count"

def extractOverhead(stdout):
    try:
        statePipeTime = re.search("# RL Seconds spent sending states\s*: (.*)", stdout)[1]
        actionPipeTime = re.search("# RL Seconds spent recieving actions\s*: (.*)", stdout)[1]
        rewardPipeTime = re.search("# RL Seconds spent sending rewards\s*: (.*)", stdout)[1]
        prepTime = re.search("# RL Seconds spent constructing 'state'\s*: (.*)", stdout)[1]
        return float(statePipeTime), float(actionPipeTime), float(rewardPipeTime), float(prepTime)
    except:
        return ["Error extracting overhead" for _ in range(4)]





def normalizeGiven(given):
    try:
        return re.search(r"cnf\([0-9a-z_]+, [a-z_]+, (.*)\)\.", given)[1]
    except:
        return None

def normalizePos(pos):
    try:
        return re.search(r"cnf\([0-9a-z_]+, [a-z_]+, (.*)\)\.", pos)[1]
    except:
        return None


def getGivenClauses(stdout):
    return [normalizeGiven(line) for line in stdout.split("\n") if "Given Clause" in line]

def getPositiveClauses(lines):
    return [normalizePos(line.split("# trainpos")[0]) for line in lines if "# trainpos" in line]

def getNegativeClauses(lines):
    return [normalizePos(line.split("#trainneg")[0]) for line in lines if "#trainneg" in line]



def getStates(stdout):
    try:
        matches = re.findall(r"RL State: (.*)\n", stdout)
        states = []
        for s in matches:
            try:
                states.append(eval(s))
            except:
                print(f"failed to eval '{s}' when parsing state")
                # print("stdout:", stdout)
                abridged = stdout[:3000] + '<...>\n'*10 + stdout[-3000:]
                old_print("stdout abridged...: ", abridged if len(stdout) > 10000 else stdout)
        return states
    except:
        print("FAILED TO GET STATES!")
        return ['FAILURE_TO_REGEX_STATES']


# def getActions(stdout):
#     try:
#         matches = re.findall("CEF Choice: ([0-9\.]*)\n", stdout)
#         return [eval(s) for s in matches]
#     except Exception as e:
#         print(f"FAILED TO GET ACTIONS: {e}")
#         print("The above was an error on line 116 of e_caller.py...")
#         return ["FAILURE_TO_REGEX_ACTIONS"]


# Function from GPT4 :)
def getActions(stdout, context_lines=5):
    lines = stdout.split('\n')
    actions = []
    for i, line in enumerate(lines):
        try:
            match = re.search("CEF Choice: ([0-9\.]*)", line)
            if match:
                actions.append(eval(match.group(1)))
        except Exception as e:
            print(f"FAILED TO GET ACTIONS: {e}")
            start = max(i - context_lines, 0)
            end = min(i + context_lines + 1, len(lines))
            s = '\n'.join(lines[start:end])
            print(f"Context around failed line {i}:\n{s}")
            return ["FAILURE_TO_REGEX_ACTIONS"]
    return actions




# def varsFromLits(lits):
#     return set(sum([re.findall(r"X[0-9]+",lit) for lit in lits], []))


# def applyMapToLit(map, lit):
#     placeholders = [f"__placeholder({i})__" for i in range(len(map))]
#     for key,placeholder in zip(map.keys(), placeholders):
#         lit = lit.replace(key,placeholder)

#     for placeholder,val in zip(placeholders, map.values()):
#         lit = lit.replace(placeholder, val)
#     return lit

# def allSubstitutions(literals, original_vars, new_vars):
#     n = len(new_vars)

#     for permutation in itertools.permutations(new_vars, n):
#         map = dict(zip(original_vars, permutation))
#         yield set([applyMapToLit(map,lit) for lit in literals])


# def existsUnifyingSub(l1, l2):
#     l1_vars = varsFromLits(l1)
#     l2_vars = varsFromLits(l1)

#     if len(l1_vars) != len(l2_vars):
#         return False
    
#     map1 = {k:"__placeholder__" for k in l1_vars}
#     masked_l1 = set([applyMapToLit(map1, lit) for lit in l1])

#     map2 = {k:"__placeholder__" for k in l2_vars}
#     masked_l2 = set([applyMapToLit(map2, lit) for lit in l2])

#     if masked_l1 != masked_l2:
#         return False

#     for subbed in allSubstitutions(l2, l2_vars, l1_vars):
#         if l1 == subbed:
#             return True
#     return False


# def litSetsEqual(lits1, lits2):

#     if len(lits1) != len(lits2):
#         return False

#     if lits1 == lits2:
#         return True

#     return existsUnifyingSub(lits1, lits2)


# def clause_in_set(g, lits_set):

#     for lits in lits_set:
#         if litSetsEqual(lits, g):
#             return True

#     return False


# def getRewards(stdout):
#     lines = stdout.split("\n")
    
#     given = getGivenClauses(stdout)

#     pos = getPositiveClauses(lines)
#     posClauses = [set(x[1:-1].split("|")) for x in pos]

#     rewards = [1 if g is not None and clause_in_set(set(g[1:-1].split("|")),posClauses) else 0 for g in given]

#     total = sum(rewards)
#     if total == 0:
#         return rewards

#     return [x / total for x in rewards]


def getRewards(stdout, n):

    rewards = set([int(x) for x in re.findall("# trainpos (-?[0-9]*)", stdout)])
    rewards = [1 if i in rewards else 0 for i in range(n)]

    total = sum(rewards)
    if total == 0:
        return rewards

    if total == 1:
        return [x / total for x in rewards] # weird that this happens ever...

    return [x / (total-1) for x in rewards]




# Override rich track so I can embed IPython
def track(*args, **kwargs):
    return args[0]


def getLatestModel(path):
    t = time()
    # If we've gotten it recently, just use that one
    if t - getLatestModel.lastUpdated < 1.0:
        return getLatestModel.model

    # Keep trying if getLatestModel.model is None
    tries = 0
    while getLatestModel.model is None or tries < 3:
        try:
            getLatestModel.model = torch.load(path)
            getLatestModel.lastUpdated = time()
            break
        except:
            sleep(1)

        tries += 1

    return getLatestModel.model

getLatestModel.lastUpdated = 0
getLatestModel.model = None

def getConfigName(stderr, yell=True):
    matches = re.findall(r"# Configuration: (.*)", stderr)
    if matches:
        return matches[-1]
    elif yell:
        print("NO CONFIG NAME!!!")

def parseHeuristicDef(heuristic):
    return {k:v for v,k in re.findall(r'([0-9]+)\.([^\)]+\))', heuristic)}

def loadConfigurations(pathToScheduleFile):
    with open(pathToScheduleFile) as f:
        lines = f.readlines()

    configs = {}
    for line in lines:
        if "heuristic_def" in line:
            key = re.findall(r'"([^\"]*)"', line)[0]
            value = re.findall(r'heuristic_def: \\"([^\\]+)', line)[0]
            configs[key] = parseHeuristicDef(value)

    return configs


def getConfig(stdout, stderr, configName=None, configurations=None, yell=True):
    if configName is None:
        configName = getConfigName(stderr+stdout, yell=yell)
    if configName is None:
        return None
    
    if configurations is None:
        if getConfig.configurations is not None:
            configurations = getConfig.configurations
        else:
            getConfig.configurations = loadConfigurations(args.eproverScheduleFile)
            configurations = getConfig.configurations
        
    return configurations[configName]
getConfig.configurations = None


def getCEFs(stdout, stderr, actions):
    try:
        configCEFs = list(getConfig(stdout, stderr).keys())
        return [configCEFs[i] for i in actions]
    except:
        IPython.embed()




class ECallerHistory:

    def __init__(self, args=None, delete=None):
        self.args = args
        self.history = defaultdict(list)
        self.timeToSave = 0
        self.lastSaved = 0

        if delete is not None:
            self.deleteRun(delete)
    
    def deleteRun(self, run):
        path = f"./ECallerHistory/{run}"
        if os.path.exists(path):
            shutil.rmtree(path)
        else:
            print(f"Run {run} not found so not deleting...")
    
    def probsSolved(self):
        return {x for x in self.history.keys() if self.getProofCount(x)}

    def merge(self, others):
        self.args = []
        for other in others:
            h = other.history
            
            # only append if other has args attribute:
            # To accommodate old version: AttributeError: 'ECallerHistory' object has no attribute 'args'
            if hasattr(other, 'args'):
                self.args.append(other.args)
            for key, value in other.history.items():
                self.history[key].extend(value)
        
    def addInfo(self, info):
        self.history[info['problem']].append(info)

    def getProofCount(self, prob):
        return len([1 for x in self.history[prob] if x['solved']])
    
    def getProofPercentage(self, prob):
        return self.getProofCount(prob) / len(self.history[prob])

    def getFirstProofInfo(self, prob):
        for info in self.history[prob]:
            if info['solved']:
                return info

    def getLatestProofInfo(self, prob):
        for info in reversed(self.history[prob]):
            if info['solved']:
                return info
    
    def getProcCounts(self, prob):
        return [info['processed_count'] for info in self.history[prob] if info['solved']]

    def getProcCountsReported(self, prob):
        return [info['processed_count_reported'] for info in self.history[prob] if info['solved']]

    def avgProcCountDecrease(self):
        decreases = []
        for prob in self.history:
            counts = self.getProcCounts(prob)
            if counts:
                decreases.append(counts[0] - counts[-1])
        return mean(decreases) if decreases else None

    def save(self, run, num=None, eager=True, refresh=True):

        t1 = time()
        if not eager and time() - self.lastSaved < 10*self.timeToSave:
            return

        if num is None:
            num = len(glob(f"./ECallerHistory/{run}/*.history"))+1

        os.makedirs(f"./ECallerHistory/{run}", exist_ok=True)
        torch.save(self, f"./ECallerHistory/{run}/{num}.history")

        if refresh:
            self.history = defaultdict(list)
        

        self.timeToSave = time() - t1
        self.lastSaved = time()


    def deleteKeysFromInfos(self, keys):
        for prob, infos in self.history.items():
            for info in infos:
                for key in keys:
                    del info[key]


    @staticmethod
    def load(run, num=None, keysToDeleteFromInfos=[]):
        histories = []
        if num is None:
            for filename in glob(f"./ECallerHistory/{run}/*.history"):
                histories.append(torch.load(filename))
                if len(keysToDeleteFromInfos):
                    histories[-1].deleteKeysFromInfos(keysToDeleteFromInfos)
        else:
            histories = [torch.load(f"./ECallerHistory/{run}/{num}.history")]
            if len(keysToDeleteFromInfos):
                histories[-1].deleteKeysFromInfos(keysToDeleteFromInfos)

        x = ECallerHistory()
        x.merge(histories)
        return x

    @staticmethod
    def load_safe(run, num=None, progress=False):

        if not progress:
            f = lambda x: x
        else:
            f = track

        if num is None:
            histories = []
            for x in f(glob(f"./ECallerHistory/{run}/*.history")):
                while True:
                    try:
                        histories.append(torch.load(x))
                        break
                    except:
                        print(f"Failed to load {x}...retrying")
                        sleep(1)
        else:
            while True:
                try:
                    histories = [torch.load(f"./ECallerHistory/{run}/{num}.history")]
                    break
                except:
                    print("Failed to load history...retrying")
                    sleep(1)
        x = ECallerHistory()
        x.merge(histories)
        return x



    def learningGraph(self, smooth=2078):
        """Shows a termplotlib plot of proof percentage over time with the designated amount of smoothing"""
        all_infos = [info for infos in self.history.values() for info in infos]
        sorted_infos = sorted(all_infos, key=lambda x: x['timestamp'])
        sorted_solved = [1.0 if x['solved'] else 0.0 for x in sorted_infos]

        ys = [mean(sorted_solved[i:i+smooth]) for i in range(0, len(sorted_solved), smooth)]
        ys = ys[:-1]
        xs = list(range(len(ys)))

        fig = tpl.figure()
        fig.plot(xs,ys)
        return fig.get_string()



    def summarize(self, smooth=2078):
        """This method makes a really cool rich dashboard to show all relevant info in this history."""
        print(self.learningGraph(smooth))
        print(f"Average Processed Count Decrease across problems: {self.avgProcCountDecrease()}")
