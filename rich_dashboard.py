from math import log2, exp
from numbers import Number
import rich
from rich import print
from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
import time
import psutil
import termplotlib as tpl
import numpy as np
from datetime import datetime
from my_profiler import Profiler
import itertools
import os
from helpers import saveToDir

def grouper(iterable, n):
    it = iter(iterable)
    while True:
        chunk = tuple(itertools.islice(it, n))
        if not chunk:
            return
        yield chunk

def mean(x):
    return sum(x) / len(x)

class DashBoard:

    def __init__(self, title="Rich Dashboard", entropy_weight=0.001, args=None):
        self.profiler = Profiler()
        self.args = args
        self.t0 = time.time()
        self.lastRendered = 0
        self.successGraphLastUpdated = 0
        self.rewardGraphLastUpdated = 0
        self.iteration = 0

        self.title = title
        self.console = Console()

        self.loss = {}
        self.running_loss = {}
        self.entropy_weight = entropy_weight
        self.losses = []
        self.smoothedLosses = []

        self.solvedPresat = []
        self.probsEverSolved = 0
        self.attemptsSinceSolution = 0
        self.patience = "Unknown"
        self.numSuccess = 0
        self.numFailure = 0

        self.messages = []

        self.queueInfo = {}

        self.proofAttemptSuccesses = []
        self.proofAttemptRewards = []




    def updateProfiler(self, otherProfiler):
        self.profiler.merge(otherProfiler)

    def getProfilerPanel(self):
        return Panel(self.profiler.report(as_str=True))


    def registerProofAttemptSuccess(self, success):
        self.proofAttemptSuccesses.append(success)

    def getProofAttemptSuccessRateGraphPanel(self):
        if time.time() - self.successGraphLastUpdated > 60:
            ys = self.proofAttemptSuccesses
            step = max(len(ys)//8,1)
            ys = np.array([mean(ys[i:i+step]) for i in range(0,len(ys),step)][:-1])
            xs = list(range(len(ys)))

            fig = tpl.figure()
            fig.plot(xs,ys)
            self.successRateGraph = fig.get_string() + f"\nmin_success_rate: {min(ys)}, max_success_rate: {max(ys)}" if len(ys) else ""
            self.successRateGraph = f"[green]{self.successRateGraph}[/green]"
            self.successGraphLastUpdated = time.time()

        try:
            return Panel(self.successRateGraph, title=f"Proof Attempt Success Rate - {round(time.time() - self.successGraphLastUpdated)} seconds ago")
        except:
            return Panel("Failed to construct Proof Attempt Success Rate Graph Panel")


    def registerProofAttemptReward(self, reward):
        self.proofAttemptRewards.append(reward)


    def getRewardGraphPanel(self):

        if time.time() - self.rewardGraphLastUpdated > 60:
            ys = self.proofAttemptRewards
            step = max(len(ys)//8,1)
            ys = np.array([mean(ys[i:i+step]) for i in range(0,len(ys),step)][:-1])
            xs = list(range(len(ys)))

            fig = tpl.figure()
            fig.plot(xs,ys)
            self.rewardGraph = fig.get_string() + f"\nmin_reward_sum: {min(ys)}, max_reward_sum: {max(ys)}" if len(ys) else ""
            self.rewardGraph = f"[green]{self.rewardGraph}[/green]"
            self.rewardGraphLastUpdated = time.time()

        try:
            return Panel(self.rewardGraph, title=f"Average Reward - {round(time.time() - self.rewardGraphLastUpdated)} seconds ago")
        except:
            return Panel("Failed to get RewardGraphPanel")


    
    def updateLoss(self, loss, running_loss):
        self.loss = loss.copy()
        self.running_loss = running_loss.copy()

        if len(list(loss.keys())):
            self.losses.append(self.loss)
        
        if len(list(running_loss.keys())):
            self.smoothedLosses.append(self.running_loss)
    
    def proofAttemptsPerMinute(self):
        minutesPassed = (time.time() - self.t0) / 60
        return len(self.losses) / minutesPassed, minutesPassed

    def getLossPanel(self):
        try:
            grid = Table.grid()

            grid.add_column()
            grid.add_column(justify="right")
            grid.add_column(justify="left")

            for key in self.loss:
                loss = f"{self.loss[key]:.6f}"
                entropy_part = ""
                if key == "entropy":
                    choices = exp(-self.loss[key] / self.entropy_weight)
                    entropyInBits = log2(choices)
                    entropy_part = f" ({entropyInBits:.2f} bits) ({choices:.2f} choices)"

                grid.add_row(f"[white]{key.capitalize()} Loss:[/white]  ", loss, entropy_part)
            
            grid.add_row("","")
            for key in self.running_loss:
                loss = f"{self.running_loss[key]:.6f}"
                entropy_part = ""
                if key == "entropy":
                    choices = exp(-self.running_loss[key] / self.entropy_weight)
                    entropyInBits = log2(choices)
                    entropy_part = f" ({entropyInBits:.2f} bits) ({choices:.2f} choices)"

                grid.add_row(f"[white]{key.capitalize()} Running Loss:[/white]  ", loss, entropy_part)

            

            papm, minutes = self.proofAttemptsPerMinute()
            return Panel(
                Group(
                    f"[white]Proof Attempts per minute:[/white] {papm:.2f} ({len(self.losses)} / {minutes:.1f})",
                    grid
                )
            )
        except:
            return Panel("Failed to get Loss panel")





    def updatePresatInfo(self, probsSolvedThisTime):
        self.solvedPresat.append(probsSolvedThisTime)

    def getPresatInfoPanel(self):
        try:
            s = str([len(x) for x in self.solvedPresat])
            return Panel(s, title="Presaturation Information (Not accurate for now...)")
        except:
            return Panel("Failed to get PresatInfoPanel")


    def updateProbsEverSolved(self, probsEverSolved, attemptsSinceSolution, patience, batchesProcessed, numSuccess, numFailure):
        self.probsEverSolved = probsEverSolved
        self.attemptsSinceSolution = attemptsSinceSolution
        self.patience = patience
        self.batchesProcessed = batchesProcessed
        self.numSuccess = numSuccess
        self.numFailure = numFailure

    def getProbsEverSolvedPanel(self):

        try:
            try:
                timeTillFinished = ((self.patience - self.attemptsSinceSolution) / self.proofAttemptsPerMinute()[0]) / 60
            except:
                timeTillFinished = "Unknown"

            successRate = 100 * self.numSuccess / (self.numSuccess + self.numFailure) if self.numSuccess + self.numFailure > 0 else 0

            lines = [
                f"[white]Problems Ever Solved:                        [/white] {self.probsEverSolved}",
                f"[white]Attempts processed since last novel solution:[/white] {self.attemptsSinceSolution} / {self.patience}",
                f"[white]Estimated Hours till finished training:       [/white] {timeTillFinished:5}",
                f"[white]Batches Processed:                           [/white] {self.batchesProcessed}",
                f"[white]Proof attempt success rate:                   [/white] {successRate:.2f}%",
            ]
            return Panel("\n".join(lines), title="Solution History")
        except:
            return Panel("Failed to get ProbsEverSolvedPanel")




    def updateQueueInfo(self, d):
        self.queueInfo.update(d)

    def getQueueInfoPanel(self):
        try:
            lines=[]
            for k,v in self.queueInfo.items():
                lines.append(f"[white]{k:25}[/white] {v}")
            return Panel("\n".join(lines), title="Queue Sizes")
        except:
            return Panel("Failed to get QueueInfoPanel")


        
    def addMessage(self, message):
        self.messages.append(self.getTimeStr() + f"[green] {message}[/green]")
    
    def getMessagePanel(self):
        return Panel("\n".join(self.messages[-6:]), title="Messages")




    def getArgsPanel(self, columns=6, color="green"):
        
        pairs = []
        for key,val in self.args.__dict__.items():
            pairs.append(f"[white]{key}:[/white] [{color}]{val}[/{color}]")
        
        grid = Table.grid(expand=True)
        for _ in range(columns):
            grid.add_column()
        
        for group in grouper(pairs, columns):
            grid.add_row(*group)

        return Panel(grid)

    def getSubprocessPanel(self):
        current_process = psutil.Process()
        children = current_process.children(recursive=True)
        return Panel(f"[white]Child Processes:[/white] {len(children)}")

    def getLossGraphPanel(self, mode="total", color="blue", title="Graph", smoothed=False):

        losses = self.smoothedLosses if smoothed else self.losses
        try:
            n = len(losses)

            step = max(n // 50, 1)
            mean_step = max(step // 10, 1)
            xs = list(range(0,n, step))
            
            ys = [mean([l[mode] for l in losses[i:i+step:mean_step]]) for i in xs]

            if mode == "entropy":
                # convert entropy loss to bits of entropy
                ys = np.log2(np.exp(-np.array(ys)/self.entropy_weight))

            fig = tpl.figure()
            fig.plot(xs,ys)

            return Panel(fig.get_string(), style=color, title=title)
        except Exception as e:
            saveToDir('debugging/buggyLosses', self.losses, 'losses')
            print("Buggy Losses!")
            print(self.losses)
            return Panel(f"Failed to get LossGraphPanel (mode={mode}) (e={e})")

    
    def getTimeStr(self):
        return "[white]" + datetime.now().strftime("%H:%M:%S") + "[/white]"


        
    def render(self, save=None):

        renderStartTime = time.time()
        if renderStartTime - self.lastRendered < 5:
            return

        self.iteration += 1
        smooth = bool(self.iteration % 2)

        lossGraphs = Table.grid(expand=True)
        lossGraphs.add_column(justify="center");lossGraphs.add_column(justify="center")
        lossGraphs.add_row(
            self.getLossGraphPanel("total", "blue", f"Total{' Smoothed' if smooth else ''} Loss", smoothed=smooth), 
            self.getLossGraphPanel("entropy", "cyan", f"{ 'Smoothed ' if smooth else ''}Entropy in bits", smoothed=smooth)
        )

        progressPanel = self.getRewardGraphPanel() if (self.iteration % 2) == 0 else self.getProofAttemptSuccessRateGraphPanel()

        leftSide = Group(
            self.getLossPanel(),
            progressPanel,
            self.getProbsEverSolvedPanel()
        )


        bottomRight = Table.grid(expand=True)
        bottomRight.add_column(justify="left");bottomRight.add_column(justify="center")

        criticLossPanel = self.getLossGraphPanel("critic", "yellow", f"{'Smoothed ' if smooth else ''}Critic Loss", smoothed=smooth)
        actorLossPanel = self.getLossGraphPanel("actor", "yellow", f"{'Smoothed ' if smooth else ''}Actor Loss", smoothed=smooth)
        profilingPanel = self.getProfilerPanel()

        bottomRight.add_row(
            Group(
                self.getSubprocessPanel(),
                self.getQueueInfoPanel(),
                self.getMessagePanel(),
                self.getPresatInfoPanel()
            ),
            criticLossPanel if (self.iteration % 3) == 0 else (profilingPanel if (self.iteration % 3) == 1 else actorLossPanel)
        )
        rightSide = Group(
            lossGraphs,
            bottomRight,
        )

        grid = Table.grid(expand=True)
        grid.add_row(Panel(leftSide, style="blue"), Panel(rightSide, style="red"))

        hours = (time.time() - self.t0) // 3600
        minutes = ((time.time() - self.t0) - hours*3600) // 60

        mainPanel = Panel(
            Group(self.getArgsPanel(),grid), 
            style="green", 
            title=self.title + f" (render took {time.time()-renderStartTime:.2f} seconds) - {self.getTimeStr()} - Iteration {self.iteration} - {hours} Hours and {minutes} minutes in"
        )
        
        self.console.clear()
        self.console.print(mainPanel)
        self.lastRendered = time.time()

        if save is not None:
            # make directory containing save file:
            os.makedirs(os.path.dirname(save), exist_ok=True)

            with open(save, "w") as f:
                c = Console(file=f)
                c.print(mainPanel)




if __name__ == "__main__":
    d = DashBoard()
    d.render()
    time.sleep(5)
    d.updateProofAttemptSuccessRateGraph("Hello ","World!")
    d.render()