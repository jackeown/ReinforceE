import sys, os
sys.path.append('.')
from e_caller import ECallerHistory
import matplotlib.pyplot as plt
from rich.progress import track
import numpy as np

from scipy.stats import expon
from scipy.stats import gaussian_kde
from rich import print


def geomean(l):
    """Geometric mean"""
    return np.exp(np.mean(np.log(l)))

def mean(l):
    return sum(l) / len(l)

def median(l):
    l = sorted(l)
    if len(l) % 2 == 1:
        return l[len(l)//2]
    else:
        return mean(l[len(l)//2 - 1 : len(l)//2 + 1])


def procCountDist(hists, pngPath, probSet=None):
    """Maps an ECallerHistory hist to a histogram with overlaid gaussian estimate of the distribution
        of the number of clauses processed in each successful proof.
    """
    fig, ax = plt.subplots()
    ax2 = ax.twinx()
    ax.set_title(f"Clauses Processed in Successful Proofs")
    ax.set_xlim(0, 6000)

    reds = ['red', 'tomato', 'orange', 'orangered', 'darkred', 'maroon']
    greens = ['green', 'lime', 'lawngreen', 'forestgreen', 'darkgreen', 'olive']
    blues = ['blue', 'dodgerblue', 'deepskyblue', 'royalblue', 'navy', 'midnightblue']

    n = 1
    colors = reds[:n] + greens[:n] + blues[:n]
    for i, (run, hist) in enumerate(hists.items()):
        color = colors[i % len(colors)]
        simpleColor = ['red','green','blue'][(i // n) % n]

        # flatten the values of hist.history
        if probSet is not None:
            allInfos = [info for infos in hist.history.values() for info in infos if info['problem'] in probSet]
        else:
            allInfos = [info for infos in hist.history.values() for info in infos]

        keep = lambda info: info['solved']
        allProcCounts = [info['processed_count'] for info in allInfos if keep(info)]


        # make histogram:
        # ax.hist(allProcCounts, bins=50, density=False, alpha=0.4, color=color)

        #make CDF curve:
        ax2.plot(np.sort(allProcCounts), np.linspace(0, 1, len(allProcCounts)), color=color, label=run, alpha=0.5, linewidth=1)

        # add gaussian estimate:
        mu, sigma = np.mean(allProcCounts), np.std(allProcCounts)
        # x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
        # ax.plot(x, 1/(sigma * np.sqrt(2 * np.pi)) * np.exp( - (x - mu)**2 / (2 * sigma**2) ), linewidth=2, color=color, label=run, alpha=0.4)
        print(f"[{simpleColor}]{run}[/{simpleColor}]: mu = {mu:.2f}, sigma = {sigma:.2f}")

        # Plot the PDF on the histogram
        # ax.plot(x, pdf_fitted, color=color, label=f'{run}', alpha=0.5)

        # Scatter plot for processed_count
        dotSize = 0.00004
        y_value_for_scatter = [-dotSize * (i+1)] * len(allProcCounts)  # Y-axis values for scatter plot
        ax.scatter(allProcCounts, y_value_for_scatter, alpha=0.06, color=color, s=150, marker='.', edgecolors='none') # larger points in scatterplot by using the scatter parameter 's 


        # compute the KDE
        ax.set_ylim(-dotSize*(len(hists)+1), 0.0019)
        # points = allProcCounts
        points = [x for x in allProcCounts if x > 1]
        kde = gaussian_kde(points, bw_method=0.2)
        kde_x = np.linspace(min(allProcCounts), max(allProcCounts), 2000)
        kde_y = kde(kde_x)

        # Plot the KDE
        ax.plot(kde_x, kde_y, color=color, label=f'{run}', alpha=0.5, linewidth=1)

        # Also print median and 25th, 75th percentiles and geometric mean
        withoutZeros = [x for x in allProcCounts if x != 0]
        print(f"[{simpleColor}]{run}[/{simpleColor}]: median = {median(withoutZeros):.2f}, 25th = {np.percentile(allProcCounts, 25):.2f}, 75th = {np.percentile(allProcCounts, 75):.2f}, geometric mean = {geomean(withoutZeros):.2f}")
        print("-"*100)
        if i%3 == 2:
            print("\n" + "-"*100)


        # plot medians as vertical lines:
        if probSet is not None:
            ax2.axvline(x=mean(allProcCounts), color=color, alpha=0.5, linewidth=1)


        # add legend:
        ax.legend(loc='center right')

        # ensure path exists:
        os.makedirs(os.path.dirname(pngPath), exist_ok=True)

    # save png:
    plt.savefig(pngPath, dpi=1000)

    



if __name__ == "__main__":
    runs = sys.argv[1:]


    from rich.console import Console
    console = Console(force_terminal=True)
    print = console.print

    hists = {run: ECallerHistory.load(run, keysToDeleteFromInfos=['stdout', 'stderr', 'states', 'actions', 'rewards']) for run in track(runs, description="Loading ECaller Histories")}
    hists = {key: val for key, val in hists.items() if len([info for infos in val.history.values() for info in infos if info['solved']]) > 0}
    # for run, hist in track(hists.items(), description="Generating Histograms"):
    #     procCountDist({run: hist}, f"/home/jack/Desktop/procCounts/{run}.png")
    
    probsSolved = {run: set([info['problem'] for infos in hist.history.values() for info in infos if info['solved']]) for run, hist in hists.items()}
    probSet = set.intersection(*[probsSolved[run] for run in hists.keys()]) # intersection of all sets
    
    print(f"{len(probSet)} problems solved in all runs.")
    print({k: len(v) for k, v in probsSolved.items()})

    from glob import glob
    COUNT = max([int(x.split("/")[-1].split(".")[0]) for x in glob("/home/jack/Desktop/procCounts/*.png")] + [-1]) # largest i.png file found using glob    
    procCountDist(hists, f"/home/jack/Desktop/procCounts/{COUNT+1}.png", probSet=probSet)
    



