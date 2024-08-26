import numpy as np
import os, sys
sys.path.append('.')
from e_caller import ECallerHistory
from rich import print
from rich.progress import track
from rich.table import Table
import matplotlib.pyplot as plt
from copy import copy
import IPython

# import geometric mean:
from scipy.stats.mstats import gmean as geomean

# math helpers
mean = lambda l: sum(l) / len(l)

def stripAndCamelCase(s):
    """Remove whitespace and capitalize the first letter of 
    each word with the first letter of the first word lowercase"""
    words = s.split()
    words = [word.lower().capitalize() for word in words]
    words[0] = words[0].lower()
    return "".join(words)


def track(*args, **kwargs):
    return args[0]


def median(l):
    l = sorted(l)
    return l[len(l)//2] if len(l)%2 == 1 else mean(l[len(l)//2 - 1 : len(l)//2 + 1])

def medianOfNonZero(l):
    nonZero = [x for x in l if x != 0]
    return median(nonZero) if len(nonZero) > 0 else -1_000_000

def smartMean(l, low=1, high=90):
    """Return mean of data between 2 percentiles"""
    l = sorted(l)
    lowIndex, highIndex = int(len(l)*low/100), int(len(l)*high/100)
    return mean(l[lowIndex:highIndex])

def logMean(l):
    l = list(l)
    return np.exp(mean(np.log(l)))


def getGeneratedCounts(history):
    generatedCounts = {}
    for prob in history.history.keys():
        infos = history.history[prob]
        stdouts = [info['stdout'] for info in infos if info['solved']]
        generatedLines = [line for line in '\n'.join(stdouts).split("\n") if "# Generated clauses" in line]
        if generatedLines:
            generatedCounts[os.path.split(prob)[1]] = mean([int(line.split(":")[1].strip()) for line in generatedLines])
    return generatedCounts

def getProofClauseCount(info):
    stdout = info['stdout']
    for line in stdout.split("\n"):
        if "# Proof object given clauses" in line:
            return int(line.split(":")[1].strip())
    return 0

def getProofClauseCounts(history):
    proofClauseCounts = {}
    for prob in history.history.keys():
        # counts = [info['stdout'].count("# trainpos") for info in history.history[prob] if info['solved']]
        counts = [getProofClauseCount(info) for info in history.history[prob] if info['solved']]
        if counts:
            proofClauseCounts[os.path.split(prob)[1]] = mean(counts)
    return proofClauseCounts

def getProcCounts(history):
    procCounts = {os.path.split(prob)[1]: history.getProcCounts(prob) for prob in history.history.keys()}
    procCounts = {k:mean(v) for k,v in procCounts.items() if len(v)}
    return procCounts

def compareSolved(solved1, solved2):
    return len(set(solved1) - set(solved2)), len(set(solved2) - set(solved1))

def compareProcessed(processed1, processed2):
    """Over the intersection of their keys, 
    return the mean/median/geomean of the difference 
    between processed1 and processed2."""
    return mean([processed1[k] - processed2[k] for k in processed1 if k in processed2])

def compareGenerated(generated1, generated2):
    """Over the intersection of their keys, 
    return the mean/median/geomean of the difference 
    between generated1 and generated2."""
    return mean([generated1[k] - generated2[k] for k in generated1 if k in generated2])

def universallySolvedProbs(hists):
    """Return the set of problems that were solved by all methods."""
    return set.intersection(*[set(getProcCounts(hists[key]).keys()) for key in hists])


def buildSmallerTables(hists, subFunc=lambda x: x):
    """
    Builds tables for:
    1. Problems solved (and vs auto) by all methods
    2. Processed clauses (and vs auto) for problems solved by all methods
    3. Generated clauses (and vs auto) for problems solved by all methods
    4. Proof clauses for problems solved by all methods
    5. Efficiency (proof clauses / processed clauses )for problems solved by all methods

    Sorts each table by the relevant column.
    For instance, sort the generated clauses table by generated clauses.
    """

    # General Setup:
    autoKey = [key for key in sys.argv[1:] if "auto" in key.lower() and not "sched".lower() in key][0]
    generated = {key: getGeneratedCounts(hists[key]) for key in hists}
    proofClauseCounts = {key: getProofClauseCounts(hists[key]) for key in hists}
    procCounts = {key: getProcCounts(hists[key]) for key in hists}
    proofClauseCountsPerProc = {key: {k: proofClauseCounts[key][k] / procCounts[key][k] for k in procCounts[key] if procCounts[key][k]} for key in hists}
    probsSolvedByAll = universallySolvedProbs(hists)

    tables = {
        "Problems Solved": Table(title="Problems Solved", show_header=True, header_style="bold magenta"),
        "Processed Clauses": Table(title="Processed Clauses", show_header=True, header_style="bold magenta"),
        "Generated Clauses": Table(title="Generated Clauses", show_header=True, header_style="bold magenta"),
        "Proof Clauses": Table(title="Proof Clauses", show_header=True, header_style="bold magenta"),
        "Efficiency": Table(title="Efficiency", show_header=True, header_style="bold magenta"),
    }

    ###### Probs Solved and Probs Solved vs Auto: ##############################################
    for col in ["Run", "Solved", "vs Auto"]:
        tables["Problems Solved"].add_column(col)

    rows = []
    # Looping over runs
    for key in track(sys.argv[1:], description="Building Solved Table"):
        processed = procCounts[key]
        hist = hists[key]

        numSolved = 0
        probsSolved = set()
        for prob,infos in hist.history.items():
            percentSolved = mean([1 if info['solved'] else 0 for info in infos])
            numSolved += percentSolved
            if percentSolved > 0.5:
                probsSolved.add(prob)

        # solved = list(processed.keys())
        solvedVSAuto = compareSolved(probsSolved, set(procCounts[autoKey].keys()))
        rows.append([
            f"[white]{subFunc(key)}[/white]",
            f"{numSolved:.2f}",
            f"[red]+{solvedVSAuto[0]} , -{solvedVSAuto[1]}[/red]",
        ])
    
    rows = sorted(rows, key=lambda x: x[1], reverse=True)
    for run, solved, solvedVSAuto in rows:
        tables["Problems Solved"].add_row(run, f"[cyan]{solved}[/cyan]", solvedVSAuto)

    ###### Processed Clauses and Processed Clauses vs Auto: ##############################################
    for col in ["Run", "median", "mean"]:    
        tables["Processed Clauses"].add_column(col)

    rows = []
    for key in track(sys.argv[1:], description="Building Processed Table"):
        processed = procCounts[key]
        # processedVSAuto = compareProcessed(processed, procCounts[autoKey])
        try:
            rows.append([
                subFunc(key),
                centralTendency([processed[k] for k in probsSolvedByAll if k in processed]),
                mean([processed[k] for k in probsSolvedByAll if k in processed]),
            ])
        except ZeroDivisionError:
            print(f"Zero Division Error: {key}")
            IPython.embed()
    
    rows = sorted(rows, key=lambda x: x[1], reverse=False)
    for run, processed, processedVSAuto in rows:
        tables["Processed Clauses"].add_row(f"[white]{run}[/white]", f"[purple]{processed:.2f}[/purple]", f"[green]{processedVSAuto:.2f}[/green]")

    ###### Generated Clauses and Generated Clauses vs Auto: ##############################################
    for col in ["Run", "median", "mean"]:
        tables["Generated Clauses"].add_column(col)
    
    rows = []
    for key in track(sys.argv[1:], description="Building Generated Table"):
        generatedCounts = generated[key]
        # generatedVSAuto = compareGenerated(generatedCounts, generated[autoKey])
        rows.append([
            subFunc(key),
            centralTendency([generatedCounts[k] for k in probsSolvedByAll if k in generatedCounts]),
            mean([generatedCounts[k] for k in probsSolvedByAll if k in generatedCounts]),
        ])
    
    rows = sorted(rows, key=lambda x: x[1], reverse=False)
    for run, generated, generatedVSAuto in rows:
        tables["Generated Clauses"].add_row(f"[white]{run}[/white]", f"[blue]{generated:.2f}[/blue]", f"[orange]{generatedVSAuto:.2f}[/orange]")


    ###### Proof Clauses and Proof Clauses vs Auto: ######################################################
    for col in ["Run", "median", "mean"]:
        tables["Proof Clauses"].add_column(col)

    rows = []
    for key in track(sys.argv[1:], description="Building Proof Clause Table"):
        proof_clause_counts = proofClauseCounts[key]
        # proof_clause_counts_vs_auto = compareGenerated(proof_clause_counts, proofClauseCounts[autoKey])
        rows.append([
            subFunc(key),
            centralTendency([proof_clause_counts[k] for k in probsSolvedByAll if k in proof_clause_counts]),
            mean([proof_clause_counts[k] for k in probsSolvedByAll if k in proof_clause_counts]),
        ])
    
    rows = sorted(rows, key=lambda x: x[2], reverse=False)
    for a,b,c in rows:
        tables["Proof Clauses"].add_row(f"[white]{a}[/white]", f"[yellow]{b:.2f}[/yellow]", f"[magenta]{c:.2f}[/magenta]")
    

    ###### Efficiency: ###################################################################################
    for col in ["Run", "Efficiency"]:
        tables["Efficiency"].add_column(col)
    
    rows = []
    for key in track(sys.argv[1:], description="Building Efficiency Table"):
        processed = procCounts[key]
        proof_clause_counts = proofClauseCounts
        efficiency = centralTendency([proofClauseCountsPerProc[key][k] for k in probsSolvedByAll if k in proofClauseCountsPerProc[key]])
        rows.append([
            subFunc(key),
            efficiency,
        ])
    
    rows = sorted(rows, key=lambda x: x[1], reverse=True)
    for run, efficiency in rows:
        tables["Efficiency"].add_row(f"[white]{run}[/white]", f"[magenta]{efficiency:.3f}[/magenta]")
    
    return tables





def buildTable(hists):
    table = Table(title="Methods Compared", show_header=True, header_style="bold magenta")
    for col in ["Run", "Solved", "Solved vs Auto", "Processed", "Processed vs Auto", "Generated", "Generated vs Auto", "Proof Clauses", "Efficiency"]:
        table.add_column(col)
    
    autoKey = [key for key in sys.argv[1:] if "auto" in key.lower() and not "sched".lower() in key][0]
    generated = {key: getGeneratedCounts(hists[key]) for key in hists}
    proofClauseCount = {key: getProofClauseCounts(hists[key]) for key in hists}
    procCounts = {key: getProcCounts(hists[key]) for key in hists}
    proofClauseCountPerProc = {key: {k: proofClauseCount[key][k] / procCounts[key][k] for k in procCounts[key] if procCounts[key][k]} for key in hists}
    probsSolvedByAll = universallySolvedProbs(hists)

    rows = []
    for key in track(sys.argv[1:], description="Building Table"):
        processed = getProcCounts(hists[key])
        solved = {k for k in processed}
        solvedVSAuto = compareSolved(solved, set(getProcCounts(hists[autoKey]).keys()))
    
        divisor = 5 if CV else 1
        rows.append([
            key,
            len(solved) / divisor,
            f"+{solvedVSAuto[0] / divisor: >5} , -{solvedVSAuto[1] / divisor: >5}",
            centralTendency(processed.values()),
            compareProcessed(processed, procCounts[autoKey]),
            centralTendency([generated[key][k] for k in probsSolvedByAll if k in generated[key]]),
            compareGenerated(generated[key], generated[autoKey]),
            centralTendency([proofClauseCount[key][k] for k in probsSolvedByAll if k in proofClauseCount[key]]),
            centralTendency([proofClauseCountPerProc[key][k] for k in probsSolvedByAll if k in proofClauseCountPerProc[key]]),
        ])
    
    rows = sorted(rows, key=lambda x: x[1], reverse=True)

    for a,b,c,d,e,f,g,h,i in rows:
        table.add_row(
            f"[white]{a}[/white]",
            f"[cyan]{b}[/cyan]",
            f"[red]{c}[/red]",
            f"[purple]{d:.2f}[/purple]",
            f"[green]{e:.2f}[/green]",
            f"[blue]{f:.2f}[/blue]",
            f"[orange]{g:.2f}[/orange]",
            f"[yellow]{h:.2f}[/yellow]",
            f"[magenta]{i:.9f}[/magenta]",
        )
    
    return table


def generatedVis(hists, output_filename, xrange=None, yrange=None, normalized=False, subFunc=lambda x: x, dataset=""):
    plt.clf()

    allXs, allYs = [], []
    for histName, hist in hists.items():
        generatedCounts = getGeneratedCounts(hist)
        if normalized:
            xs, ys = sorted(generatedCounts.values()), np.linspace(0,1,len(generatedCounts))
        else:
            xs, ys = sorted(generatedCounts.values()), list(range(len(generatedCounts)))
            
        allXs.extend(xs)
        allYs.extend(ys)
        plt.plot(xs,ys, label=f"{subFunc(histName)}")

    if xrange:
        plt.xlim(xrange)
    else:
        low,high = np.percentile(allXs, [1,99])
        plt.xlim((low - (high-low)*0.05, high + (high-low)*0.05))
    
    if yrange:
        plt.ylim(yrange)
    else:
        low,high = np.percentile(allYs, [20,99])
        # replace low with smallest y value for which x is greater than 1% of the way to xrange[1]
        onePercent = xrange[1] / 100
        _, low = min([(x,y) for x,y in zip(allXs, allYs) if x > onePercent], key = lambda x: x[1])
        plt.ylim((low - (high-low)*0.05, high + (high-low)*0.1))


    dataset = f"{dataset} - " if dataset else ""
    plt.title(dataset + "Generated Clause Distribution" if normalized else "Y problems solved with X generated clauses")
    plt.xlabel("Generated Clauses")
    plt.ylabel("CDF" if normalized else "Problems Solved")
    plt.legend()
    plt.savefig(output_filename, dpi=600)

def processedVis(hists, output_filename, xrange=None, yrange=None, normalized=False, subFunc=lambda x: x, dataset=""):
    plt.clf()

    allXs = []
    allYs = []

    for histName, hist in hists.items():
        processedCounts = getProcCounts(hist)
        numSolved = len(processedCounts)
        if normalized:
            xs, ys = sorted(processedCounts.values()), np.linspace(0,1,numSolved)
        else:
            xs, ys = sorted(processedCounts.values()), list(range(numSolved))
            
        allXs.extend(xs)
        allYs.extend(ys)
        plt.plot(xs,ys, label=f"{subFunc(histName)}")
    
    if xrange:
        plt.xlim(xrange)
    else:
        low,high = np.percentile(allXs, [1,99])
        plt.xlim((low - (high-low)*0.05, high + (high-low)*0.05))
    
    if yrange:
        plt.ylim(yrange)
    else:
        low,high = np.percentile(allYs, [1,99])
        # replace low with smallest y value for which x is greater than 1% of the way to xrange[1]
        onePercent = xrange[1] / 100
        _, low = min([(x,y) for x,y in zip(allXs, allYs) if x > onePercent], key = lambda x: x[1])

        plt.ylim((low - (high-low)*0.05, high + (high-low)*0.1))

    dataset = f"{dataset} - " if dataset else ""
    plt.title(dataset + "Processed Clause Distribution" if normalized else "Y problems solved with X processed clauses")
    plt.xlabel("Processed Clauses")
    plt.ylabel("CDF" if normalized else "Problems Solved")
    plt.legend()
    plt.savefig(output_filename, dpi=600)

def multi_lcs(words):
    words = copy(words)
    words.sort(key=lambda x:len(x))
    search = words.pop(0)
    s_len = len(search)
    for ln in range(s_len, 0, -1):
        for start in range(0, s_len-ln+1):
            cand = search[start:start+ln]
            for word in words:
                if cand not in word:
                    break
            else:
                return cand
    return False


def getTableRows(table):
    """
    table.rows stupidly doesn't work, so this is a workaround.
    """
    
    columns = [col._cells for col in table.columns] # get content from column._cells
    rows = list(zip(*columns)) # transpose to get rows

    return rows



def tableToHTML(table):
    # return html string version of the input rich table

    tableHTML = f"<table style=\"font-size:0.5em;\">\n"
    tableHTML += f"<tr>\n"
    for col in table.columns:
        tableHTML += f"<th>{col.header}</th>\n"
    tableHTML += f"</tr>\n"

    for row in getTableRows(table):
        tableHTML += f"<tr>\n"
        for cell in row:
            tableHTML += f"<td>{cell}</td>\n"
        tableHTML += f"</tr>\n"

    tableHTML += f"</table>\n"

    return tableHTML



def boldMine(row):
    criteria = lambda x: ("NeuralNet" in x) or ("ConstCat" in x)
    return [r'\textbf{' + x + r'}' if criteria(x) else x for x in row]

# GPT4 says this will match the phd formatting I used.
def tableToLaTeX(table, caption, label, dataset=None):
    """Converts a rich table to LaTeX format wrapped in a minipage environment.
    The input table is expected to have column names as headers.
    """
    
    # Beginning of Minipage and Table Boilerplate
    tableLaTeX = "\\begin{minipage}{0.48\\textwidth}\n"
    tableLaTeX += "\\centering\n"
    columnFormatting = 'l' + 'r' * (len(table.columns) - 1)  # Assuming 'l' for the first column and 'r' for the rest
    tableLaTeX += f"\\begin{{tabular}}{{{columnFormatting}}}\n"
    
    # Header row: Column names
    tableLaTeX += "\\toprule\n"
    tableLaTeX += f"{' & '.join([col.header for col in table.columns])} \\\\\n"
    tableLaTeX += "\\midrule\n"
    
    # Actual data rows
    for row in getTableRows(table):  # Assuming getTableRows is a function that returns the rows of the table
        tableLaTeX += f"{' & '.join(boldMine(row))} \\\\\n"
    
    # End of Table Boilerplate
    tableLaTeX += "\\bottomrule\n"
    tableLaTeX += "\\end{tabular}\n"
    
    # Label and Caption using \captionof for compatibility with minipage
    tableLaTeX += f"\\captionof{{table}}{{{caption}}}\n"
    tableLaTeX += f"\\label{{{label}}}\n"
    tableLaTeX += "\\end{minipage}%\n"  # The '%' at the end helps to avoid unwanted spaces after the minipage if used in a sequence
    
    return tableLaTeX









if __name__ == "__main__":


    if "--median" in sys.argv:
        centralTendency = median
    elif "--mean" in sys.argv:
        centralTendency = mean
    elif "--logMean" in sys.argv:
        centralTendency = logMean
    else:
        centralTendency = medianOfNonZero
    
    sys.argv = [x for x in sys.argv if x not in ["--median", "--mean", "--logMean"]]

    if "--cv" in sys.argv:
        CV = True
        sys.argv = [x for x in sys.argv if x != "--cv"]

        hists = {}
        for histName in track(sys.argv[1:], description="Loading Histories"):
            hists[histName] = ECallerHistory.load(f"{histName}0")
            hists[histName].merge([ECallerHistory.load(f"{histName}{i}") for i in range(1,5)])
    else:        
        # if --num=a is in an argv, then add a to the rest of the sys.argv args:
        if any([x.startswith("--num=") for x in sys.argv]):
            num = [x for x in sys.argv if x.startswith("--num=")][0].split("=")[1]
            sys.argv[1:] = [f"{x}{num}" for x in sys.argv[1:] if not x.startswith("--num=")]

        CV = False
        hists = {x: ECallerHistory.load(x) for x in track(sys.argv[1:], description="Loading Histories")}
    
    if "--iwil" in sys.argv:
        iwil = True
        sys.argv = [x for x in sys.argv if x != "--iwil"]
    else:
        iwil = False

    if "--ijait" in sys.argv:
        ijait = True
        sys.argv = [x for x in sys.argv if x != "--ijait"]
    else:
        ijait = False



    sub = {
        "NN": r"\_NeuralNet",
        "ConstCat": r"\_ConstCat",
        "ConstCatDistilled_gain5_": r"\_Distilled",
        "Auto": r"\_Auto",
        "AutoSched": r"\_AutoSched",
        "RoundRobin": r"\_MasterWeighted",
        "RoundRobinAllOnes": r"\_MasterAllOnes",
        "RoundRobinIncremental": r"\_MasterIncremental",
        "SuccessRoundRobin": r"\_MasterSuccess",
        
        "ConstCatAudit": r"\_ConstCatAudit",
        "NNAudit": r"\_NNAudit",

        "NNMem": r"\_Mem",
        "NNMem1e3Ent": r"\_Mem1e3Ent",
        "NNMem5e3Ent": r"\_Mem5e3Ent",
        "NNMem5e5Ent": r"\_Mem5e5Ent",

        "NN1Hist": r"\_NeuralNet",
        "ConstCat1Hist": r"\_ConstCat",
        "CommonHeuristic": r"\_CommonHeuristic",
        "CommonElse": r"\_CommonElse",

        "MasterRR": r"\_MasterWeightedRR",
        "StratOmnipotent": r"\_AutoAll",

        "SchedMerge": r"\_SchedMerge",
    }

    def subFunc(runName):
        withoutPrefix = runName[3:]
        return (runName[:3] + sub[withoutPrefix]) if withoutPrefix in sub else runName

    def subFunc2(runName):
        """This one removes the prefix including the backslash and underscore from subFunc1."""
        return subFunc(runName)[5:]




    # largest substring present in all hists:
    dataset = multi_lcs(sys.argv[1:])
    # print(f"Dataset: {dataset}")

    tables = buildSmallerTables(hists, subFunc2)

    def keyFunc(stuff):
        histName, (key,table) = stuff
        majorOrder = ["Problems Solved", "Processed Clauses", "Generated Clauses", "Proof Clauses", "Efficiency"].index(key)
        minorOrder = 0 if histName.startswith("MPT") else 1 if histName.startswith("VBT") else 2 if histName.startswith("SLH") else 3
        return 10*majorOrder + minorOrder

    for histName, (key,table) in sorted(zip(sys.argv[1:], tables.items()), key=keyFunc):
        if histName.startswith("MPT"):
            print(tableToLaTeX(table, caption=f"M'2078 {key.lower()}", label=f"mpt{stripAndCamelCase(key)}Table",dataset=f"MPTPTP2078 {key}"))
        elif histName.startswith("VBT"):
            print(tableToLaTeX(table, caption=f"VBT {key.lower()}", label=f"vbt{stripAndCamelCase(key)}Table",dataset=f"VBT {key}"))
        elif histName.startswith("SLH"):
            print(tableToLaTeX(table, caption=f"SLH {key.lower()}", label=f"slh{stripAndCamelCase(key)}Table",dataset=f"SLH-29 {key}"))
        else:
            print("UNKNOWN Dataset ", histName)


    if ijait:
        venue = "_IJAIT"
    elif iwil:
        venue = "_IWIL"
    else:
        venue = ""

    # ----------------------------------------------------------------------------------------------------------------
    # print("Making Generated Clauses Figures...")
    path = f"figures/dists/{dataset}/generated{venue}.png"
    os.makedirs(os.path.split(path)[0], exist_ok=True)
    # print("hists: ",list(hists.items()))
    generatedVis(hists, path, normalized=True, subFunc=subFunc2, dataset=dataset, xrange=[0,1_000_000])

    path = f"figures/dists/{dataset}/generated_scaled{venue}.png"
    os.makedirs(os.path.split(path)[0], exist_ok=True)
    generatedVis(hists, path, normalized=False, subFunc=subFunc2, dataset=dataset, xrange=[0,1_000_000])


    # ----------------------------------------------------------------------------------------------------------------
    # print("Making Processed Clauses Figures...")
    path = f"figures/dists/{dataset}/processed{venue}.png"
    os.makedirs(os.path.split(path)[0], exist_ok=True)
    processedVis(hists, path, normalized=True, subFunc=subFunc2, dataset=dataset, xrange=[0,30_000])

    path = f"figures/dists/{dataset}/processed_scaled{venue}.png"
    os.makedirs(os.path.split(path)[0], exist_ok=True)
    processedVis(hists, path, normalized=False, subFunc=subFunc2, dataset=dataset, xrange=[0,30_000])

