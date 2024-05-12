# This takes in 3 files (one for each dataset: output by compareSolved.py) and regroups
# the sections so that each metric is together.
# These files will probably be in the latexTables dir.

import sys, os
import argparse
import IPython


def listsplit(lines):
    chunks = []
    
    chunk = []
    for line in lines:
        chunk.append(line)
        if line.strip() == "":
            chunks.append(chunk)
            chunk = []
    chunks.append(chunk)
    return chunks


def removeLinesContaining(chunk, s):
    return [line for line in chunk if s not in line]


def regroup(mpt,vbt,slh):
    """
    Takes in 3 lists of lines and returns
    num_metrics lists of lines.
    """

    # split each list on empty lines:
    mptSections = listsplit(mpt)
    vbtSections = listsplit(vbt)
    slhSections = listsplit(slh)
    slhSections = [removeLinesContaining(x, "minipage") for x in slhSections] # hacky

    metricGroups = []
    for mptSec, vbtSec, slhSec in zip(mptSections, vbtSections, slhSections):
        # Remove newline between mpt and vbt:
        mptSec = [s for s in mptSec if s.strip() != '']

        lines = mptSec + ['\\hfill%\n'] + vbtSec + ['\\vspace{1cm}\n'] + slhSec
        metricGroups.append(lines)
    return metricGroups


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mptFile")
    parser.add_argument("vbtFile")
    parser.add_argument("slhFile")
    parser.add_argument("outputDir")
    args = parser.parse_args()

    metrics = ['probsSolved', 'procClauses', 'genClauses', 'proofClauses', 'efficiency']

    with open(args.mptFile) as f:
        mpt = f.readlines()
    with open(args.vbtFile) as f:
        vbt = f.readlines()
    with open(args.slhFile) as f:
        slh = f.readlines()

    regrouped = regroup(mpt,vbt,slh)

    for metric, metricGroup in zip(metrics, regrouped):
        with open(f"{args.outputDir}/{metric}.tex", "w") as f:
            for line in metricGroup:
                f.write(line)

    