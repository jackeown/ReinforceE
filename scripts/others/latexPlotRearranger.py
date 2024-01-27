# This is meant to be used with graphs.sh
# This script takes in a file containing latex code
# for many tables.
# The tables are separated by "\n\n" and 
# we would like to output a file with the same tables
# rearranged.
import os,sys

def keyFunc(tableText):
    metrics = ["Problems Solved", "Processed Clauses", "Generated Clauses", "Proof Clauses", "Efficiency"]
    datasets = ["MPTPTP2078", "VBT", "SLH-29"]

    # What is the index of the first metric that is in tableText?
    try:
        assert [m in tableText for m in metrics].count(True) == 1, "Not exactly one metric in tableText"
    except:
        print("tabletext: ", tableText)

    metricIndex = [i for i,m in enumerate(metrics) if m in tableText][0]

    # What is the index of the first dataset that is in tableText?
    assert [d in tableText for d in datasets].count(True) == 1, "Not exactly one dataset in tableText"
    datasetIndex = [i for i,d in enumerate(datasets) if d in tableText][0]

    return 10*metricIndex + datasetIndex

if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("Usage: python latexPlotRearranger.py <latex code>")
        exit(1)
    
    # 1.) Read tables from file
    with open(sys.argv[1]) as inFile:
        tables = inFile.read().split("\n\n")[:-1]
    
    # 2.) Split and rearrange tables
    tables = sorted(tables, key=keyFunc)

    # 3.) Output to file
    with open(sys.argv[2], 'w') as outFile:
        # outFile.write("\n\n".join(tables))

        # Between every 3 tables, insert a page break
        for i,table in enumerate(tables):
            outFile.write(table)
            outFile.write("\n\n")
            if i % 3 == 2:
                outFile.write("\newpage\n\n")