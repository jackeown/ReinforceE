from e_caller import ECallerHistory
import argparse
from glob import glob
import IPython
from rich import print
from rich.panel import Panel





if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("run")
    # parser.add_argument("--reported", action="store_true")
    # args = parser.parse_args()

    # procCounts = []
    # for run in [f"ECallerHistory/{args.run}{i}" for i in range(5)]:
    #     hist = ECallerHistory.load(run)
    #     infos_lists = list(hist.history.values())
    #     infos = sum(infos_lists, [])

    #     key = 'processed_count' + ("_reported" if args.reported else "")
    #     procCounts.append({info['problem']:info[key] for info in infos})

    # IPython.embed()
