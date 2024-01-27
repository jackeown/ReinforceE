# Simple wrapper around runE function from main.py so that it's easy to run E with a learned policy...
import argparse
import torch
from main import runE, extractCEF_no
from policy_grad import PolicyNetConstCategorical, PolicyNet
import IPython


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("problem")
    parser.add_argument("--eprover_path", default="eprover_RL_HIST")
    parser.add_argument("--policy", default="")
    parser.add_argument("--cpu_limit", default=300, type=int)
    parser.add_argument("--create_info", action="store_true")
    parser.add_argument("--dryRun", action="store_true")
    parser.add_argument("--strat_file", default="")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--state_dim", default=59, type=int)
    args = parser.parse_args()

    try:
        policy = torch.load(args.policy)
        print(f"Policy loaded: {args.policy}")
    except Exception as e:
        print(f"Could not load policy: {args.policy}")
        n = extractCEF_no(args.strat_file)
        print("Num CEFS: ", n)
        # policy = PolicyNetConstCategorical(args.state_dim, n)
        policy = PolicyNet(args.state_dim, 100, n, 2)

    strat_file = args.strat_file if len(args.strat_file) else None
    info = runE(policy, args.eprover_path, args.problem, args.state_dim,
        max(args.cpu_limit-5, 5), args.cpu_limit,
        auto=False, create_info=args.create_info, verbose=args.verbose, dryRun=args.dryRun, strat_file=strat_file)
    
    if args.dryRun:
        print(info)

    if args.create_info:
        IPython.embed()
