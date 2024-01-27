import argparse
import subprocess
from glob import glob
from rich.progress import track
from rich import print
import os

# this script calls policyHeatmap.py to make a video 
# of how a policy heatmap evolves on the same problem
# over the course of training.


# def makeVideo(png_paths, output, framerate):
#     """Call ffmpeg to combine a list of pngs into a video"""

#     png_paths = " ".join(png_paths)
#     ffmpeg_cmd = f"ffmpeg -framerate {framerate} -i {png_paths} -c:v libx264 -pix_fmt yuv420p {output}"

#     print("Making video...")
#     print(ffmpeg_cmd)
#     subprocess.run(ffmpeg_cmd, shell=True)


def makeVideo(heatmap_dir, policy_glob, problem, output, framerate):
    """Create a video from a sequence of png files."""
    # Escape the policy name
    escaped_policy_glob = escapePath(policy_glob).replace("*", "%05d")

    # Generate the file pattern
    file_pattern = f"{heatmap_dir}/{escaped_policy_glob}_{problem}.png"

    # Generate the ffmpeg command with padding filter
    # This example assumes padding to the next even dimensions with black color. Adjust as needed.
    ffmpeg_cmd = f"ffmpeg -framerate {framerate} -i '{file_pattern}' -vf 'pad=ceil(iw/2)*2:ceil(ih/2)*2:color=black' -c:v libx264 -pix_fmt yuv420p {output}"

    # Execute the command
    print("Making video...")
    print(ffmpeg_cmd)
    subprocess.run(ffmpeg_cmd, shell=True)



def escapePath(s):
    return s.replace("/", "_")


def largestCommonSubstr(strs):
    if not strs:
        return ""
    shortest = min(strs, key=len)
    for i, char in enumerate(shortest):
        for other in strs:
            if other[i] != char:
                return shortest[:i]
    return shortest


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Creates a heatmap of the policy's CEF preferences over the course of a proof search")
    parser.add_argument("policy_glob", help="The policy name (i.e MPTNN1)")
    parser.add_argument("run", help="The run name (under ECallerHistory)")
    parser.add_argument("--problem", help="The problem name from the run that you want to visualize")
    parser.add_argument("--dataset", choices=["MPT","VBT","SLH"], required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip_images", action="store_true")
    args = parser.parse_args()

    policies = sorted(glob(args.policy_glob))
    print("Policies: ")
    print(policies)
    if not args.skip_images:

        if args.problem is not None:
            problem = f"--problem {args.problem}"
        else:
            problem = f"--seed {args.seed}"

        for policy in track(policies):
            subprocess.run(f"python scripts/graphs/policyHeatmap.py {policy} {args.run} {problem} --dataset {args.dataset}", shell=True)

    os.makedirs(f"figures/heatmaps/{args.dataset}/videos", exist_ok=True)
    policy = escapePath(largestCommonSubstr(policies))
    output = f"figures/heatmaps/{args.dataset}/videos/{args.run}_{policy}_{args.problem}.mp4"
    makeVideo(f"figures/heatmaps/{args.dataset}", args.policy_glob, args.problem, output, 12)

