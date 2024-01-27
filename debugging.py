import argparse
import psutil

# 1.) Interactively select the process ID of the process you want to debug
#     using the toSearch substring.
# 2.) List all open files for the selected process.


def FindProcesses(substring):
    matching_processes = []
    for process in psutil.process_iter(['pid', 'name']):
        if substring.lower() in process.info['name'].lower():
            matching_processes.append(process.info)
    return matching_processes


def ListOpenFiles(pid):
    if pid is None:
        print("No process found")
        return
    
    print("Listing open files for process %s" % pid)
    with open("/proc/%s/maps" % pid, "r") as f:
        for line in f:
            print(line)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Debugging')
    parser.add_argument('toSearch', help="substring to help find a relevant process")
    args = parser.parse_args()

    procs = FindProcesses(args.toSearch)
    for proc in procs:
        print(f"Process {proc['pid']} - {proc['name']}")
        ListOpenFiles(proc['pid'])