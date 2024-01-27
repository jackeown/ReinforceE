# This script shows a nice display of what percent of CPU / Memory / Disk / Shared memory / Network resources are used by each user.
# Doesn't work :(
from rich import print
import psutil
from rich.table import Table


def getProcesses():
    return list(psutil.process_iter())

def groupByUser(procs):
    grouped = {}
    for proc in procs:
        user = proc.username()
        if user not in grouped:
            grouped[user] = []
        grouped[user].append(proc)
    return grouped


def getUserResources(grouped):
    user_resources = {}
    for user, procs in grouped.items():
        cpu = 0
        memory = 0
        disk = 0
        shared_memory = 0
        network = 0
        for proc in procs:
            cpu += proc.cpu_percent()
            memory += proc.memory_percent()
            disk += proc.io_counters().read_bytes + proc.io_counters().write_bytes
            shared_memory += proc.memory_info().shared
            network += proc.io_counters().bytes_sent + proc.io_counters().bytes_recv
        user_resources[user] = {
            "cpu": cpu,
            "memory": memory,
            "disk": disk,
            "shared_memory": shared_memory,
            "network": network
        }
    return user_resources


def printUserResources(user_resources):
    table = Table(title="User Resources")
    table.add_column("User")
    table.add_column("CPU")
    table.add_column("Memory")
    table.add_column("Disk")
    table.add_column("Shared Memory")
    table.add_column("Network")
    for user, resources in user_resources.items():
        table.add_row(user, f"{resources['cpu']}%", f"{resources['memory']}%", f"{resources['disk']}B", f"{resources['shared_memory']}B", f"{resources['network']}B")
    print(table)




if __name__ == "__main__":

    # 1.) Get all running processes.
    # 2.) Group processes by user.
    # 3.) For each user, get the sum of CPU / Memory / Disk / Shared memory / Network resources used by all of their processes.

    procs = getProcesses()
    grouped = groupByUser(procs)
    user_resources = getUserResources(grouped)
    printUserResources(user_resources)