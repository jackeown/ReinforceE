#!/bin/bash

while true; do
    python scripts/results/checkProcCounts.py "$@"
    sleep 600 # 10 minutes
    # sleep 1200 # 20 minutes
done
