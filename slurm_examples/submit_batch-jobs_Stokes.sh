#!/bin/bash

#START_INDEX=1
#END_INDEX=6
#START_INDEX=7
#END_INDEX=10
#START_INDEX=11
#END_INDEX=50
START_INDEX=1
END_INDEX=6

RUNS_DIR="Runs"
SCRIPT_NAME="Grafted_HPS1_individual_run_Stokes.sh"

for i in $(seq $START_INDEX $END_INDEX); do

    RUN_DIR="$RUNS_DIR/$i"

    if [ -d "$RUN_DIR" ]; then
        echo "Submitting job for $RUN_DIR"

        cd "$RUN_DIR" || exit 1
        sbatch "$SCRIPT_NAME"
        cd - > /dev/null

    else
        echo "Directory $RUN_DIR does not exist. Skipping..."
    fi

done

echo "All jobs submitted."