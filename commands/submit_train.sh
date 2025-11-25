#!/bin/bash

# Sequential job submission script for multi-node training
script_path=${1:-examples/train/acetoolreason/train_multi_node.sh}
nodes=${2:-8}
num_submits=${3:-1}
dep_jid=$4

job_name=$(basename $script_path .sh)

echo "Using $nodes nodes for submission"
echo "Submitting $num_submits sequential jobs"

COMMON_ARGS="--parsable --nodes=$nodes --account=llmservice_fm_post --job-name=$job_name"

# Submit the first job
if [ -n "$dep_jid" ]; then
    echo "Submitting first job with dependency on job $dep_jid"
    jid=$(sbatch $COMMON_ARGS --dependency=afterany:$dep_jid $script_path)
else
    echo "Submitting first job without dependencies"
    jid=$(sbatch $COMMON_ARGS $script_path)
fi
echo "Submitted job 1 with Job ID: $jid"

# Submit remaining jobs with dependencies
for i in $(seq 2 $num_submits); do
    echo "Submitting job $i with dependency on job $jid"
    jid=$(sbatch $COMMON_ARGS --dependency=afterany:$jid $script_path)
    echo "Submitted job $i with Job ID: $jid"
done

echo "All $num_submits jobs submitted successfully!"


# Usage:
# bash commands/submit_train.sh examples/train/acetoolreason/train_multi_node_math_rl_no_tool.sh 8 4