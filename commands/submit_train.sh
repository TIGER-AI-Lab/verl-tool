#!/bin/bash

# Sequential job submission script for multi-node training
nodes=${1:-8}
num_submits=${2:-1}
dep_jid=$3

echo "Using $nodes nodes for submission"
echo "Submitting $num_submits sequential jobs"

COMMON_ARGS="--parsable --nodes=$nodes --account=llmservice_fm_post"

# Submit the first job
if [ -n "$dep_jid" ]; then
    echo "Submitting first job with dependency on job $dep_jid"
    jid=$(sbatch $COMMON_ARGS --dependency=afterany:$dep_jid examples/train/acetoolreason/train_multi_node.sh)
else
    echo "Submitting first job without dependencies"
    jid=$(sbatch $COMMON_ARGS examples/train/acetoolreason/train_multi_node.sh)
fi
echo "Submitted job 1 with Job ID: $jid"

# Submit remaining jobs with dependencies
for i in $(seq 2 $num_submits); do
    echo "Submitting job $i with dependency on job $jid"
    jid=$(sbatch $COMMON_ARGS --dependency=afterany:$jid examples/train/acetoolreason/train_multi_node.sh)
    echo "Submitted job $i with Job ID: $jid"
done

echo "All $num_submits jobs submitted successfully!"