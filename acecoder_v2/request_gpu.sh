# for actual training
srun --account=aip-wenhu --gres=gpu:h100:8 --time=48:00:00 --job-name=wyett_acecoder_v2 -c 16 --mem=300GB --pty bash

# for testing
srun --account=aip-wenhu --gres=gpu:l40s:4 --time=12:00:00 --job-name=wyett_acecoder_v2_test -c 8 --mem=300GB --pty bash