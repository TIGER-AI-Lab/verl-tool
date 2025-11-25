# scaling turns to 1000 steps, 7b 100s/step, 1.5b 70s/step
bash commands/submit_train.sh examples/train/ablation/scaling_turns/math_tir_7b_grpo_1_turns.sh 2 6
bash commands/submit_train.sh examples/train/ablation/scaling_turns/math_tir_7b_grpo_5_turns.sh 2 6
bash commands/submit_train.sh examples/train/ablation/scaling_turns/math_tir_7b_grpo_5_turns_mtrl.sh 2 6
bash commands/submit_train.sh examples/train/ablation/scaling_turns/math_tir_7b_grpo_10_turns.sh 2 6

bash commands/submit_train.sh examples/train/ablation/scaling_turns/math_tir_1.5b_grpo_1_turns.sh 2 6
bash commands/submit_train.sh examples/train/ablation/scaling_turns/math_tir_1.5b_grpo_5_turns.sh 2 6
bash commands/submit_train.sh examples/train/ablation/scaling_turns/math_tir_1.5b_grpo_5_turns_mtrl.sh 2 6
bash commands/submit_train.sh examples/train/ablation/scaling_turns/math_tir_1.5b_grpo_10_turns.sh 2 6


# scaling turns on different datasets

bash commands/submit_train.sh examples/train/ablation/scaling_turns/deepscaler/math_tir_1.5b_grpo_5_turns.sh 2 6
bash commands/submit_train.sh examples/train/ablation/scaling_turns/dapo/math_tir_1.5b_grpo_5_turns.sh 2 6

bash commands/submit_train.sh examples/train/ablation/scaling_turns/deepscaler/math_tir_7b_grpo_5_turns.sh 2 6
bash commands/submit_train.sh examples/train/ablation/scaling_turns/dapo/math_tir_7b_grpo_5_turns.sh 2 6

# simpletir experiments

bash commands/submit_train.sh examples/train/ablation/simpletir/multi_node_7b_grpo_5_turns.sh 2 8
bash commands/submit_train.sh examples/train/ablation/simpletir/multi_node_7b_grpo_5_turns_no_filtering.sh 2 8

# sql retokenization experiments

bash commands/submit_train.sh examples/train/ablation/tokenization/multi_node_sql_retokenization.sh 1 4
bash commands/submit_train.sh examples/train/ablation/tokenization/multi_node_sql.sh 1 4

# 