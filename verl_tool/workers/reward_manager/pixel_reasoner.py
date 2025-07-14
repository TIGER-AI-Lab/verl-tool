# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from .reward_score import _default_compute_score
from .reward_score.torl_math import compute_score as torl_compute_score
from verl.workers.reward_manager import register
from collections import defaultdict
from .torl import ToRLRewardManager

@register("pixel_reasoner")
class PixelReasonerRewardManager(ToRLRewardManager):
    """
    A reward manager for the Pixel Reasoner.
    It uses the TORL framework to compute rewards based on the outputs of the model.
    """
    name = "pixel_reasoner"
    def __init__(self, tokenizer, num_examine, compute_score=None, reward_fn_key='data_source') -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score if compute_score else _default_compute_score
        self.torl_compute_score = torl_compute_score
        self.reward_fn_key = reward_fn_key
        self.step = None
        self.add_format_think_penalty = False # -0.5 if not begines with <think> and end with </think>
        self.add_format_answer_penalty = False # -0.5 if not having <answer> </answer>
        self.add_valid_action_penalty = True # -0.25 if num turns > 0 not action not valid
        self.add_unfinished_traj_penalty = True # -0.25 if the traj is not finished
        self.add_no_tool_interact_penalty = False # -0.25 if the traj's num turn is 0, no interaction at all
        self.add_code_exec_penalty = False # -0.25 if the execution has an error.
