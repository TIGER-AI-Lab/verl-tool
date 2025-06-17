from verl.trainer.ppo.ray_trainer import RayPPOTrainer, _timer
from verl.trainer.ppo.ray_trainer import *
from .metric_utils import (
    agent_compute_data_metrics as compute_data_metrics,
    compute_timing_metrics,
)
from tqdm import tqdm


def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty='kl'):
    responses = data.batch['responses']
    response_length = responses.size(1)
    token_level_scores = data.batch['token_level_scores']
    batch_size = data.batch.batch_size[0]
    if 'info_mask' in data.batch.keys():
        # masking observations, instead of directly using original `attention_mask`
        attention_mask = data.batch['info_mask']
    else:
        attention_mask = data.batch['attention_mask']
    response_mask = attention_mask[:, -response_length:]

    # compute kl between ref_policy and current policy
    # When apply_kl_penalty, algorithm.use_kl_in_reward=True, so the reference model has been enabled.
    kld = core_algos.kl_penalty(data.batch['old_log_probs'], data.batch['ref_log_prob'],
                                kl_penalty=kl_penalty)  # (batch_size, response_length)
    kld = kld * response_mask
    beta = kl_ctrl.value

    token_level_rewards = token_level_scores - beta * kld

    current_kl = masked_mean(kld, mask=response_mask, axis=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()

    # according to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch['token_level_rewards'] = token_level_rewards

    metrics = {'actor/reward_kl_penalty': current_kl, 'actor/reward_kl_penalty_coeff': beta}

    return data, metrics


def compute_response_mask(data: DataProto):
    responses = data.batch['responses']
    response_length = responses.size(1)
    if 'info_mask' in data.batch.keys():
        # masking observations, instead of directly using original `attention_mask`
        attention_mask = data.batch['info_mask']
    else:
        attention_mask = data.batch['attention_mask']
    return attention_mask[:, -response_length:]


class AgentRayPPOTrainer(RayPPOTrainer):

    def index_select(self, batch: DataProto, indices: list):
        """
        Select a subset of the DataProto based on the given indices.

        Args:
            batch (DataProto): The DataProto object to select from.
            indices (list): A list of indices to select.

        Returns:
            DataProto: A new DataProto object containing the selected data.
        """
        if batch.batch is not None:
            selected_batch = batch.batch[indices]
        else:
            selected_batch = None

        selected_non_tensor = {}
        for key, val in batch.non_tensor_batch.items():
            selected_non_tensor[key] = val[indices]
        selected_meta_info = batch.meta_info.copy()
        # Create a new DataProto object with the selected data
        selected_data = DataProto(batch=selected_batch, non_tensor_batch=selected_non_tensor,
                                  meta_info=selected_meta_info)
        return selected_data

    def dynamic_filter(self, batch: DataProto, metrics: dict):
        """
        Dynamic filter for the batch based on the reward scores
        """
        # we combine with rule-based rm
        reward_extra_infos_dict: dict[str, list]
        batch.meta_info['save_record'] = False
        try:
            reward_result = self.reward_fn(batch, return_dict=True)
            reward_tensor = reward_result['reward_tensor']
            reward_extra_infos_dict = reward_result['reward_extra_info']
            for key, value in reward_extra_infos_dict.items():
                metrics[f'dynamic_filter_reward/{key}'] = np.mean([x for x in value if x is not None])
        except Exception as e:
            print(f'Error in reward_fn: {e}')
            reward_tensor = self.reward_fn(batch)
            reward_extra_infos_dict = {}
            raise e

        response_acc = reward_extra_infos_dict['acc_score']
        question_uids = batch.non_tensor_batch['uid']
        question_acc = {}
        for i in range(len(question_uids)):
            uid = question_uids[i]
            if uid not in question_acc:
                question_acc[uid] = []
            question_acc[uid].append(response_acc[i])
        question_acc_std = {k: np.std(v) for k, v in question_acc.items()}
        # filter out samples with std == 0, which means all the responses are the same reward
        kept_uids = []
        for uid, std in question_acc_std.items():
            if std > 0:
                kept_uids.append(uid)
        kept_uids = set(kept_uids)
        kept_indices = [i for i in range(len(question_uids)) if question_uids[i] in kept_uids]
        batch = self.index_select(batch, kept_indices)
        print(f"Dynamic filter kept {len(kept_indices)} samples out of {len(question_uids)}")
        print(f"Average accuracy after filtering: {np.mean([question_acc[uid] for uid in kept_uids])}")
        metrics.update({
            'dynamic_filter/actural_filter_ratio': len(kept_indices) / len(question_uids),
            'dynamic_filter/before_filtering/avg_acc': np.mean(list(question_acc.values())),
            'dynamic_filter/after_filtering/avg_acc': np.mean([question_acc[uid] for uid in kept_uids]),
            'dynamic_filter/before_filtering/num_samples': len(question_acc),
            'dynamic_filter/after_filtering/num_samples': len(kept_uids),
            'dynamic_filter/before_filtering/full_pass_ratio': len(
                [uid for uid in question_uids if question_acc[uid] >= 1]) / len(question_uids),
            'dynamic_filter/after_filtering/full_pass_ratio': len(
                [uid for uid in kept_uids if question_acc[uid] >= 1]) / len(kept_uids),
            'dynamic_filter/before_filtering/zero_pass_ratio': len(
                [uid for uid in question_uids if question_acc[uid] <= 0]) / len(question_uids),
            'dynamic_filter/after_filtering/zero_pass_ratio': len(
                [uid for uid in kept_uids if question_acc[uid] <= 0]) / len(kept_uids),
        })
        return batch

    def _validate(self):
        data_source_lst = []
        reward_extra_infos_dict: dict[str, list] = defaultdict(list)

        # Lists to collect samples for the table
        sample_inputs = []
        sample_outputs = []
        sample_scores = []

        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)

            # repeat test batch
            test_batch = test_batch.repeat(
                repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n, interleave=True
            )

            # we only do validation on rule-based rm
            if self.config.reward_model.enable and test_batch[0].non_tensor_batch["reward_model"]["style"] == "model":
                return {}

            # Store original inputs
            input_ids = test_batch.batch["input_ids"]
            # TODO: Can we keep special tokens except for padding tokens?
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            sample_inputs.extend(input_texts)
            
            if "multi_modal_inputs" in test_batch.non_tensor_batch.keys():
                test_gen_batch = test_batch.pop(
                    batch_keys=["input_ids", "attention_mask", "position_ids"],
                    non_tensor_batch_keys=["raw_prompt_ids", "multi_modal_data", "multi_modal_inputs"],
                )
            else:
                test_gen_batch = test_batch.pop(
                    batch_keys=["input_ids", "attention_mask", "position_ids"],
                    non_tensor_batch_keys=["raw_prompt_ids"],
                )
            if 'agent' in self.config.actor_rollout_ref.actor.strategy:
                additional_non_tensor_keys = ['extra_info']
                additional_non_tensor_keys = [k for k in additional_non_tensor_keys if k in test_batch.non_tensor_batch.keys()]
                for key in additional_non_tensor_keys:
                    test_gen_batch.non_tensor_batch[key] = test_batch.non_tensor_batch[key]
                test_gen_batch.non_tensor_batch['traj_ids'] = np.array([str(uuid.uuid4()) for _ in range(len(test_gen_batch.batch))], dtype=object)

            test_gen_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                "validate": True,
            }
            print(f"test_gen_batch meta info: {test_gen_batch.meta_info}")

            # pad to be divisible by dp_size
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, self.actor_rollout_wg.world_size)
            test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)

            # unpad
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)
            print("validation generation end")

            # Store generated outputs
            output_ids = test_output_gen_batch.batch["responses"]
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            sample_outputs.extend(output_texts)

            test_batch = test_batch.union(test_output_gen_batch)

            # evaluate using reward_function
            test_batch.meta_info["global_step"] = self.global_steps
            result = self.val_reward_fn(test_batch, return_dict=True)
            reward_tensor = result["reward_tensor"]
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)

            reward_extra_infos_dict["reward"].extend(scores)
            if "reward_extra_info" in result:
                for key, lst in result["reward_extra_info"].items():
                    reward_extra_infos_dict[key].extend(lst)

            data_source_lst.append(test_batch.non_tensor_batch.get("data_source", ["unknown"] * reward_tensor.shape[0]))

        self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)

        for key_info, lst in reward_extra_infos_dict.items():
            assert len(lst) == 0 or len(lst) == len(sample_scores), f"{key_info}: {len(lst)=}, {len(sample_scores)=}"

        data_sources = np.concatenate(data_source_lst, axis=0)

        data_src2var2metric2val = process_validation_metrics(data_sources, sample_inputs, reward_extra_infos_dict)
        metric_dict = {}
        for data_source, var2metric2val in data_src2var2metric2val.items():
            core_var = "acc" if "acc" in var2metric2val else "reward"
            for var_name, metric2val in var2metric2val.items():
                n_max = max([int(name.split("@")[-1].split("/")[0]) for name in metric2val.keys()])
                for metric_name, metric_val in metric2val.items():
                    if (
                        (var_name == core_var)
                        and any(metric_name.startswith(pfx) for pfx in ["mean", "maj", "best"])
                        and (f"@{n_max}" in metric_name)
                    ):
                        metric_sec = "val-core"
                    else:
                        metric_sec = "val-aux"
                    pfx = f"{metric_sec}/{data_source}/{var_name}/{metric_name}"
                    metric_dict[pfx] = metric_val

        return metric_dict
    
    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from verl.utils.tracking import Tracking
        from omegaconf import OmegaConf

        logger = Tracking(project_name=self.config.trainer.project_name,
                          experiment_name=self.config.trainer.experiment_name,
                          default_backend=self.config.trainer.logger,
                          config=OmegaConf.to_container(self.config, resolve=True))

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get('val_before_train', True):
            val_metrics = self._validate()
            pprint(f'Initial validation metrics: {val_metrics}')
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get('val_only', False):
                return

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None

        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                print(f'epoch {epoch}, step {self.global_steps}, with {batch_dict["input_ids"].shape} samples')
                metrics = {}
                timing_raw = {}

                batch: DataProto = DataProto.from_single_dict(batch_dict)

                # pop those keys for generation
                if 'multi_modal_inputs' in batch.non_tensor_batch.keys():
                    gen_batch = batch.pop(
                        batch_keys=['input_ids', 'attention_mask', 'position_ids'],
                        non_tensor_batch_keys=['raw_prompt_ids', 'multi_modal_data', 'multi_modal_inputs']
                    )
                else:
                    gen_batch = batch.pop(
                        batch_keys=['input_ids', 'attention_mask', 'position_ids'],
                        non_tensor_batch_keys=['raw_prompt_ids']
                    )
                if 'agent' in self.config.actor_rollout_ref.actor.strategy:
                    additional_non_tensor_keys = ['extra_info']
                    additional_non_tensor_keys = [k for k in additional_non_tensor_keys if k in batch.non_tensor_batch.keys()]
                    for key in additional_non_tensor_keys:
                        gen_batch.non_tensor_batch[key] = batch.non_tensor_batch[key]
                    gen_batch.non_tensor_batch['traj_ids'] = np.array([str(uuid.uuid4()) for _ in range(len(gen_batch.batch))], dtype=object)

                is_last_step = self.global_steps >= self.total_training_steps

                with _timer('step', timing_raw):
                    # generate a batch
                    print(f"====> step{self.global_steps} starting rollouts")
                    with _timer('gen', timing_raw):
                        gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
                    print(f"====> rollout completed")
                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        with _timer('gen_max', timing_raw):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info['do_sample'] = False
                            gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)

                            batch = batch.union(gen_baseline_output)
                            reward_baseline_tensor = self.reward_fn(batch)
                            reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                            batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))

                            batch.batch['reward_baselines'] = reward_baseline_tensor

                            del gen_baseline_batch, gen_baseline_output

                    batch.non_tensor_batch['uid'] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))],
                                                             dtype=object)
                    # repeat to align with repeated responses in rollout
                    batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    batch = batch.union(gen_batch_output)

                    if getattr(self.config.trainer, 'dynamic_filter', False):
                        batch = self.dynamic_filter(batch, metrics)

                    batch.batch['response_mask'] = compute_response_mask(batch)
                    # balance the number of valid tokens on each dp rank.
                    # Note that this breaks the order of data inside the batch.
                    # Please take care when you implement group based adv computation such as GRPO and rloo
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info['global_token_num'] = torch.sum(batch.batch['attention_mask'], dim=-1).tolist()
                    print(f"====> Computing log probs")
                    # recompute old_log_probs
                    with _timer('old_log_prob', timing_raw):
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        entropys = old_log_prob.batch["entropys"]
                        response_masks = batch.batch["response_mask"]
                        loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                        entropy_loss = agg_loss(
                            loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode
                        )
                        old_log_prob_metrics = {"actor/entropy_loss": entropy_loss.detach().item()}
                        metrics.update(old_log_prob_metrics)
                        old_log_prob.batch.pop("entropys")
                        batch = batch.union(old_log_prob)

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with _timer('ref', timing_raw):
                            ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with _timer('values', timing_raw):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with _timer('adv', timing_raw):
                        # compute scores. Support both model and function-based.
                        # We first compute the scores using reward model. Then, we call reward_fn to combine
                        # the results from reward model and rule-based results.
                        if self.use_rm:
                            # we first compute reward model score
                            reward_tensor = self.rm_wg.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)

                        batch.meta_info['global_step'] = self.global_steps
                        # we combine with rule-based rm
                        reward_extra_infos_dict: dict[str, list]
                        try:
                            reward_result = self.reward_fn(batch, return_dict=True)
                            print(f"====> Done Rewards")
                            reward_tensor = reward_result['reward_tensor']
                            reward_extra_infos_dict = reward_result['reward_extra_info']
                            # update metrics of reward extra info
                            to_remove_keys = []
                            for k, v in reward_extra_infos_dict.items():
                                mean_v = np.mean([x for x in v if x is not None])
                                metrics[f'reward_extra_info/{k}'] = mean_v
                                if None in v:
                                    to_remove_keys.append(k)
                            for k in to_remove_keys:
                                reward_extra_infos_dict.pop(k)
                        except Exception as e:
                            print(f'Error in reward_fn: {e}')
                            reward_tensor = self.reward_fn(batch)
                            reward_extra_infos_dict = {}

                        batch.batch['token_level_scores'] = reward_tensor

                        print(f'{list(reward_extra_infos_dict.keys())=}')
                        if reward_extra_infos_dict:
                            batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                        # compute rewards. apply_kl_penalty if available
                        if self.config.algorithm.use_kl_in_reward:
                            batch, kl_metrics = apply_kl_penalty(batch,
                                                                 kl_ctrl=self.kl_ctrl_in_reward,
                                                                 kl_penalty=self.config.algorithm.kl_penalty)
                            metrics.update(kl_metrics)
                        else:
                            batch.batch['token_level_rewards'] = batch.batch['token_level_scores']

                        # compute advantages, executed on the driver process
                        batch = compute_advantage(batch,
                                                  adv_estimator=self.config.algorithm.adv_estimator,
                                                  gamma=self.config.algorithm.gamma,
                                                  lam=self.config.algorithm.lam,
                                                  num_repeat=self.config.actor_rollout_ref.rollout.n)
                        print(f"====> Done Adv")
                    if "info_mask" in batch.batch.keys():
                        # masking observations, instead of directly using original `attention_mask`
                        ori_attention_mask = batch.batch['attention_mask']
                        batch.batch['attention_mask'] = batch.batch['info_mask']
                    # update critic
                    if self.use_critic:
                        with _timer('update_critic', timing_raw):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info['metrics'])
                        metrics.update(critic_output_metrics)
                    print(f"====> Policy Update")
                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with _timer('update_actor', timing_raw):
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info['metrics'])
                        metrics.update(actor_output_metrics)
                    print(f"====> Done Gradient")
                    if "info_mask" in batch.batch.keys():
                        # restore original attention mask
                        batch.batch['attention_mask'] = ori_attention_mask

                    # validate
                    if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and \
                            (is_last_step or self.global_steps % self.config.trainer.test_freq == 0):
                        with _timer('testing', timing_raw):
                            val_metrics: dict = self._validate()
                            if is_last_step:
                                last_val_metrics = val_metrics
                        metrics.update(val_metrics)

                    if self.config.trainer.save_freq > 0 and (is_last_step or \
                                                              self.global_steps % self.config.trainer.save_freq == 0):
                        with _timer('save_checkpoint', timing_raw):
                            self._save_checkpoint()

                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                if is_last_step:
                    pprint(f'Final validation metrics: {last_val_metrics}')
                    progress_bar.close()
                    return

                progress_bar.update(1)
                self.global_steps += 1
