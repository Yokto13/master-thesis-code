import logging
import os
import random
from collections import deque
from time import time

import gin
import numpy as np
import torch
import torch.nn as nn
from debug_mode import DebugMode
from dreamer import Dreamer
from einops import rearrange
from envs import make_env
from episode import Episode
from replay_buffer import SingleEnvFastReplayBuffer, SingleEnvOnlineReplayBuffer
from torch.distributions import Categorical

import wandb

logger = logging.getLogger(__name__)


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@gin.configurable
class Trainer:
    def __init__(
        self,
        total_env_steps: int,
        batch_size: int,
        seq_len: int,
        eval_interval: int,
        log_dir: str,
        warmup_steps: int,
        log_interval: int,
        latent_dim: int,
        train_every: int = 1,
        eval_episodes: int = 10,
        save_each: int = 50000,
        debug_mode: str = "OFF",
        seed: int = 42,
        burn_in_steps: int = 0,
        thorough_eval: bool = False,
        run_name: str = "",
        save_obs: bool = False,
        save_buffer: bool = False,
        load_buffer: str = "",
        reset_rnn_states: bool = False,
        return_smooth_window: int = 3,
        supervised: bool = False,
        train_only_wm: bool = False,
    ):
        seed_everything(seed)
        self._save_obs = save_obs
        self._save_buffer = save_buffer
        self._load_buffer = load_buffer
        self._reset_rnn_states = reset_rnn_states
        self.supervised = supervised
        self.train_only_wm = train_only_wm

        self.total_env_steps = total_env_steps
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.eval_interval = eval_interval
        self.save_dir = log_dir
        self.warmup_steps = warmup_steps
        self.log_interval = log_interval
        self.debug_mode = DebugMode[debug_mode]
        self.burn_in_steps = burn_in_steps
        self.seed = seed
        self.run_name = run_name

        self.latent_dim = latent_dim
        self.deter_dim = 8 * latent_dim
        self.stoch_dim = latent_dim // 16
        self.thorough_eval = thorough_eval
        self.return_smooth_window = return_smooth_window

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Using device: %s", self.device)

        os.makedirs(self.save_dir, exist_ok=True)

        self.env = make_env(seed=seed)
        self.action_dim = self.env.action_space.n
        logger.info("Action dim: %d", self.action_dim)

        # First reset seeds the environment making the run reproducible
        self.env.reset(seed=seed)

        self.dreamer: Dreamer = Dreamer(
            stoch_dim=self.stoch_dim,
            deter_dim=self.deter_dim,
            latent_dim=self.latent_dim,
            action_dim=self.action_dim,
            total_training_steps=self.total_env_steps,
        ).to(self.device)
        # self.buffer = FastReplayBuffer(store_env_states=(self.debug_mode == DebugMode.RL),
        #    rnn_deter_dim=self.deter_dim, rnn_stoch_dim=self.dreamer.stoch_dim)
        if self._load_buffer:
            self.buffer = SingleEnvOnlineReplayBuffer.load(self._load_buffer, self._reset_rnn_states)
            logger.info("Loaded buffer from %s (%d transitions)", self._load_buffer, len(self.buffer))
        else:
            self.buffer = SingleEnvOnlineReplayBuffer(
                store_env_states=(self.debug_mode == DebugMode.RL),
                rnn_deter_dim=self.deter_dim,
                rnn_stoch_dim=self.dreamer.stoch_dim,
                action_dim=self.action_dim,
            )
        self.train_every = train_every
        self.eval_episodes = eval_episodes
        self.save_each = save_each

        self._last_log_time = time()
        self._train_episode_returns = []
        self._train_episode_lengths = []
        self._recent_episode_returns = deque(maxlen=self.return_smooth_window)
        self._recent_episode_lengths = deque(maxlen=self.return_smooth_window)
        self._last_log_step = 0

        if self.supervised:
            assert self._load_buffer, "Supervised mode requires --load_buffer to be specified."
            logger.warning(
                "Running in supervised mode based on loaded buffer. "
                "The agent still interacts with the environment but never train on these interactions."
                "The interactions are here just to monitor agent's performance in the environment. "
            )
            # Disable pushing new transitions to buffer in supervised mode
            self.buffer.push_transition = lambda *args, **kwargs: None

        if self.train_only_wm:
            logger.warning(
                "train_only_wm enabled: actor/critic training and target-critic updates "
                "will be skipped. Env rollouts still use the (never-updated) actor for monitoring only."
            )

    def train(self):
        global_step = 0
        train_step_counter = 0
        episode_idx = 0
        current_episode_return = 0.0
        current_episode_length = 0

        stoch_dim = self.dreamer.rssm.stoch_dim
        deter_dim = self.dreamer.rssm.deter_dim

        obs, _ = self.env.reset()
        is_first = True

        rssm_state = {
            "deter": torch.zeros(1, deter_dim, device=self.device),
            "stoch": torch.zeros(1, stoch_dim, device=self.device),
        }
        prev_action, prev_reward, prev_done = None, None, None

        ep_obs, ep_acts, ep_rews, ep_dones, ep_deter_states, ep_stoch_states = ([], [], [], [], [], [])

        if self.debug_mode == DebugMode.RL:
            # Required to seed the env in imagination.
            # As RL turns off WM we need to keep the states somewhere.
            # Essentially works as hidden states to start RNNs but we are starting
            # ale env.
            ep_env_states = []

        while global_step < self.total_env_steps:
            # Sampling actions uniformly as warmup is not needed as the policy is itself nearly uniform at the start.
            prev_rssm_state = rssm_state
            action, rssm_state = self._get_action(obs, rssm_state, prev_action, in_eval=False)

            next_obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            current_episode_return += reward
            current_episode_length += 1

            env_state = None

            if type(self.buffer) in (SingleEnvFastReplayBuffer, SingleEnvOnlineReplayBuffer):
                # In single env, we need to pass the state explicitly because

                current_env_state = env_state if self.debug_mode == DebugMode.RL else None

                self.buffer.push_transition(
                    obs,
                    action,
                    reward,
                    done,
                    prev_rssm_state["deter"][0].cpu().numpy(),
                    prev_rssm_state["stoch"][0].cpu().numpy(),
                    current_env_state,
                    is_first=is_first,
                    prev_action=prev_action,
                    prev_reward=prev_reward,
                    prev_done=prev_done,
                )
                is_first = False
            else:
                # Batch buffer: just accumulate
                ep_obs.append(obs)
                ep_acts.append(action)
                ep_rews.append(reward)
                ep_dones.append(done)
                ep_deter_states.append(rssm_state["deter"][0].cpu().numpy())
                ep_stoch_states.append(rssm_state["stoch"][0].cpu().numpy())
                if self.debug_mode == DebugMode.RL:
                    ep_env_states.append(env_state)

            # 2. Handle Episode End (Common logic)
            if done:
                # Flush episode for Batch Buffer ONLY
                if type(self.buffer) not in (SingleEnvFastReplayBuffer, SingleEnvOnlineReplayBuffer):
                    episode = Episode(
                        states=np.array(ep_obs),
                        actions=np.array(ep_acts),
                        rewards=np.array(ep_rews),
                        dones=np.array(ep_dones),
                        deter_states=np.array(ep_deter_states),
                        stoch_states=np.array(ep_stoch_states),
                    )
                    self.buffer.push(episode)
                    # Clear accumulators
                    (ep_obs, ep_acts, ep_rews, ep_dones, ep_deter_states, ep_stoch_states) = ([], [], [], [], [], [])
                    if self.debug_mode == DebugMode.RL:
                        ep_env_states = []

                # Run for both buffers
                self._log_train_episode(current_episode_return, current_episode_length, global_step)
                current_episode_return = 0.0
                current_episode_length = 0
                next_obs, _ = self.env.reset()
                is_first = True
                prev_action = None
                prev_done = None
                prev_reward = None
                rssm_state = {
                    "deter": torch.zeros(1, deter_dim, device=self.device),
                    "stoch": torch.zeros(1, stoch_dim, device=self.device),
                }
                action = 0  # Reset previous action
                episode_idx += 1

            obs = next_obs
            prev_action = action
            prev_done = done
            prev_reward = reward
            global_step += 1

            if global_step % self.save_each == 0:
                self.save_checkpoint(global_step)

            if global_step >= self.warmup_steps and len(self.buffer) >= self.batch_size * (
                self.seq_len + self.burn_in_steps
            ):
                train_step_counter += 1
                if train_step_counter % self.train_every == 0:
                    batch, rnn_indices = self.buffer.sample(
                        self.batch_size, self.seq_len + self.burn_in_steps, return_rnn_indices=True
                    )
                    for k, v in batch.items():
                        if k == "rnn_states":
                            for kk in ["deter", "stoch"]:
                                batch[k][kk] = torch.from_numpy(v[kk]).to(self.device, non_blocking=True)
                        else:
                            batch[k] = torch.from_numpy(v).to(self.device, non_blocking=True)
                    # Turn training before we start training WM and AC.
                    self.dreamer.train()
                    rssm_out, metrics = self.dreamer.train_step(
                        batch,
                        burn_in_steps=self.burn_in_steps,
                        train_only_wm=self.train_only_wm,
                        step=global_step,
                    )

                    rnn_indices = rnn_indices[:, self.burn_in_steps :]  # numpy array, use direct slicing

                    self.dreamer.eval()

                    deter = rssm_out["deter_state"].detach().float()
                    stoch = rssm_out["posterior"]["sample"].detach().float()
                    self.buffer.update_rnn_states(
                        rnn_indices[:, 1:], deter[:, :-1, :].cpu().numpy(), stoch[:, :-1, :].cpu().numpy()
                    )

                    if global_step - self._last_log_step >= self.log_interval and train_step_counter >= 500:
                        steps_since_log = global_step - self._last_log_step
                        self._last_log_step = global_step
                        current_time = time()
                        elapsed = current_time - self._last_log_time
                        per_step = elapsed / steps_since_log
                        self._last_log_time = current_time

                        logger.info(
                            "[Train step %d] Loss WM: %.4f | Actor: %.4f | Critic: %.4f | Steps/s: %.2f",
                            train_step_counter,
                            metrics["wm_loss"],
                            metrics.get("actor", float("nan")),
                            metrics.get("critic", float("nan")),
                            1 / per_step,
                        )

                        # # Verbose metrics (DEBUG level only)
                        # logger.debug(
                        #     "  WM Details: kl=%.4f, recon=%.4f, reward=%.4f, discount=%.4f, "
                        #     "posterior_max=%.2f, prior_max=%.2f",
                        #     wm_metrics["kl_loss"],
                        #     wm_metrics["recon_loss"],
                        #     wm_metrics["reward_loss"],
                        #     wm_metrics["discount_loss"],
                        #     wm_metrics["posterior_max_logit"],
                        #     wm_metrics["prior_max_logit"],
                        # )
                        # logger.debug(
                        #     "  AC Details: entropy=%.4f, avg_imag_reward=%.4f, avg_imag_value=%.4f, "
                        #     "avg_real_value=%.4f, avg_discount=%.4f",
                        #     ac_metrics["entropy_loss"],
                        #     ac_metrics["metadata"]["average_imagined_reward"],
                        #     ac_metrics["metadata"]["average_imagined_value"],
                        #     ac_metrics["metadata"]["average_real_value"],
                        #     ac_metrics["metadata"]["average_discount"],
                        # )
                        logger.debug("  Timing: %.4fs/step | Buffer size: %d", per_step, len(self.buffer))

                        log_data = {}
                        if self._train_episode_returns:
                            log_data["train/episode_return"] = np.mean(self._train_episode_returns)
                            self._train_episode_returns.clear()
                        if self._recent_episode_returns:
                            log_data["train/episode_return_smooth"] = np.mean(self._recent_episode_returns)
                        if self._train_episode_lengths:
                            log_data["train/episode_length"] = np.mean(self._train_episode_lengths)
                            self._train_episode_lengths.clear()
                        if self._recent_episode_lengths:
                            log_data["train/episode_length_smooth"] = np.mean(self._recent_episode_lengths)
                        metric_key_map = {
                            "wm/loss": "wm_loss",
                            "wm/kl_loss": "kl",
                            "wm/recon_loss": "recon",
                            "wm/reward_loss": "reward_loss",
                            "wm/discount_loss": "discount_loss",
                            "ac/actor_loss": "actor",
                            "ac/critic_loss": "critic",
                            "ac/entropy_loss": "entropy",
                            "ac/average_imagined_reward": "average_imagined_reward",
                            "ac/average_imagined_value": "average_imagined_value",
                            "ac/average_real_value": "average_real_value",
                            "ac/average_discount": "average_discount",
                            "ac/current_gamma": "current_gamma",
                            "wm/kl_dyn": "kl_dyn",
                            "wm/kl_rep": "kl_rep",
                            "wm/grad_norm": "norm",
                            "wm/posterior_max_logit": "posterior_max_logit",
                            "wm/prior_max_logit": "prior_max_logit",
                            "ac/adv_mean": "adv_mean",
                            "ac/adv_std": "adv_std",
                            "ac/adv_min": "adv_min",
                            "ac/adv_max": "adv_max",
                            "ac/entropy_mean": "entropy_mean",
                            "ac/value_std": "value_std",
                            "ac/actor_grad_norm": "actor_grad_norm",
                            "ac/critic_grad_norm": "critic_grad_norm",
                            "wm/prior_entropy": "wm/prior_entropy",
                            "wm/posterior_entropy": "wm/posterior_entropy",
                        }

                        for log_key, metric_key in metric_key_map.items():
                            if metric_key not in metrics:
                                continue

                            value = metrics[metric_key]
                            if isinstance(value, torch.Tensor):
                                if value.numel() == 1:
                                    value = value.item()
                                else:
                                    continue
                            log_data[log_key] = value

                        log_data.update(
                            {
                                "time/per_train_step": per_step,
                                "replay_buffer/size": len(self.buffer),
                            }
                        )
                        if train_step_counter % (self.log_interval * 10) == 0:
                            # Logging too often does not provide additional benefit and
                            # just wastes some disk space somewhere.

                            # Extract one frame for visualization (Batch 0, Time 0)
                            # States are (B, T, C, W, H) with values 0-255
                            vis_frame = batch["states"][0, 0].cpu().numpy().astype(np.uint8)
                            vis_frame = np.transpose(vis_frame, (1, 2, 0))

                            log_data["train/observation"] = wandb.Image(vis_frame)

                            if "reconstruction_example" in metrics:
                                recon_example = (metrics["reconstruction_example"].float().cpu().numpy()) * 255.0
                                recon_example = np.transpose(recon_example, (1, 2, 0)).astype(np.uint8)
                                log_data["train/reconstruction"] = wandb.Image(recon_example)
                        wandb.log(log_data, step=global_step)

            if self._should_evaluate(global_step):
                logger.info("Running evaluation episode...")
                eval_env = make_env()
                self.dreamer.eval()
                avg_return, std, _ = self.evaluate(eval_env, self.eval_episodes)
                eval_env.close()
                self._log_eval_results(avg_return, std, global_step)

        if self._train_episode_returns:
            log_data = {"train/episode_return": np.mean(self._train_episode_returns)}
            if self._recent_episode_returns:
                log_data["train/episode_return_smooth"] = np.mean(self._recent_episode_returns)
            if self._train_episode_lengths:
                log_data["train/episode_length"] = np.mean(self._train_episode_lengths)
            if self._recent_episode_lengths:
                log_data["train/episode_length_smooth"] = np.mean(self._recent_episode_lengths)
            wandb.log(log_data, step=global_step)
            self._train_episode_returns.clear()
            self._train_episode_lengths.clear()

        self.save_checkpoint("final")
        if self._save_obs:
            self.save_obs()
        if self._save_buffer:
            self.save_buffer()
        self.env.close()

    def _should_evaluate(self, global_step):
        periodic_eval = global_step % self.eval_interval == 0
        # run more evals towards the end of training
        thorough_eval = (
            self.thorough_eval
            and self.total_env_steps - global_step <= 3000
            and global_step % (self.eval_interval // 10) == 0
        )
        return periodic_eval or thorough_eval

    def _log_train_episode(self, episode_return, episode_length, global_step):
        logger.info(
            "[Train] Episode Return: %.2f | Length: %d | Step: %d", episode_return, episode_length, global_step
        )
        self._train_episode_returns.append(episode_return)
        self._train_episode_lengths.append(episode_length)
        self._recent_episode_returns.append(episode_return)
        self._recent_episode_lengths.append(episode_length)

    def _log_eval_results(self, avg_return, std, global_step):
        logger.info("[Evaluation] Average Return: %.2f | Std: %.2f", avg_return, std)
        wandb.log({"eval/avg_return": avg_return, "eval/std_return": std}, step=global_step)

    def _get_action(self, obs, prev_rssm_state, prev_action, in_eval=False):
        """Get action from observation.

        Args:
            obs: Single observation with shape (C, H, W)
            prev_rssm_state: Dict with 'deter' and 'stoch' tensors of shape (1, dim)
            prev_action: Scalar action index (int)
            in_eval: Whether we are in evaluation mode

        Returns:
            action: Scalar action index (int)
            new_rssm_state: Updated RSSM state dict
        """
        with torch.inference_mode():
            # Add batch and time dims: (C,H,W) -> (1,1,C,H,W)
            obs_tensor = torch.from_numpy(obs).unsqueeze(0).unsqueeze(0).to(self.device)
            action_dim = self.dreamer.actor.action_dim

            if prev_action is None:
                action_tensor = torch.zeros(1, action_dim, device=self.device)
            else:
                prev_action_tensor = torch.tensor([prev_action], device=self.device).long()
                action_tensor = torch.nn.functional.one_hot(prev_action_tensor, num_classes=action_dim).float()

            embed = self.dreamer.encoder(obs_tensor)
            prev_deter = prev_rssm_state["deter"]
            prev_stoch = prev_rssm_state["stoch"]

            step_out = self.dreamer.rssm.wake_step(prev_stoch, action_tensor, prev_deter, embed[:, 0])
            posterior_logits = step_out["posterior"]["logits"]
            if in_eval:
                posterior_logits = rearrange(posterior_logits, "b (c k) -> b c k", c=self.dreamer.num_categories)
                num_classes = self.dreamer.stoch_dim // self.dreamer.num_categories
                stoch_state = nn.functional.one_hot(
                    Categorical(logits=posterior_logits).sample(), num_classes=num_classes
                )
                stoch_state = rearrange(stoch_state, "b c k -> b (c k)")
                stoch_state = stoch_state.float()
            else:
                stoch_state = step_out["posterior"]["sample"]

            state_feature = torch.cat([stoch_state, step_out["deter_state"]], dim=-1)

            probs = self.dreamer.actor.policy(state_feature, add_unimix=not in_eval)
            action_idx = torch.distributions.Categorical(probs=probs).sample()

            new_rssm_state = {"deter": step_out["deter_state"], "stoch": stoch_state}

        return action_idx.item(), new_rssm_state

    def evaluate(self, env, num_episodes=1, render=False):
        stoch_dim = self.dreamer.rssm.stoch_dim
        deter_dim = self.dreamer.rssm.deter_dim
        returns = []

        for _ in range(num_episodes):
            obs, _ = env.reset()
            done = False
            episode_return = 0.0

            rssm_state = {
                "deter": torch.zeros(1, deter_dim, device=self.device),
                "stoch": torch.zeros(1, stoch_dim, device=self.device),
            }
            prev_action = None

            while not done:
                if render:
                    env.render()

                action, rssm_state = self._get_action(obs, rssm_state, prev_action, in_eval=True)

                next_obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                episode_return += reward
                obs = next_obs
                prev_action = action

            returns.append(episode_return)

        avg_return = float(np.mean(returns)) if returns else 0.0
        std = float(np.std(returns)) if returns else 0.0
        logger.info("[EVAL] Episodes: %d | Average Return: %.2f | Std: %.2f", num_episodes, avg_return, std)
        return avg_return, std, returns

    def save_checkpoint(self, step):
        prefix = f"dreamer_{self.run_name}" if self.run_name else "dreamer"
        save_path = os.path.join(self.save_dir, f"{prefix}_s{self.seed}_{step}.pth")
        torch.save(self.dreamer.state_dict(), save_path)
        logger.info("Model saved to %s", save_path)

    def save_obs(self):
        prefix = f"dreamer_{self.run_name}" if self.run_name else "dreamer"
        save_path = os.path.join(self.save_dir, f"{prefix}_s{self.seed}_obs.npy")
        self.buffer.save_obs(save_path, k=10)
        logger.info("Observations saved to %s", save_path)

    def save_buffer(self):
        prefix = f"dreamer_{self.run_name}" if self.run_name else "dreamer"
        save_path = os.path.join(self.save_dir, f"{prefix}_s{self.seed}_buffer.npz")
        self.buffer.save(save_path)
        logger.info("Buffer saved to %s", save_path)
