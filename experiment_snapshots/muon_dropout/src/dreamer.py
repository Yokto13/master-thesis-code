import logging
from itertools import chain

import gin
import torch
import torch.nn as nn
from decoder import Decoder
from einops import rearrange
from encoder import Encoder
from modules import ActionHead, DenseHead, TwoHotHead
from normalize import PercScaler
from optim.agc import clip_grad_agc_
from optim.composite_optimizer import CompositeOptimizer
from optim.keller import SingleDeviceMuon
from optim.laprop import LaProp
from rssm import RSSM
from scalar_head_type import ScalarHeadType
from torch.amp import GradScaler, autocast
from torch.distributions import Categorical
from torch.optim.lr_scheduler import LambdaLR
from torchrl.modules import OneHotCategorical
from utils import dict_apply, get_optimizer_param_groups, get_post_burn_in, unimix

logger = logging.getLogger(__name__)


@gin.configurable
class Dreamer(nn.Module):
    def __init__(
        self,
        stoch_dim: int,
        deter_dim: int,
        latent_dim: int,
        action_dim: int,
        lr: float,
        ac_loss_type: str,
        entropy_regularization: float,
        returns_type: str = "1step",
        eps: float = 1e-4,
        num_categories: int = 32,
        unimix_ratio: float = 0.0,
        critic_type: str = "MSE",
        critic_ema_decay: float = 0.98,
        use_target_network: bool = False,
        rb_loss_scale: float = 0.3,
        slowreg: float = 1.0,
        gae_lambda: float = 0.95,
        gae_gamma: float = 0.997,
        norm: str = "layernorm",
        # Initialization outscale parameters
        reward_head_outscale: float = 0.0,
        cont_head_outscale: float = 1.0,
        actor_outscale: float = 0.01,
        critic_outscale: float = 0.0,
        warmup_steps: int = 0,
        optimizer: str = "adam",
        use_amp: bool = True,
        repval_grad: bool = False,
        horizon: int = 15,
        ac_weight_decay: float = 0.0,
        muon_lr: float = 0.002,
        muon_momentum: float = 0.95,
        muon_weight_decay: float = 0.01,
        muon_ac: bool = False,
    ) -> None:
        super().__init__()

        # Validate optimizer choice
        allowed_optimizers = ["adam", "muon", "laprop", "muon_enc_dec"]
        if optimizer.lower() not in allowed_optimizers:
            raise ValueError(f"optimizer must be one of {allowed_optimizers}, got '{optimizer}'")
        self.optimizer_name = optimizer.lower()

        self.muon_lr = muon_lr
        self.muon_momentum = muon_momentum
        self.muon_weight_decay = muon_weight_decay

        logger.debug(
            "Dreamer Params: stoch_dim=%s deter_dim=%s latent_dim=%s max_action=%s "
            "lr=%s eps=%s ac_loss_type=%s "
            "entropy_regularization=%s returns_type=%s num_categories=%s "
            "unimix_ratio=%s critic_type=%s critic_ema_decay=%s use_target_network=%s "
            "rb_loss_scale=%s slowreg=%s optimizer=%s"
            "repval_grad=%s horizon=%s",
            stoch_dim,
            deter_dim,
            latent_dim,
            action_dim,
            lr,
            eps,
            ac_loss_type,
            entropy_regularization,
            returns_type,
            num_categories,
            unimix_ratio,
            critic_type,
            critic_ema_decay,
            use_target_network,
            rb_loss_scale,
            slowreg,
            optimizer,
            repval_grad,
            horizon,
        )
        self.horizon = horizon
        self.repval_grad = repval_grad
        self.action_dim = int(action_dim)

        self.rb_loss_scale = rb_loss_scale
        self.slowreg = slowreg
        self.stoch_dim = stoch_dim

        self.num_categories = num_categories

        self.stoch_dim *= num_categories
        logger.debug("Using Categorical latent representation. (Multiplied stoch dim by %d)", num_categories)
        logger.debug("stoch_dim is now: %d", self.stoch_dim)

        self.adv_scaler = PercScaler(lo_p=0.05, hi_p=0.95, ema_decay=0.99)

        self.critic_type = ScalarHeadType[critic_type]

        self.encoder = Encoder()
        encoder_output_dim = self.encoder.output_dim
        self.encoder = torch.compile(self.encoder)
        self.decoder = Decoder(stoch_dim=self.stoch_dim, deter_dim=deter_dim, hidden_dim=latent_dim)
        # RSSM should also take norm
        self.rssm = RSSM(
            deter_dim=deter_dim,
            stoch_dim=self.stoch_dim,
            embed_dim=encoder_output_dim,
            hidden_dim=latent_dim,
            num_categories=num_categories,
            unimix_ratio=unimix_ratio,
            action_dim=self.action_dim,
        )

        input_dim_ac = self.stoch_dim + deter_dim

        self.actor = ActionHead(
            input_dim=input_dim_ac,
            action_dim=self.action_dim,
            unimix_ratio=unimix_ratio,
            num_hidden_layers=3,
            outscale=actor_outscale,
            norm=norm,
        )

        if self.critic_type == ScalarHeadType.MSE:
            self.critic = DenseHead(
                input_dim=input_dim_ac, output_dim=1, num_hidden_layers=3, outscale=critic_outscale, norm=norm
            )
            self.target_critic = DenseHead(
                input_dim=input_dim_ac, output_dim=1, num_hidden_layers=3, outscale=critic_outscale, norm=norm
            )
            self.rewards = DenseHead(
                input_dim=input_dim_ac, output_dim=1, num_hidden_layers=1, outscale=reward_head_outscale, norm=norm
            )
        elif self.critic_type == ScalarHeadType.TWO_HOT:
            self.critic = TwoHotHead(input_dim=input_dim_ac, num_hidden_layers=3, outscale=critic_outscale, norm=norm)
            self.target_critic = TwoHotHead(
                input_dim=input_dim_ac, num_hidden_layers=3, outscale=critic_outscale, norm=norm
            )
            self.rewards = TwoHotHead(
                input_dim=input_dim_ac, num_hidden_layers=1, outscale=reward_head_outscale, norm=norm
            )
        elif self.critic_type == ScalarHeadType.HL_GAUSS:
            raise NotImplementedError("HL_GAUSS critic not implemented yet")
        else:
            raise ValueError(f"Unknown critic type: {self.critic_type}")

        self.target_critic.load_state_dict(self.critic.state_dict())
        self.target_critic.requires_grad_(False)

        self.discounts = DenseHead(
            input_dim=input_dim_ac, output_dim=1, num_hidden_layers=1, outscale=cont_head_outscale, norm=norm
        )

        wm_params = list(
            chain(
                self.encoder.parameters(),
                self.decoder.parameters(),
                self.rssm.parameters(),
                self.rewards.parameters(),
                self.discounts.parameters(),
            )
        )
        ac_params = list(chain(self.actor.parameters(), self.critic.parameters()))
        param_groups = [
            {"params": wm_params, "weight_decay": 0.0},
            {"params": ac_params, "weight_decay": ac_weight_decay},
        ]

        if self.optimizer_name in ("muon", "muon_enc_dec"):
            # "muon": Muon on all WM hidden weights; "muon_enc_dec": Muon on encoder/decoder only.
            # LaProp everywhere else; muon_ac additionally routes actor/critic hidden weights to Muon.
            wm_modules = [("encoder", self.encoder), ("decoder", self.decoder)]
            if self.optimizer_name == "muon":
                wm_modules += [("rssm", self.rssm), ("rewards", self.rewards), ("discounts", self.discounts)]
                rest_wm_modules = []
            else:
                rest_wm_modules = [self.rssm, self.rewards, self.discounts]
            muon_params, laprop_wm_params = [], []
            for _, m in wm_modules:
                mu, rest = get_optimizer_param_groups(m)
                muon_params += mu
                laprop_wm_params += rest
            for m in rest_wm_modules:
                laprop_wm_params += list(m.parameters())
            laprop_ac_params = ac_params
            if muon_ac:
                muon_actor, rest_actor = get_optimizer_param_groups(self.actor)
                muon_critic, rest_critic = get_optimizer_param_groups(self.critic)
                muon_params += muon_actor + muon_critic
                laprop_ac_params = rest_actor + rest_critic
                wm_modules += [("actor", self.actor), ("critic", self.critic)]
            self._log_muon_partition(muon_params, wm_modules)
            self._muon_groups = {
                "muon": {
                    "params": muon_params,
                    "lr": muon_lr,
                    "weight_decay": muon_weight_decay,
                    "momentum": muon_momentum,
                },
                "laprop_wm": {"params": laprop_wm_params, "lr": lr, "eps": eps, "weight_decay": 0.0},
                "laprop_ac": {"params": laprop_ac_params, "lr": lr, "eps": eps, "weight_decay": ac_weight_decay},
            }

        self.optimizer = self._create_optimizer(param_groups, lr=lr, eps=eps)

        # Linear warmup schedulers
        def _warmup_lr(step: int) -> float:
            if warmup_steps <= 0:
                return 1.0
            return min(1.0, (step + 1) / warmup_steps)

        self.scheduler = LambdaLR(self.optimizer, lr_lambda=_warmup_lr)

        self.ac_loss_type = ac_loss_type

        self.beta_pred = 1.0
        self.beta_dyn = 1.0
        self.beta_rep = 0.1

        self._policy_loss = torch.nn.CrossEntropyLoss(reduction="none")

        self.entropy_regularization = entropy_regularization

        self.discount_loss = torch.nn.BCEWithLogitsLoss()

        self.returns_type = returns_type

        self.unimix_ratio = unimix_ratio

        self.critic_ema_decay = critic_ema_decay

        self.use_target_network = use_target_network
        logger.debug("use_target_network: %s", self.use_target_network)

        self.rb_loss_scale = rb_loss_scale
        self.slowreg = slowreg

        self.gae_lambda = gae_lambda
        self.gae_gamma = gae_gamma

        # Mixed precision training with bfloat16
        self.use_amp = use_amp
        self.amp_dtype = torch.bfloat16
        self.amp_device = "cuda"
        self.scaler = GradScaler(device=self.amp_device, enabled=self.amp_dtype == torch.float16)

        modules = {
            "rssm": self.rssm,
            "actor": self.actor,
            "critic": self.critic,
            "rewards": self.rewards,
            "discounts": self.discounts,
            "encoder": self.encoder,
            "decoder": self.decoder,
        }
        total = 0
        for key, module in modules.items():
            n = sum(p.numel() for p in module.parameters())
            total += n
            logger.info("%14s: %s", f"{n:,}", key)
        logger.info("%14s: %s", f"{total:,}", "total")

    def train(self, mode=True):
        super().train(mode)
        self.target_critic.train(False)
        return self

    def _log_muon_partition(self, muon_params, named_modules):
        muon_ids = {id(p) for p in muon_params}
        for mod_name, module in named_modules:
            for name, param in module.named_parameters():
                side = "muon" if id(param) in muon_ids else "laprop"
                shape = list(param.shape)
                logger.info("%s partition: %s.%s -> %s %s", self.optimizer_name, mod_name, name, side, shape)
        n_muon = sum(p.numel() for p in muon_params)
        logger.info("%s: %d tensors (%s params) on Muon", self.optimizer_name, len(muon_params), f"{n_muon:,}")

    def _create_optimizer(self, params, lr: float, eps: float):
        """Create optimizer based on self.optimizer_name."""
        if self.optimizer_name == "adam":
            return torch.optim.Adam(params, lr=lr, fused=True, eps=eps)
        elif self.optimizer_name in ("muon", "muon_enc_dec"):
            return CompositeOptimizer(
                constructors={"muon": SingleDeviceMuon, "laprop_wm": LaProp, "laprop_ac": LaProp},
                params=self._muon_groups,
            )
        elif self.optimizer_name == "laprop":
            return LaProp(params, lr=lr, eps=eps)
        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer_name}")

    def forward(self, batch) -> torch.Tensor:
        states = batch["states"]
        actions = batch["actions"]
        is_first = batch["is_first"]
        previous_states = batch.get("rnn_states", None)
        # previous_states = None

        embeds = self.encoder(states)
        rssm_out = self.rssm(actions, embeds, is_first, state=previous_states)

        latent = rssm_out["posterior"]["sample"]
        deter_state = rssm_out["deter_state"]

        reconstructions = self.decoder(latent, deter_state)

        return reconstructions, rssm_out

    def train_step(self, batch, burn_in_steps: int, train_only_wm: bool = False) -> tuple[dict, dict]:
        wm_losses, rssm_out, wm_metrics = self.wm_forward_step(batch, burn_in_steps)

        rssm_out = dict_apply(lambda x: get_post_burn_in(burn_in_steps, x), rssm_out)
        rewards = get_post_burn_in(burn_in_steps, batch["rewards"])
        dones = get_post_burn_in(burn_in_steps, batch["dones"])

        if train_only_wm:
            ac_losses, ac_metrics = {}, {}
        else:
            # Dreams are essentially inference: keep Dropout etc. off in the WM during imagination
            self.rssm.eval()
            ac_losses, ac_metrics = self.ac_forward_step(rssm_out, rewards, dones)
            self.rssm.train()

        losses = {**wm_losses, **ac_losses}

        backward_metrics = self._backward(losses)

        if not train_only_wm:
            self.update_target_network()

        return rssm_out, {**wm_metrics, **ac_metrics, **backward_metrics, **losses}

    def _backward(self, losses: dict):
        metrics = {}

        pred_loss = losses["recon"] + losses["reward"] + losses["discount"]
        wm_loss = pred_loss * self.beta_pred + losses["kl"]
        metrics["wm_loss"] = wm_loss

        train_only_wm = "actor" not in losses
        if train_only_wm:
            total_loss = wm_loss
        else:
            ac_loss = losses["actor"] + losses["critic"]
            total_loss = wm_loss + ac_loss

        trained_modules = [self.encoder, self.decoder, self.rssm, self.rewards, self.discounts]
        if not train_only_wm:
            trained_modules += [self.actor, self.critic]

        def trained_params():
            return chain(*(m.parameters() for m in trained_modules))

        self.optimizer.zero_grad()

        self.scaler.scale(total_loss).backward()
        self.scaler.unscale_(self.optimizer)
        clip_grad_agc_(trained_params())

        metrics["norm"] = torch.nn.utils.clip_grad_norm_(
            trained_params(),
            max_norm=float("inf"),
        )

        with torch.no_grad():
            self.scaler.step(self.optimizer)
        self.scaler.update()
        self.scheduler.step()

        return metrics

    def wm_forward_step(self, batch, burn_in_steps: int) -> torch.Tensor:
        loss, metrics = {}, {}
        with autocast(device_type=self.amp_device, dtype=self.amp_dtype, enabled=self.use_amp):
            reconstructions, rssm_out = self(batch)
            states = batch["states"]
            rewards = batch["rewards"]
            dones = batch["dones"]

            rewards = get_post_burn_in(burn_in_steps, rewards)
            dones = get_post_burn_in(burn_in_steps, dones)

            loss["recon"] = self.decoder.reconstruction_loss(
                get_post_burn_in(burn_in_steps, reconstructions), get_post_burn_in(burn_in_steps, states)
            )

        posterior_max_logit = torch.tensor(0.0)
        prior_max_logit = torch.tensor(0.0)

        assert self.unimix_ratio > 0, "debug"
        posterior_logits = rssm_out["posterior"]["logits"].float()
        prior_logits = rssm_out["prior"]["logits"].float()

        posterior_logits = get_post_burn_in(burn_in_steps, posterior_logits)
        prior_logits = get_post_burn_in(burn_in_steps, prior_logits)

        posterior_logits = rearrange(posterior_logits, "b t (c k) -> b t c k", c=self.num_categories)
        prior_logits = rearrange(prior_logits, "b t (c k) -> b t c k", c=self.num_categories)

        # num categories is a paramt but the original repo uses 32 everywhere.
        # I'm fixing it here to make sure that there are no suprises.
        assert self.num_categories == 32

        # Apply unimix independently to raw logits for KL computation
        # This matches the JAX implementation where unimix is applied fresh at distribution creation time
        if self.unimix_ratio > 0.0:
            # Apply unimix independently to non-detached and detached logits
            # (both applied to the same raw logits before overwriting)
            p_post = unimix(posterior_logits, self.unimix_ratio)
            p_post_detached = unimix(posterior_logits.detach(), self.unimix_ratio)
            posterior_logits = torch.log(p_post)
            posterior_logits_detached = torch.log(p_post_detached)

            p_prior = unimix(prior_logits, self.unimix_ratio)
            p_prior_detached = unimix(prior_logits.detach(), self.unimix_ratio)
            prior_logits = torch.log(p_prior)
            prior_logits_detached = torch.log(p_prior_detached)
        else:
            posterior_logits_detached = posterior_logits.detach()
            prior_logits_detached = prior_logits.detach()

        posterior_max_logit = torch.max(posterior_logits[0, 0, 0, :])
        prior_max_logit = torch.max(prior_logits[0, 0, 0, :])

        post_dist = OneHotCategorical(logits=posterior_logits)
        prior_dist = OneHotCategorical(logits=prior_logits)

        metrics["wm/posterior_entropy"] = post_dist.entropy().mean()
        metrics["wm/prior_entropy"] = prior_dist.entropy().mean()

        post_dist_detached = OneHotCategorical(logits=posterior_logits_detached)
        prior_dist_detached = OneHotCategorical(logits=prior_logits_detached)

        kl_dyn = torch.distributions.kl.kl_divergence(post_dist_detached, prior_dist)
        kl_rep = torch.distributions.kl.kl_divergence(post_dist, prior_dist_detached)

        # Sum over categorical dimensions as in the original Hafner's implementation (done in agg call in the orig)
        kl_dyn = kl_dyn.sum(dim=-1)
        kl_rep = kl_rep.sum(dim=-1)

        # To avoid a degenerate solution where the dynamics are trivial to predict but fail
        # to contain enough information about the inputs, we employ free bits by clipping the dynamics and
        # representation losses below the value of 1 nat
        # clamp kills gradients for values less than min so that is what we need here.
        kl_dyn = torch.clamp(kl_dyn, min=1.0)
        kl_rep = torch.clamp(kl_rep, min=1.0)

        loss["kl"] = self.beta_dyn * kl_dyn.mean() + self.beta_rep * kl_rep.mean()

        with autocast(device_type=self.amp_device, dtype=self.amp_dtype, enabled=self.use_amp):
            latent = get_post_burn_in(burn_in_steps, rssm_out["posterior"]["sample"])
            deter = get_post_burn_in(burn_in_steps, rssm_out["deter_state"])
            feature = torch.cat([latent, deter], dim=-1)

            if self.critic_type == ScalarHeadType.TWO_HOT:
                _, logits = self.rewards(feature, return_logits=True)
                reward_loss = self.rewards.two_hot_loss(logits, rewards)
            else:
                reward_pred = self.rewards(feature)
                reward_loss = self.rewards.mse_loss(reward_pred, rewards.unsqueeze(-1))

            # 4. Discount Loss
            discount_pred = self.discounts(feature)
            discount_target = (1.0 - dones.float()).unsqueeze(-1)

            discount_target *= self.gae_gamma

            assert discount_pred.shape == discount_target.shape
            discount_loss = self.discount_loss(discount_pred, discount_target)

            loss["reward"] = reward_loss
            loss["discount"] = discount_loss

        # loss = kl_loss + self.beta_pred * pred_loss

        # self.wm_optimizer.zero_grad()

        # self.scaler.scale(loss).backward()

        # self.scaler.unscale_(self.wm_optimizer)
        # # nn.utils.clip_grad_norm_(self.parameters(), 100.0)
        # clip_grad_agc_(
        #     chain(
        #         self.encoder.parameters(),
        #         self.decoder.parameters(),
        #         self.rssm.parameters(),
        #         self.rewards.parameters(),
        #         self.discounts.parameters(),
        #     )
        # )
        # wm_grad_norm = torch.nn.utils.clip_grad_norm_(
        #     chain(
        #         self.encoder.parameters(),
        #         self.decoder.parameters(),
        #         self.rssm.parameters(),
        #         self.rewards.parameters(),
        #         self.discounts.parameters(),
        #     ),
        #     max_norm=float("inf"),
        # )

        # self.scaler.step(self.wm_optimizer)
        # self.scaler.update()
        # self.wm_scheduler.step()

        metrics.update(
            {
                "kl_dyn": kl_dyn.mean(),
                "kl_rep": kl_rep.mean(),
                "reward_loss": reward_loss,
                "discount_loss": discount_loss,
                "posterior_max_logit": posterior_max_logit,
                "prior_max_logit": prior_max_logit,
                "reconstruction_example": reconstructions[0, 0].detach().clone(),
            }
        )

        return loss, rssm_out, metrics

    def imagine(self, start_state):
        stoch_state = start_state["stoch"].detach()
        deter_state = start_state["deter"].detach()

        imagined = {
            "stoch": [],
            "deter": [],
            "action": [],
            "action_logits": [],
        }
        feature = torch.cat([stoch_state, deter_state], dim=-1).detach()

        for _ in range(self.horizon):
            action_logits = self.actor(feature)
            probs = unimix(action_logits, self.unimix_ratio)
            action = nn.functional.one_hot(Categorical(probs=probs).sample(), num_classes=self.action_dim)
            action = action + probs - probs.detach()

            # Action logits could be either the original or the unimixed version
            # The unimixed makes currently more sense to me because they correspond to
            # the distribution that is being sampled.
            action_logits = torch.log(probs)

            imagined["stoch"].append(stoch_state)
            imagined["deter"].append(deter_state)
            imagined["action"].append(action)
            imagined["action_logits"].append(action_logits)

            with torch.no_grad():
                d = self.rssm.dream_step(stoch_state, action, deter_state)

            stoch_state = d["prior"]["sample"].detach()
            deter_state = d["deter_state"].detach()
            feature = torch.cat([stoch_state, deter_state], dim=-1)

        # Append final post-dream-step state so we can slice next-step features
        imagined["stoch"].append(stoch_state)
        imagined["deter"].append(deter_state)

        imagined = {k: torch.stack(v, dim=0) for k, v in imagined.items()}

        # Batch-compute rewards and discounts from next-step features
        # next_feature[t] = state after taking action[t], matching the old per-step computation
        # stoch/deter keep horizon+1 entries — the extra final state is needed for critic bootstrapping
        next_feature = torch.cat([imagined["stoch"][1:], imagined["deter"][1:]], dim=-1)

        imagined["reward"] = self.rewards(next_feature)
        imagined["discount"] = torch.sigmoid(self.discounts(next_feature))

        return imagined

    def ac_forward_step(self, rssm_out, rb_rewards, rb_dones):
        loss = {}
        with autocast(device_type=self.amp_device, dtype=self.amp_dtype, enabled=self.use_amp):
            stoch_rb = rssm_out["posterior"]["sample"]
            deter_rb = rssm_out["deter_state"]

            rb_features = torch.cat([stoch_rb, deter_rb], dim=-1)
            rb_features = rearrange(rb_features, "b t ... -> t b ...")

            if not self.repval_grad:
                rb_features = rb_features.detach()

            stoch_rb = stoch_rb.detach()
            deter_rb = deter_rb.detach()

            rb_rewards = rearrange(rb_rewards, "b t ... -> t b ...")[1:]
            rb_dones = rearrange(rb_dones, "b t ... -> t b ...")[1:]
            rb_dones = rb_dones.float()

            B, T, _ = stoch_rb.shape
            logger.debug("Batch size (B) for AC training: %d", B)
            logger.debug("Trajectory length (T) for AC training: %d", T)
            stoch_rb = stoch_rb.reshape(B * T, -1)
            deter_rb = deter_rb.reshape(B * T, -1)
            start_state = {"stoch": stoch_rb, "deter": deter_rb}

            traj = self.imagine(start_state)

            # 1. Setup Data
            rewards = traj["reward"]
            discounts = traj["discount"]

            discounts = rearrange(discounts, "t b 1 -> t b")
            weights = torch.cumprod(discounts, dim=0).detach()

            stoch = traj["stoch"]
            deter = traj["deter"]
            feature = torch.cat([stoch, deter], dim=-1)

            # 2. Compute Targets
            with torch.inference_mode():
                target_values = self.target_critic(feature).detach()
                rb_target_values = self.target_critic(rb_features).detach()

            # We detach so that we do not create gradients inside the world model
            # We want to stop critic from pulling world model to make its loss lower.
            if self.critic_type == ScalarHeadType.MSE:
                values = self.critic(feature.detach())
                rb_values = self.critic(rb_features)
            if self.critic_type == ScalarHeadType.TWO_HOT:
                values, logits = self.critic(feature.detach(), return_logits=True)
                rb_values, rb_logits = self.critic(rb_features, return_logits=True)

        # GAE computation in fp32 for numerical stability (compute_gae uses torch.compile)
        # Cast to fp32 for GAE computation
        rewards = rewards.float()
        target_values = target_values.float()
        values = values.float()
        rb_target_values = rb_target_values.float()
        rb_values = rb_values.float()
        discounts = discounts.float()
        rb_rewards = rb_rewards.float()
        rb_dones = rb_dones.float()

        if len(rewards.shape) == 3:
            rewards = rearrange(rewards, "t b 1 -> t b")

        if len(target_values.shape) == 3:
            target_values = rearrange(target_values, "t b 1 -> t b")
            values = rearrange(values, "t b 1 -> t b")
        if len(rb_target_values.shape) == 3:
            rb_target_values = rearrange(rb_target_values, "t b 1 -> t b")
            rb_values = rearrange(rb_values, "t b 1 -> t b")

        adv, returns = compute_gae(
            rewards=rewards,
            values=target_values if self.use_target_network else values,
            dones=1.0 - discounts,
            lam=self.gae_lambda,
            gamma=1.0,
        )
        # Ground rb bootstrap in imagination: use the first-step imagined lambda-returns as
        # the bootstrap for rb GAE. returns[0] has shape [B*T] — the lambda-return at the
        # start of each imagined rollout, i.e. the model's multi-step estimate of value for
        # each replay buffer state. This matches the reference which uses
        # boot = imgloss_out['ret'][:, 0].reshape(B, K) in repl_loss.

        # The following reshape + swap requires 2 steps because the ordering inside returns
        imag_values_for_rb = returns[0].detach().reshape(B, T)  # [B, T]
        imag_values_for_rb = rearrange(imag_values_for_rb, "b t -> t b")  # [T, B]

        # To improve value prediction in environments where rewards are challenging
        # to predict, we apply the critic loss both to imagined trajectories with loss scale βval = 1 and to
        # trajectories sampled from the replay buffer with loss scale βrepval = 0.3.
        _, rb_returns = compute_gae(
            rewards=rb_rewards,
            values=imag_values_for_rb,
            dones=rb_dones,
            lam=self.gae_lambda,
            gamma=self.gae_gamma,
        )

        adv = adv.detach()

        adv = self.adv_scaler(adv, returns)

        if self.critic_type == ScalarHeadType.TWO_HOT:
            returns_loss_img = self.critic.loss(logits[:-1], returns.detach(), reduction="none")
            reg_img = self.critic.loss(logits[:-1], target_values[:-1].detach(), reduction="none")
            critic_loss_img = returns_loss_img + self.slowreg * reg_img
            assert weights.shape == critic_loss_img.shape
            critic_loss_img = weights * critic_loss_img

            returns_loss_rb = self.critic.loss(rb_logits[:-1], rb_returns.detach(), reduction="none")
            reg_rb = self.critic.loss(rb_logits[:-1], rb_target_values[:-1].detach(), reduction="none")
            critic_loss_rb = returns_loss_rb + self.slowreg * reg_rb

            rb_weights = 1 - rb_dones
            assert rb_weights.shape == critic_loss_rb.shape
            critic_loss_rb = rb_weights * critic_loss_rb

            loss["critic"] = critic_loss_img.mean() + self.rb_loss_scale * (
                critic_loss_rb.sum() / rb_weights.sum().clamp(min=1)
            )

        entropy_loss = torch.tensor(0.0)

        policy_logits = traj["action_logits"].float()  # fp32 for entropy computation
        actions = traj["action"].detach()

        T = policy_logits.shape[0]
        B = policy_logits.shape[1]
        logger.debug("B based on policy_logits shape: %d", B)
        logger.debug("T based on policy_logits shape: %d", T)

        # TODO: this could be slightly faster without the distribution creation overhead
        entropy = torch.distributions.Categorical(logits=policy_logits).entropy()
        entropy_loss = self.entropy_regularization * entropy.mean()
        action_logprob = self._policy_loss(policy_logits.movedim(-1, 1), actions.movedim(-1, 1))

        assert adv.shape == action_logprob.shape == entropy.shape == (T, B)

        loss["actor"] = (
            weights.detach() * (adv.detach() * action_logprob - self.entropy_regularization * entropy)
        ).mean()

        metrics = {
            "average_imagined_reward": rewards.mean(),
            "average_imagined_value": values[:-1].mean(),
            "average_real_value": rb_values[:-1].mean(),
            "average_discount": discounts.mean(),
            "adv_mean": adv.mean(),
            "adv_std": adv.std(),
            "adv_min": adv.min(),
            "adv_max": adv.max(),
            "entropy_mean": entropy.mean(),
            "value_std": values[:-1].std(),
            "entropy": entropy_loss,
        }

        return loss, metrics

    def update_target_network(self):
        with torch.inference_mode():
            for param, target_param in zip(self.critic.parameters(), self.target_critic.parameters()):
                target_param.data.lerp_(param.data, 1 - self.critic_ema_decay)


# Did not profile this but this function is a good candidate for compilation.
@torch.compile
def compute_gae(rewards, values, dones, gamma: float, lam: float):
    """
    Generalized Advantage Estimation (GAE).

    # taken from: https://mochan.org/posts/gae/

    Args:
        rewards: Tensor [T, B]
        values:  Tensor [T+1, B]  (bootstrap value in the last row)
        dones:   Tensor [T, B]     (1.0 if episode ended at step t, else 0.0)
        gamma:   discount factor (float)
        lam:     GAE lambda (float)

    Returns:
        advantages: Tensor [T, B]
        returns:    Tensor [T, B]  where returns = advantages + values[:-1]
    """
    # Validate input shapes where T is time steps and B is batch size
    T, B = rewards.shape
    assert values.shape == (T + 1, B), "values must be time-major with T+1 rows for bootstrapping"
    assert dones.shape == (T, B), "dones must be time-major with T rows"

    # Initialize tensors
    advantages = torch.zeros_like(rewards)  # [T, B]. Holds the advantage estimates
    gae = torch.zeros(B, dtype=values.dtype, device=values.device)  # [B]. Holds the running GAE value

    # Compute GAE in reverse order from T-1 down to 0 of the trajectory
    for t in reversed(range(T)):
        # If the episode ended at step t, we do not propagate the advantage
        not_done = 1.0 - dones[t]
        # TD error
        delta = rewards[t] + gamma * values[t + 1] * not_done - values[t]
        # GAE recursive formula
        gae = delta + gamma * lam * not_done * gae
        # Store the advantage estimate
        advantages[t] = gae

    # Compute returns
    returns = advantages + values[:-1]
    return advantages, returns
