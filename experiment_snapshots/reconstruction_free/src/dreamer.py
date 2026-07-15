import logging
from itertools import chain

import gin
import torch
import torch.nn as nn
from decoder import Decoder
from einops import rearrange
from encoder import ConvNeXtEncoder, Encoder, ImpalaEncoder, ImpalaEncoderDreamerSize
from modules import ActionHead, DenseHead, TwoHotHead
from normalize import PercScaler
from optim.agc import clip_grad_agc_
from optim.composite_optimizer import CompositeOptimizer
from optim.keller import SingleDeviceMuon
from optim.laprop import LaProp
from rssm import RSSM
from scalar_head_type import ScalarHeadType
from sigreg import EppsPulley, SlicingUnivariateTest
from torch.amp import GradScaler, autocast
from torch.distributions import Categorical
from torch.optim.lr_scheduler import LambdaLR
from torchrl.modules import OneHotCategorical
from utils import dict_apply, get_optimizer_param_groups, get_post_burn_in, random_shift, trunc_normal_init, unimix

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
        sigreg_lambda: float = 0.05,
        sigreg_num_slices: int = 1024,
        sigreg_t_max: float = 3.0,
        sigreg_n_points: int = 17,
        lejepa_weight: float = 1.0,
        emb_recon_weight: float = 0.0,
        inv_dyn_weight: float = 0.0,
        aug_random_shift: bool = False,
        aug_shift_pad: int = 4,
        aug_two_view: bool = False,
        encoder_type: str = "default",
        muon_ac: bool = False,
        muon_momentum: float = 0.95,
        muon_lr: float = 0.02,
        muon_weight_decay: float = 0.0,
        debug_decoders: bool = False,
    ) -> None:
        super().__init__()

        # Validate optimizer choice
        allowed_optimizers = ["adam", "muon", "laprop"]
        if optimizer.lower() not in allowed_optimizers:
            raise ValueError(f"optimizer must be one of {allowed_optimizers}, got '{optimizer}'")
        self.optimizer_name = optimizer.lower()

        logger.debug(
            "Dreamer Params: stoch_dim=%s deter_dim=%s latent_dim=%s max_action=%s "
            "lr=%s eps=%s ac_loss_type=%s "
            "entropy_regularization=%s returns_type=%s num_categories=%s "
            "unimix_ratio=%s critic_type=%s critic_ema_decay=%s use_target_network=%s "
            "rb_loss_scale=%s slowreg=%s optimizer=%s "
            "repval_grad=%s horizon=%s lejepa_weight=%s emb_recon_weight=%s inv_dyn_weight=%s",
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
            lejepa_weight,
            emb_recon_weight,
            inv_dyn_weight,
        )
        self.horizon = horizon
        self.repval_grad = repval_grad
        self.muon_momentum = muon_momentum
        self.muon_lr = muon_lr
        self.muon_weight_decay = muon_weight_decay
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

        if encoder_type == "default":
            self.encoder = Encoder()
        elif encoder_type == "impala":
            self.encoder = ImpalaEncoder()
        elif encoder_type == "impala_dreamer_size":
            self.encoder = ImpalaEncoderDreamerSize()
        elif encoder_type == "convnext":
            self.encoder = ConvNeXtEncoder()
        else:
            raise ValueError(
                f"Unknown encoder_type: {encoder_type!r}. "
                "Expected 'default', 'impala', 'impala_dreamer_size', or 'convnext'."
            )
        encoder_output_dim = self.encoder.output_dim

        # self.encoder = torch.compile(self.encoder)
        # Debug-only pixel decoders for wandb visualization (trained on detached
        # features). Disable via `debug_decoders=False` to save GPU memory.
        # self.decoder reconstructs from [z, h]; self.embed_probe from the detached
        # encoder embedding alone, separating "encoder never captured it" from
        # "posterior dropped it".
        self.debug_decoders = debug_decoders
        if self.debug_decoders:
            self.decoder = Decoder(stoch_dim=self.stoch_dim, deter_dim=deter_dim, hidden_dim=latent_dim)
            self.embed_probe = Decoder(stoch_dim=encoder_output_dim, deter_dim=None, hidden_dim=latent_dim)
        else:
            self.decoder = None
            self.embed_probe = None
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

        # JEPA-style predictor: maps RSSM deterministic state to encoder embedding
        # space. Replaces the image-reconstruction head as the grounding signal
        # for the encoder. SIGReg (below) prevents the trivial collapse where
        # encoder + predictor settle on a constant output.
        self.emb_predictor = nn.Sequential(
            nn.Linear(deter_dim, 2048),
            nn.BatchNorm1d(2048, eps=1e-4),
            nn.SiLU(),
            nn.Linear(2048, 1024),
        )

        # JEPA-style embedding decoder: reconstructs embed_t from the posterior
        # feature [deter, posterior_stoch]. Closed-loop counterpart to emb_predictor
        # (which predicts embed_t from the prior) and the embedding-space analog of
        # the deleted pixel decoder — it forces the posterior latent the policy
        # consumes to actually encode the current observation. Gated by emb_recon_weight.
        self.emb_decoder = nn.Sequential(
            nn.Linear(deter_dim + self.stoch_dim, 2048),
            nn.BatchNorm1d(2048, eps=1e-4),
            nn.SiLU(),
            nn.Linear(2048, encoder_output_dim),
        )

        # Inverse-dynamics head: predicts action_t from (embed_{t-1}, embed_t).
        # Operates on raw encoder embeddings (not detached) so its gradient trains
        # the encoder to retain controllable content — the content-grounding signal
        # SIGReg lacks. Independent of emb_predictor (forward dynamics) and
        # emb_decoder (posterior reconstruction). Gated by inv_dyn_weight.
        # Uses RMSNorm (Dreamer house style); BN's anti-collapse role doesn't apply
        # to a discriminative CE classifier.
        self.inv_dyn = nn.Sequential(
            nn.Linear(2 * encoder_output_dim, 2048),
            nn.RMSNorm(2048, eps=1e-4),
            nn.SiLU(),
            nn.Linear(2048, self.action_dim),
        )

        # Debug-only inverse-dynamics probe: same architecture as inv_dyn but reads
        # *detached* embeddings, so it measures action-decodability from the encoder
        # without shaping it. The inv_dyn-on vs inv_dyn-off contrast in its accuracy
        # tells whether the real inv_dyn term actually *adds* controllable content
        # the encoder wouldn't otherwise carry. Gated by debug_decoders.
        self.inv_dyn_probe = (
            nn.Sequential(
                nn.Linear(2 * encoder_output_dim, 2048),
                nn.RMSNorm(2048, eps=1e-4),
                nn.SiLU(),
                nn.Linear(2048, self.action_dim),
            )
            if self.debug_decoders
            else None
        )

        # init
        probe_init = [self.inv_dyn_probe] if self.debug_decoders else []
        for m in [self.emb_predictor, self.emb_decoder, self.inv_dyn, *probe_init]:
            for layer in m:
                if isinstance(layer, nn.Linear):
                    trunc_normal_init(layer.weight, fan="in", scale=1)
                    nn.init.zeros_(layer.bias)

        self.embed_dim = encoder_output_dim
        self.sigreg_lambda = sigreg_lambda
        self.lejepa_weight = lejepa_weight
        self.emb_recon_weight = emb_recon_weight
        self.inv_dyn_weight = inv_dyn_weight
        self.aug_random_shift = aug_random_shift
        self.aug_shift_pad = aug_shift_pad
        self.aug_two_view = aug_two_view
        logger.debug(
            "aug_random_shift=%s aug_shift_pad=%s aug_two_view=%s", aug_random_shift, aug_shift_pad, aug_two_view
        )
        self.sigreg = SlicingUnivariateTest(
            univariate_test=EppsPulley(t_max=sigreg_t_max, n_points=sigreg_n_points),
            num_slices=sigreg_num_slices,
            reduction="mean",
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

        # print(list(self.rssm.parameters()))
        muon_enc, rest_enc = get_optimizer_param_groups(self.encoder)
        muon_dec, rest_dec = get_optimizer_param_groups(self.decoder) if self.debug_decoders else ([], [])
        muon_rssm, rest_rssm = get_optimizer_param_groups(self.rssm)
        muon_rew, rest_rew = get_optimizer_param_groups(self.rewards)
        muon_disc, rest_disc = get_optimizer_param_groups(self.discounts)
        muon_actor, rest_actor = get_optimizer_param_groups(self.actor)
        muon_critic, rest_critic = get_optimizer_param_groups(self.critic)

        muon = list(chain(muon_enc, muon_dec, muon_rssm, muon_rew, muon_disc))
        rest = list(chain(rest_enc, rest_dec, rest_rssm, rest_rew, rest_disc))
        if muon_ac:
            muon += muon_actor
            rest += rest_actor
            muon += muon_critic
            rest += rest_critic

        debug_models = [self.encoder, self.rssm, self.rewards, self.discounts, self.actor, self.critic]
        if self.debug_decoders:
            debug_models.insert(1, self.decoder)
        for model in debug_models:
            m, r = get_optimizer_param_groups(model)
            muon_ids = {id(p) for p in m}
            adamw_ids = {id(p) for p in r}
            print("=== MUON PARAMETERS ===")
            for name, param in model.named_parameters():
                if id(param) in muon_ids:
                    print(f"{name} | Shape: {list(param.shape)}")

            print("\n=== ADAMW PARAMETERS ===")
            for name, param in model.named_parameters():
                if id(param) in adamw_ids:
                    print(f"{name} | Shape: {list(param.shape)}")

        wm_param_iters = [self.encoder.parameters()]
        if self.debug_decoders:
            # Debug-only decoders, trained on detached features (see wm_forward_step).
            wm_param_iters += [
                self.decoder.parameters(),
                self.embed_probe.parameters(),
                self.inv_dyn_probe.parameters(),
            ]
        wm_param_iters += [
            self.emb_predictor.parameters(),
            self.emb_decoder.parameters(),
            self.inv_dyn.parameters(),
            self.rssm.parameters(),
            self.rewards.parameters(),
            self.discounts.parameters(),
        ]
        self.optimizer = self._create_optimizer(
            chain(*wm_param_iters),
            chain(
                self.actor.parameters(),
                self.critic.parameters(),
            ),
            lr=lr,
            eps=eps,
            muon_params=muon,
            laprop_params=rest + ([] if muon_ac else list(self.actor.parameters()) + list(self.critic.parameters())),
        )

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
            "emb_predictor": self.emb_predictor,
            "emb_decoder": self.emb_decoder,
            "inv_dyn": self.inv_dyn,
        }
        if self.debug_decoders:
            modules["decoder"] = self.decoder
            modules["embed_probe"] = self.embed_probe
            modules["inv_dyn_probe"] = self.inv_dyn_probe
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

    def _create_optimizer(self, wm_params, ac_params, lr: float, eps: float, muon_params=None, laprop_params=None):
        """Create optimizer based on self.optimizer_name."""
        params = chain(wm_params, ac_params)
        if self.optimizer_name == "adam":
            return torch.optim.Adam(params, lr=lr, fused=True, eps=eps)
        elif self.optimizer_name == "muon":
            wm_params = list(wm_params)
            if muon_params is None and laprop_params is None:
                muon_params = [p for p in wm_params if p.ndim == 2]
                laprop_params = [p for p in wm_params if p.ndim != 2] + list(ac_params)

            return CompositeOptimizer(
                constructors={"laprop": LaProp, "muon": SingleDeviceMuon},
                params={
                    "laprop": {"params": laprop_params, "lr": lr, "eps": eps},
                    "muon": {
                        "params": muon_params,
                        "lr": self.muon_lr,
                        "weight_decay": self.muon_weight_decay,
                        "momentum": self.muon_momentum,
                    },
                },
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

        # DrQ-style random-shift augmentation (training only). One shift per
        # sequence keeps the dynamics consistent. `states` stays clean (used by the
        # two-view target and the debug decoder); `states_online` drives the
        # encoder/RSSM and carries the SIGReg gradient.
        aug_on = self.aug_random_shift and self.training
        states_online = random_shift(states, self.aug_shift_pad) if aug_on else states

        embeds = self.encoder(states_online)
        rssm_out = self.rssm(actions, embeds, is_first, state=previous_states)

        # Debug-only decoder: inputs are detached so gradients never flow back
        # into the RSSM/encoder. The decoder is trained on these detached features
        # purely to visualize what the reconstruction-free latents encode.
        reconstructions = None
        if self.debug_decoders:
            latent = rssm_out["posterior"]["sample"]
            deter_state = rssm_out["deter_state"]
            reconstructions = self.decoder(latent.detach(), deter_state.detach())

        # Two-view target: predict the CLEAN embedding from the augmented-input
        # state, so the shift trains encoder invariance without injecting positional
        # noise into the predicted dynamics (the pred/emb_recon targets stay clean).
        # The target is stop-gradiented downstream, so the extra encode is no_grad.
        # When two-view is off, the target is just the (online) embeds -> single view.
        if aug_on and self.aug_two_view:
            with torch.no_grad():
                target_embeds = self.encoder(states)
        else:
            target_embeds = embeds

        return reconstructions, rssm_out, embeds, target_embeds

    def train_step(
        self, batch, burn_in_steps: int, train_only_wm: bool = False, train: bool = True
    ) -> tuple[dict, dict]:
        self.train()
        wm_losses, rssm_out, wm_metrics = self.wm_forward_step(batch, burn_in_steps)

        rssm_out = dict_apply(lambda x: get_post_burn_in(burn_in_steps, x), rssm_out)
        rewards = get_post_burn_in(burn_in_steps, batch["rewards"])
        dones = get_post_burn_in(burn_in_steps, batch["dones"])

        if train_only_wm:
            ac_losses, ac_metrics = {}, {}
        else:
            # Dreams are essentially inference
            # we should be carefull to turn of Dropout and related things in the WM
            self.wm_to_eval()
            ac_losses, ac_metrics = self.ac_forward_step(rssm_out, rewards, dones)
            self.wm_to_train()

        losses = {**wm_losses, **ac_losses}

        backward_metrics = self._backward(losses, train)

        if not train_only_wm:
            self._update_target_network()

        return rssm_out, {**wm_metrics, **ac_metrics, **backward_metrics, **losses}

    def wm_to_eval(self):
        self.encoder.eval()
        if self.debug_decoders:
            self.decoder.eval()
        self.rssm.eval()

    def wm_to_train(self):
        self.encoder.train()
        if self.debug_decoders:
            self.decoder.train()
        self.rssm.train()

    def _backward(self, losses: dict, train: bool = True):
        metrics = {}

        # PoC: "recon" loss replaced by JEPA-style "pred" + SIGReg.
        # LeJEPA convention: λ·sigreg + (1-λ)·pred_per_dim, single trade-off param.
        # pred_loss_term = losses["recon"] + losses["reward"] + losses["discount"]
        # LEJEPA lambda * sigreg + (1 - lambda) * MSE
        # LEWM (lambda * sigreg + MSE) * beta_sigreg <--- we use this
        jepa_loss = self.sigreg_lambda * losses["sigreg"] + losses["pred"]
        pred_loss_term = self.lejepa_weight * jepa_loss + losses["reward"] + losses["discount"]
        if "emb_recon" in losses:
            pred_loss_term = pred_loss_term + self.emb_recon_weight * losses["emb_recon"]
        if "inv_dyn" in losses:
            pred_loss_term = pred_loss_term + self.inv_dyn_weight * losses["inv_dyn"]
        wm_loss = pred_loss_term * self.beta_pred + losses["kl"]
        if "recon" in losses:
            # Decoder inputs are detached, so this term only updates the debug
            # decoder; its weighting is irrelevant to the rest of the model.
            wm_loss = wm_loss + losses["recon"]
        if "embed_probe_recon" in losses:
            # Same: detached input, only updates the debug embed probe.
            wm_loss = wm_loss + losses["embed_probe_recon"]
        if "inv_dyn_probe" in losses:
            # Same: detached input, only updates the debug inverse-dynamics probe.
            wm_loss = wm_loss + losses["inv_dyn_probe"]
        metrics["wm_loss"] = wm_loss

        train_only_wm = "actor" not in losses
        if train_only_wm:
            total_loss = wm_loss
        else:
            ac_loss = losses["actor"] + losses["critic"]
            total_loss = wm_loss + ac_loss
        if not train:
            return metrics

        trained_modules = [
            self.encoder,
            self.emb_predictor,
            self.emb_decoder,
            self.inv_dyn,
            self.rssm,
            self.rewards,
            self.discounts,
        ]
        if self.debug_decoders:
            # Debug-only decoders, trained on detached features (see wm_forward_step).
            trained_modules += [self.decoder, self.embed_probe, self.inv_dyn_probe]
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
            reconstructions, rssm_out, embeds, target_embeds = self(batch)
            rewards = batch["rewards"]
            dones = batch["dones"]

            rewards = get_post_burn_in(burn_in_steps, rewards)
            dones = get_post_burn_in(burn_in_steps, dones)

            # --- JEPA-style predictor loss (replaces image reconstruction) ---
            # deter_t is computed from (prev_stoch, action_t, prev_deter) BEFORE
            # the posterior sees embed_t (see RSSM.wake_step). The codebase
            # invariant features[t] <-> s_t makes embed_t = encoder(s_t) the
            # correct prediction target for deter_t.
            # Target is stop-gradiented: the world model (predictor/deter) should
            # adapt to the encoder, not the other way around. The encoder is shaped
            # independently by SIGReg, inv_dyn, and reward — not by the quality of
            # a learned model that is itself still training.
            deter_post = get_post_burn_in(burn_in_steps, rssm_out["deter_state"])
            embeds_post = get_post_burn_in(burn_in_steps, embeds)
            # pred/emb_recon target. With two-view aug this is the CLEAN embedding
            # (the encoder is shaped on the augmented `embeds_post` via SIGReg/RSSM;
            # the targets stay clean). Without two-view it equals embeds_post.
            target_post = get_post_burn_in(burn_in_steps, target_embeds)
            is_first_post = get_post_burn_in(burn_in_steps, batch["is_first"])
            b = embeds_post.size(0)
            pred_embed = rearrange(
                self.emb_predictor(rearrange(deter_post, "b t d -> (b t) d")), "(b t) d -> b t d", b=b
            )
            # At is_first positions, prev_deter/prev_stoch and the (arrival-aligned)
            # action are all zeroed, so deter_t is a constant. The pred loss there
            # is irreducible — mask those positions out so the only signal in
            # loss["pred"] comes from steps the predictor can actually learn.
            per_pos_pred = ((pred_embed - target_post.detach()) ** 2).mean(dim=-1)
            valid_mask = 1.0 - is_first_post.float()
            loss["pred"] = (per_pos_pred * valid_mask).sum() / valid_mask.sum().clamp(min=1)

            # --- Embedding decoder loss (closed-loop, posterior side) ---
            # Reconstruct embed_t from the policy feature [deter, posterior_stoch].
            # Target is detached: SIGReg pins embed to unit-variance/full-rank, so it
            # can't collapse, and the gradient trains the posterior + decoder (not the
            # encoder target). Unlike pred, no is_first mask: the posterior saw embed_t
            # even at episode starts, so that step is reconstructable.
            if self.emb_recon_weight > 0:
                post_stoch_sample = get_post_burn_in(burn_in_steps, rssm_out["posterior"]["sample"])
                recon_input = torch.cat([deter_post, post_stoch_sample], dim=-1)
                emb_recon = rearrange(
                    self.emb_decoder(rearrange(recon_input, "b t d -> (b t) d")), "(b t) d -> b t d", b=b
                )
                loss["emb_recon"] = ((emb_recon - target_post.detach()) ** 2).mean()

            # --- Inverse-dynamics loss (encoder content grounding) ---
            # Predict action_t from (embed_{t-1}, embed_t). action[:, t] is the
            # arrival-aligned action driving the s_{t-1} -> s_t transition (see
            # RSSM.forward), so it pairs with embeds[:, t-1] and embeds[:, t].
            # Embeds are NOT detached: the gradient trains the encoder to retain
            # controllable content. Masking is on the destination index (1:): a
            # transition crosses an episode boundary iff its second frame is a
            # reset (is_first), which is also where the dummy/zeroed action sits.
            if self.inv_dyn_weight > 0:
                actions_post = get_post_burn_in(burn_in_steps, batch["actions"])
                inv_input = torch.cat([embeds_post[:, :-1], embeds_post[:, 1:]], dim=-1)
                inv_logits = rearrange(self.inv_dyn(rearrange(inv_input, "b t d -> (b t) d")), "(b t) d -> b t d", b=b)
                inv_targets = actions_post[:, 1:].argmax(dim=-1)
                inv_valid = 1.0 - is_first_post[:, 1:].float()
                inv_ce = nn.functional.cross_entropy(
                    rearrange(inv_logits.float(), "b t d -> (b t) d"),
                    rearrange(inv_targets, "b t -> (b t)"),
                    reduction="none",
                )
                inv_ce = rearrange(inv_ce, "(b t) -> b t", b=b)
                loss["inv_dyn"] = (inv_ce * inv_valid).sum() / inv_valid.sum().clamp(min=1)
                with torch.no_grad():
                    inv_correct = (inv_logits.argmax(dim=-1) == inv_targets).float()
                    inv_dyn_acc = (inv_correct * inv_valid).sum() / inv_valid.sum().clamp(min=1)

            embed_probe_recons = None
            if self.debug_decoders:
                # --- Debug-only reconstruction loss (detached inputs => trains only the decoder) ---
                loss["recon"] = self.decoder.reconstruction_loss(
                    get_post_burn_in(burn_in_steps, reconstructions),
                    get_post_burn_in(burn_in_steps, batch["states"]),
                )

                # --- Debug-only embed probe (detached embeds => trains only the probe) ---
                # Probes the online encoder output directly, bypassing the RSSM.
                embed_probe_recons = self.embed_probe(embeds.detach())
                loss["embed_probe_recon"] = self.embed_probe.reconstruction_loss(
                    get_post_burn_in(burn_in_steps, embed_probe_recons),
                    get_post_burn_in(burn_in_steps, batch["states"]),
                )

                # --- Debug-only inverse-dynamics probe (detached embeds => trains only the probe) ---
                # Mirrors the inv_dyn head on stop-grad'd embeds, so it measures how
                # decodable action_t is from (embed_{t-1}, embed_t) *without* shaping
                # the encoder. Compare inv_dyn_probe_acc at inv_dyn_weight 0 vs 0.05:
                # equal => the term adds no controllable content (redundant).
                inv_probe_actions = get_post_burn_in(burn_in_steps, batch["actions"])
                inv_probe_input = torch.cat([embeds_post[:, :-1], embeds_post[:, 1:]], dim=-1).detach()
                inv_probe_logits = rearrange(
                    self.inv_dyn_probe(rearrange(inv_probe_input, "b t d -> (b t) d")), "(b t) d -> b t d", b=b
                )
                inv_probe_targets = inv_probe_actions[:, 1:].argmax(dim=-1)
                inv_probe_valid = 1.0 - is_first_post[:, 1:].float()
                inv_probe_ce = nn.functional.cross_entropy(
                    rearrange(inv_probe_logits.float(), "b t d -> (b t) d"),
                    rearrange(inv_probe_targets, "b t -> (b t)"),
                    reduction="none",
                )
                inv_probe_ce = rearrange(inv_probe_ce, "(b t) -> b t", b=b)
                loss["inv_dyn_probe"] = (inv_probe_ce * inv_probe_valid).sum() / inv_probe_valid.sum().clamp(min=1)
                with torch.no_grad():
                    inv_probe_correct = (inv_probe_logits.argmax(dim=-1) == inv_probe_targets).float()
                    inv_dyn_probe_acc = (inv_probe_correct * inv_probe_valid).sum() / inv_probe_valid.sum().clamp(
                        min=1
                    )

        # SIGReg in fp32: Epps-Pulley uses cos/sin/exp over CF points and is
        # numerically sensitive. Flatten (B, T, D) -> (B*T, D) so each sample
        # is one embedding vector.
        embeds_flat = rearrange(embeds_post, "b t d -> (b t) d").float()
        loss["sigreg"] = self.sigreg(embeds_flat)
        metrics["embed_std"] = embeds_flat.std()
        metrics["embed_mean_abs"] = embeds_flat.abs().mean()
        metrics["pred_loss"] = loss["pred"].detach()
        metrics["sigreg_loss"] = loss["sigreg"].detach()
        if "emb_recon" in loss:
            metrics["emb_recon_loss"] = loss["emb_recon"].detach()
        if "inv_dyn" in loss:
            metrics["inv_dyn_loss"] = loss["inv_dyn"].detach()
            metrics["inv_dyn_acc"] = inv_dyn_acc
        if "inv_dyn_probe" in loss:
            metrics["inv_dyn_probe_loss"] = loss["inv_dyn_probe"].detach()
            metrics["inv_dyn_probe_acc"] = inv_dyn_probe_acc

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
            }
        )
        if self.debug_decoders:
            metrics["reconstruction_example"] = reconstructions[0, 0].detach().clone()
            metrics["embed_probe_reconstruction_example"] = embed_probe_recons[0, 0].detach().clone()

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

    def _update_target_network(self):
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
