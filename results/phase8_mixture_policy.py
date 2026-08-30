"""
phase8_mixture_policy.py
========================
PHASE 8C -- a MULTIMODAL (2-component tanh-squashed Gaussian mixture) actor for
SAC. This is the ONLY intended scientific change vs the CONTROL: the unimodal
squashed-Gaussian actor becomes a 2-component mixture so the policy can place
mass on BOTH the engine-OFF lobe and the engine-ON/LPS lobe of a bimodal Q.

Everything else is inherited unchanged from SB3 SAC:
  * same critic (twin-Q, min), same target construction, same tau/gamma
  * same MlpExtractor features, same net_arch [256,256]
  * same entropy-temperature auto-tuning (SAC uses -log_pi(a) as the entropy
    proxy; the mixture log-prob below is a correct single-sample estimate)
  * same observation, reward, env, action feasibility, replay buffer

NO benchmark / ECMS action is used anywhere. The mixture is trained purely by
the SAC objective.
"""
from __future__ import annotations

import numpy as np
import torch as th
from torch import nn

from stable_baselines3.sac.policies import Actor, SACPolicy, LOG_STD_MIN, LOG_STD_MAX
from stable_baselines3.common.torch_layers import create_mlp

K_COMPONENTS = 2
_LOG_2 = float(np.log(2.0))


def _atanh(x: th.Tensor) -> th.Tensor:
    return 0.5 * (th.log1p(x) - th.log1p(-x))


class MixtureActor(Actor):
    """2-component tanh-squashed diagonal-Gaussian mixture policy."""

    def __init__(self, *args, n_components: int = K_COMPONENTS, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_components = int(n_components)
        last = self.net_arch[-1] if len(self.net_arch) > 0 else self.features_dim
        adim = self.action_space.shape[0]
        self._adim = adim
        # component means / log-stds / mixing logits (base self.mu/self.log_std unused)
        self.mix_mu = nn.Linear(last, adim * self.n_components)
        self.mix_log_std = nn.Linear(last, adim * self.n_components)
        self.mix_logits = nn.Linear(last, self.n_components)
        # small init so the two components start near-identical, then separate
        nn.init.zeros_(self.mix_logits.weight); nn.init.zeros_(self.mix_logits.bias)

    # ---- raw params -------------------------------------------------------- #
    def _mixture_params(self, obs):
        feat = self.extract_features(obs, self.features_extractor)
        z = self.latent_pi(feat)
        b = z.shape[0]
        mu = self.mix_mu(z).view(b, self.n_components, self._adim)
        log_std = th.clamp(self.mix_log_std(z).view(b, self.n_components, self._adim),
                           LOG_STD_MIN, LOG_STD_MAX)
        logw = th.log_softmax(self.mix_logits(z), dim=1)            # [b, K]
        return mu, log_std, logw

    # ---- SB3 compatibility: return the DOMINANT component's params -------- #
    def get_action_dist_params(self, obs):
        mu, log_std, logw = self._mixture_params(obs)
        k = logw.argmax(dim=1)                                       # [b]
        idx = k.view(-1, 1, 1).expand(-1, 1, self._adim)
        mu_d = mu.gather(1, idx).squeeze(1)
        ls_d = log_std.gather(1, idx).squeeze(1)
        return mu_d, ls_d, {}

    # ---- helpers -------------------------------------------------------- #
    def _component_logprob_u(self, u, mu, log_std):
        # u: [b, adim]  mu/log_std: [b, K, adim]  -> [b, K]
        u = u.unsqueeze(1)
        var = th.exp(2.0 * log_std)
        lp = -0.5 * (((u - mu) ** 2) / var) - log_std - 0.5 * np.log(2.0 * np.pi)
        return lp.sum(dim=2)

    def _tanh_correction(self, u):
        # sum_adim log(1 - tanh(u)^2) ; stable form
        return (2.0 * (_LOG_2 - u - nn.functional.softplus(-2.0 * u))).sum(dim=1)

    def _sample(self, mu, log_std, logw, deterministic):
        b, K, adim = mu.shape
        if deterministic:
            k = logw.argmax(dim=1)
            idx = k.view(-1, 1, 1).expand(-1, 1, adim)
            u = mu.gather(1, idx).squeeze(1)
            return u, k
        k = th.distributions.Categorical(logits=logw).sample()      # [b]
        idx = k.view(-1, 1, 1).expand(-1, 1, adim)
        mu_k = mu.gather(1, idx).squeeze(1)
        std_k = th.exp(log_std.gather(1, idx).squeeze(1))
        eps = th.randn_like(mu_k)
        u = mu_k + std_k * eps
        return u, k

    # ---- SB3 API -------------------------------------------------------- #
    def forward(self, obs, deterministic: bool = False):
        mu, log_std, logw = self._mixture_params(obs)
        u, _ = self._sample(mu, log_std, logw, deterministic)
        return th.tanh(u)

    def action_log_prob(self, obs):
        mu, log_std, logw = self._mixture_params(obs)
        u, _ = self._sample(mu, log_std, logw, deterministic=False)
        comp_lp = self._component_logprob_u(u, mu, log_std)          # [b, K]
        log_pu = th.logsumexp(logw + comp_lp, dim=1)                 # [b]
        log_pa = log_pu - self._tanh_correction(u)                   # [b]
        return th.tanh(u), log_pa

    def _predict(self, observation, deterministic: bool = False):
        return self(observation, deterministic)


class MixtureSACPolicy(SACPolicy):
    """SACPolicy that builds a MixtureActor instead of the unimodal Actor."""

    def make_actor(self, features_extractor=None):
        actor_kwargs = self._update_features_extractor(self.actor_kwargs, features_extractor)
        return MixtureActor(**actor_kwargs).to(self.device)


# convenience for the forensic tools: mixture-aware readouts
def mixture_readout(model, obs_np):
    """Return per-component (weight, mean_tanh, sigma) for a single obs row."""
    actor = model.actor
    ot = th.as_tensor(np.asarray(obs_np).reshape(1, -1)).float().to(model.device)
    with th.no_grad():
        mu, log_std, logw = actor._mixture_params(ot)
    w = th.exp(logw).cpu().numpy().ravel()
    mu = mu.cpu().numpy().reshape(-1)
    sig = np.exp(log_std.cpu().numpy().reshape(-1))
    return [dict(weight=float(w[k]), mean=float(np.tanh(mu[k])), sigma=float(sig[k]))
            for k in range(len(w))]
