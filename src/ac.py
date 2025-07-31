import torch
from torch import nn
from torch.nn import functional as F
from torch import distributions as torchd

import nets
import utils


class ActorCriticPolicy(nn.Module):
    """Actor-Critic policy for discrete action spaces."""

    def __init__(self, x_dim, a_dim, actor, critic, *, compile_, device=None):
        super().__init__()
        self.actor = compile_(Actor(x_dim, a_dim, **actor, device=device))
        self.critic = compile_(nets.ScalarMLP(x_dim, **critic, device=device))

    @torch.no_grad()
    def forward(self, x):
        assert not self.actor.training
        a = self.actor(x)
        return a

    def create_trainer(self, config, *, total_its, rng, autocast, compile_):
        return ActorCriticTrainer(self, **config, total_its=total_its, rng=rng, autocast=autocast, compile_=compile_)


class Actor(nn.Module):
    """Actor for discrete action spaces."""

    def __init__(
        self,
        x_dim,
        a_dim,
        dims,
        norm,
        act,
        init,
        out_init,
        dist="normal",
        std=1.0,
        min_std=0.1,
        max_std=10.0,
        absmax=None,
        temp=0.1,
        unimix_ratio=0.01,
        symlog_inputs=False,
        device=None,
    ):
        super().__init__()
        self._dist = dist
        self._std = std if isinstance(std, str) else torch.tensor((std,), device=device)
        self._min_std = min_std
        self._max_std = max_std
        self._absmax = absmax
        self._temp = temp
        self._unimix_ratio = unimix_ratio
        self._symlog_inputs = symlog_inputs

        modules, backbone_dim = nets.mlp(x_dim, dims, norm, act, init, out_bias=True, out_norm=True, device=device)
        self.layers = nn.Sequential(*modules)

        self.mean_layer = nets.init_(nn.Linear(backbone_dim, a_dim, device=device), out_init)
        if self._std == "learned":
            assert dist in ("tanh_normal", "normal", "trunc_normal", "huber"), dist
            self.std_layer = nets.init_(nn.Linear(backbone_dim, a_dim, device=device), out_init)

    def get_stats(self, inp, full_precision=False):
        x = inp
        if self._symlog_inputs:
            x = utils.symlog(x)
        x = self.layers(x)
        mean = self.mean_layer(x)
        std = self.std_layer(x) if self._std == "learned" else self._std

        if full_precision:
            mean = mean.type(torch.float32)
            std = std.type(torch.float32)

        dist = self.dist(self._dist, mean, std, mean.shape)
        return dist

    @torch.no_grad()
    def predict(self, dist):
        return dist.sample()

    @torch.no_grad()
    def forward(self, x):
        dist = self.get_stats(x)
        sample = self.predict(dist)
        return sample

    def reinforce_loss(self, stats, a, adv, mask):
        assert self.training
        log_probs = stats.log_prob(a)

        dtype = log_probs.dtype
        adv, mask = adv.type(dtype), mask.type(dtype)
        
        reinforce_loss = mask.mean(-(adv * log_probs))
        return reinforce_loss

    def entropy_loss(self, stats, a, mask):
        assert self.training
        log_probs = stats.log_prob(a)
        neg_entropy = log_probs.exp() * log_probs
        loss = (mask.type(neg_entropy.dtype)).mean(neg_entropy)
        return loss

    def dist(self, dist, mean, std, shape):
        if dist == "tanh_normal":
            mean = torch.tanh(mean)
            std = F.softplus(std) + self._min_std
            dist = torchd.normal.Normal(mean, std)
            dist = torchd.transformed_distribution.TransformedDistribution(dist, utils.TanhBijector())
            dist = torchd.independent.Independent(dist, 1)
            dist = utils.SampleDist(dist)
        elif dist == "normal":
            std = (self._max_std - self._min_std) * torch.sigmoid(std + 2.0) + self._min_std
            dist = torchd.normal.Normal(torch.tanh(mean), std)
            dist = utils.ContDist(torchd.independent.Independent(dist, 1), absmax=self._absmax)
        elif dist == "normal_std_fixed":
            dist = torchd.normal.Normal(mean, self._std)
            dist = utils.ContDist(torchd.independent.Independent(dist, 1), absmax=self._absmax)
        elif dist == "trunc_normal":
            mean = torch.tanh(mean)
            std = 2 * torch.sigmoid(std / 2) + self._min_std
            dist = utils.SafeTruncatedNormal(mean, std, -1, 1)
            dist = utils.ContDist(torchd.independent.Independent(dist, 1), absmax=self._absmax)
        elif dist == "onehot":
            dist = utils.OneHotDist(mean, unimix_ratio=self._unimix_ratio)
        elif dist == "onehot_gumble":
            dist = utils.ContDist(torchd.gumbel.Gumbel(mean, 1 / self._temp), absmax=self._absmax)
        elif dist == "huber":
            dist = utils.ContDist(
                torchd.independent.Independent(
                    utils.UnnormalizedHuber(mean, std, 1.0),
                    len(shape),
                    absmax=self._absmax,
                )
            )
        elif dist == "binary":
            dist = utils.Bernoulli(torchd.independent.Independent(torchd.bernoulli.Bernoulli(logits=mean), len(shape)))
        elif dist == "symlog_disc":
            dist = utils.DiscDist(logits=mean, device=self._device)
        elif dist == "symlog_mse":
            dist = utils.SymlogDist(mean)
        else:
            raise NotImplementedError(dist)
        return dist


class ActorCriticTrainer:
    """Trainer for Actor-Critic policies."""

    def __init__(
        self,
        policy,
        actor_optimizer,
        critic_optimizer,
        reward_act,
        return_norm,
        gamma,
        lmbda,
        entropy_coef,
        target_decay,
        target_returns,
        target_coef,
        target_every,
        *,
        total_its,
        rng,
        autocast,
        compile_,
    ):

        self.policy = policy
        self.reward_act = nets.activation(reward_act)
        self.return_norm = ReturnNorm(device=utils.device(policy)) if return_norm else None

        self.actor_optimizer = nets.Optimizer(policy.actor, **actor_optimizer, total_its=total_its, autocast=autocast)
        self.critic_optimizer = nets.Optimizer(policy.critic, **critic_optimizer, total_its=total_its, autocast=autocast)

        self.gamma = gamma
        self.lmbda = lmbda
        self.entropy_coef = entropy_coef

        if target_decay < 1 and target_every > 0:
            self.has_target = True
            self.target_critic = utils.target_network(policy.critic)
            self.target_every = target_every
            self.target_decay = target_decay
            self.target_returns = target_returns
            self.target_coef = target_coef
            self._target_it = 0
        else:
            self.has_target = False

        self.total_its = total_its
        self.rng = rng
        self.autocast = autocast

        self._optimize = compile_(self._optimize)

    def _optimize(self, x, a, target_v, ret, adv, seq_mask, it):
        # This method is compiled with torch.compile

        actor, critic = self.policy.actor, self.policy.critic

        with self.autocast():
            actor_stats = actor.get_stats(x, full_precision=True)
            entropy_loss = actor.entropy_loss(actor_stats, a, seq_mask)
            reinforce_loss = actor.reinforce_loss(actor_stats, a, adv, seq_mask)
            actor_loss = reinforce_loss + self.entropy_coef * entropy_loss

            metrics = {
                'reinforce_loss': reinforce_loss,
                'entropy_loss': entropy_loss,
                'actor_loss': actor_loss,
            }

        batch_size = x.shape[0]
        self.actor_optimizer.step(actor_loss, batch_size, it)

        with self.autocast():
            critic_stats = critic.get_stats(x, full_precision=True)
            return_loss = critic.loss(critic_stats, ret, seq_mask)

            if self.has_target:
                target_loss = critic.loss(critic_stats, target_v, seq_mask)
                critic_loss = return_loss + self.target_coef * target_loss
                metrics.update(
                    {
                        'return_loss': return_loss,
                        'target_loss': target_loss,
                        'critic_loss': critic_loss,
                    }
                )
            else:
                critic_loss = return_loss
                metrics['critic_loss'] = critic_loss

        self.critic_optimizer.step(critic_loss, batch_size, it)
        return metrics

    def train(self, it, xs, final_x, as_, next_rs, next_terms):
        policy = self.policy
        return_norm = self.return_norm
        critic = policy.critic

        critic.eval()
        with self.autocast():
            with torch.no_grad():
                vs = utils.apply_seq(critic, xs)

                if self.has_target:
                    self.target_critic.eval()
                    target_vs = utils.apply_seq(self.target_critic, xs)

                if self.has_target and self.target_returns:
                    final_target_v = self.target_critic(final_x)
                    next_vs = torch.cat([target_vs[1:], final_target_v.unsqueeze(0)], 0)
                    baselines = target_vs  # also use target baseline
                else:
                    # reuse the same values; this is actually not correct, since next_x != x when next_term is True
                    # however, this should not be a problem since `masks` is zero when next_term is True
                    final_v = critic(final_x)
                    next_vs = torch.cat([vs[1:], final_v.unsqueeze(0)], 0)
                    baselines = vs

                next_masks = utils.get_mask(1 - next_terms.to(next_rs.dtype))
                seq_masks = next_masks.cumulative_sequence(shifted=True)
                next_gammas = next_masks.values * self.gamma
                rets = utils.lambda_return(self.reward_act(next_rs), next_vs, next_gammas, self.lmbda)

                if return_norm is not None:
                    return_norm.update(rets)
                    adv = return_norm(rets) - return_norm(baselines)
                else:
                    adv = rets - baselines

                x, a, ret, adv, seq_mask = [utils.flatten_seq(val) for val in (xs, as_, rets, adv, seq_masks)]
                if self.has_target:
                    target_v = utils.flatten_seq(target_vs)

                batch_metrics = {
                    'rewards': next_rs.mean(),
                    'terminals': next_terms.float().mean(),
                    'values': vs.mean(),
                    'advantages': adv.mean(),
                    'returns': rets.mean(),
                }

        policy.train()
        metrics = self._optimize(x, a, target_v, ret, adv, seq_mask, it)

        if self.has_target:
            while (it - self._target_it) >= self.target_every:
                utils.ema_update(critic, self.target_critic, self.target_decay)
                self._target_it += self.target_every

        metrics.update(batch_metrics)
        return metrics


# adopted from https://github.com/danijar/dreamerv3/blob/main/dreamerv3/jaxutils.py
class ReturnNorm(nn.Module):

    def __init__(self, low_percentile=5.0, high_percentile=95.0, decay=0.99, maximum=1.0, device=None):
        super().__init__()
        self.register_buffer('inv_max', torch.tensor(1 / maximum, device=device))
        self.register_buffer('q', torch.tensor([low_percentile / 100, high_percentile / 100], device=device))
        self.register_buffer('decay', torch.tensor(decay, device=device))
        self.register_buffer('low', torch.zeros(1, device=device))
        self.register_buffer('high', torch.zeros(1, device=device))
        self.register_buffer('inv_scale', torch.zeros(1, device=device))

    def update(self, ret):
        ret = ret.type(torch.float32)
        ret_low, ret_high = torch.quantile(ret.flatten(), self.q)
        decay = self.decay
        self.low.data = decay * self.low + (1 - decay) * ret_low
        self.high.data = decay * self.high + (1 - decay) * ret_high
        self.inv_scale.data = torch.maximum(self.inv_max, self.high - self.low)

    def forward(self, ret):
        return (ret - self.low) / self.inv_scale
