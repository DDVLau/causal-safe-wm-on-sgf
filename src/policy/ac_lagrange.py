import torch
from torch import nn

import nets
import utils
from policy.ac import Actor, SafeReturnNorm


class ActorCriticLagrangePolicy(nn.Module):
    """Actor-Critic policy for discrete action spaces."""

    def __init__(self, x_dim, a_dim, actor, reward_critic, cost_critic, *, compile_, device=None):
        super().__init__()
        self.actor = compile_(Actor(x_dim, a_dim, **actor, device=device))
        self.reward_critic = compile_(nets.ScalarMLP(x_dim, **reward_critic, device=device))
        self.cost_critic = compile_(nets.ScalarMLP(x_dim, **cost_critic, device=device))

    @torch.no_grad()
    def forward(self, x):
        assert not self.actor.training
        a = self.actor(x)
        return a

    def step(self, x, deterministic=False):
        with torch.no_grad():
            if deterministic:
                dist = self.actor.get_stats(x)
                act = dist.mode if hasattr(dist, "mode") else dist.mean
            else:
                act = self.actor(x)

            log_prob = self.actor.get_stats(x).log_prob(act)
            value_r = self.reward_critic(x)
            value_c = self.cost_critic(x)

            return act, log_prob, value_r, value_c

    def create_trainer(self, config, *, total_its, rng, autocast, compile_):
        return ActorCriticLagrangeTrainer(self, **config, total_its=total_its, rng=rng, autocast=autocast, compile_=compile_)


class ActorCriticLagrangeTrainer:
    """Trainer for Actor-Critic policies with naive Lagrange constraints."""

    def __init__(
        self,
        policy,
        actor_optimizer,
        critic_optimizer,
        lagrange_params,
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
        self.lagrangian_multiplier = torch.nn.Parameter(torch.as_tensor(lagrange_params["lagrangian_multiplier_init"]), requires_grad=True)
        self.return_norm = SafeReturnNorm(device=utils.device(policy)) if return_norm else None

        self.actor_optimizer = nets.Optimizer(policy.actor, **actor_optimizer, total_its=total_its, autocast=autocast)
        self.reward_critic_optimizer = nets.Optimizer(policy.reward_critic, **critic_optimizer, total_its=total_its, autocast=autocast)
        self.cost_critic_optimizer = nets.Optimizer(policy.cost_critic, **critic_optimizer, total_its=total_its, autocast=autocast)
        self.lambda_optimizer = nets.Optimizer(
            torch.nn.ParameterDict({"bias": self.lagrangian_multiplier}), **lagrange_params["optimizer"], total_its=total_its, autocast=autocast
        )  # cheating the grouping of nets.Optimizer

        self.gamma = gamma
        self.lmbda = lmbda
        self.entropy_coef = entropy_coef
        self.cost_limit = lagrange_params["cost_limit"]
        self.lagrangian_upper_bound = lagrange_params["lagrangian_upper_bound"]

        if target_decay < 1 and target_every > 0:
            self.has_target = True
            self.target_reward_critic = utils.target_network(policy.reward_critic)
            self.target_cost_critic = utils.target_network(policy.cost_critic)
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

    def _optimize(self, x, a, target_v_r, target_v_c, ret_r, ret_c, adv_r, adv_c, seq_mask, it):
        # This method is compiled with torch.compile

        actor, reward_critic, cost_critic = self.policy.actor, self.policy.reward_critic, self.policy.cost_critic

        with self.autocast():
            actor_stats = actor.get_stats(x, full_precision=True)
            penalty = self.lagrangian_multiplier.item()
            adv = (adv_r - penalty * adv_c) / (1 + penalty)
            reinforce_loss = actor.reinforce_loss(actor_stats, a, adv, seq_mask)
            actor_loss = reinforce_loss
            lambda_loss = self.compute_lambda_loss(ret_c.mean())

            metrics = {
                "reinforce_loss": reinforce_loss,
                "lagrange_lambda_loss": lambda_loss,
                "actor_loss": actor_loss,
            }

        batch_size = x.shape[0]
        # Lagrange multiplier
        lagrange_norm = self.lambda_optimizer.step(lambda_loss, batch_size, it)
        self.lagrangian_multiplier.data.clamp_(0.0, self.lagrangian_upper_bound)

        # Actor
        actor_norm = self.actor_optimizer.step(actor_loss, batch_size, it)

        with self.autocast():
            reward_critic_stats = reward_critic.get_stats(x, full_precision=True)
            reward_return_loss = reward_critic.loss(reward_critic_stats, ret_r, seq_mask)

            cost_critic_stats = cost_critic.get_stats(x, full_precision=True)
            cost_return_loss = cost_critic.loss(cost_critic_stats, ret_c, seq_mask)

            if self.has_target:
                reward_target_loss = reward_critic.loss(reward_critic_stats, target_v_r, seq_mask)
                cost_target_loss = cost_critic.loss(cost_critic_stats, target_v_c, seq_mask)
                reward_critic_loss = reward_return_loss + self.target_coef * reward_target_loss
                cost_critic_loss = cost_return_loss + self.target_coef * cost_target_loss
                metrics.update(
                    {
                        "reward_return_loss": reward_return_loss,
                        "reward_target_loss": reward_target_loss,
                        "reward_critic_loss": reward_critic_loss,
                        "cost_return_loss": cost_return_loss,
                        "cost_target_loss": cost_target_loss,
                        "cost_critic_loss": cost_critic_loss,
                    }
                )
            else:
                reward_critic_loss = reward_return_loss
                cost_critic_loss = cost_return_loss
                metrics.update({"reward_critic_loss": reward_critic_loss, "cost_critic_loss": cost_critic_loss})

        # Critic
        r_critic_norm = self.reward_critic_optimizer.step(reward_critic_loss, batch_size, it)
        c_critic_norm = self.cost_critic_optimizer.step(cost_critic_loss, batch_size, it)
        metrics.update({**actor_norm, **r_critic_norm, **c_critic_norm, **lagrange_norm})
        return metrics

    def train(self, it, xs, final_x, as_, next_rs, next_cs, next_terms):
        """Training step for Actor-Critic Lagrange."""
        policy = self.policy
        return_norm = self.return_norm
        reward_critic = policy.reward_critic
        cost_critic = policy.cost_critic

        reward_critic.eval()
        cost_critic.eval()

        with self.autocast():
            with torch.no_grad():
                vs_r = utils.apply_seq(reward_critic, xs)
                vs_c = utils.apply_seq(cost_critic, xs)

                if self.has_target:
                    self.target_reward_critic.eval()
                    target_vs_r = utils.apply_seq(self.target_reward_critic, xs)
                    target_vs_c = utils.apply_seq(self.target_cost_critic, xs)

                if self.has_target and self.target_returns:
                    final_target_v_r = self.target_reward_critic(final_x)
                    final_target_v_c = self.target_cost_critic(final_x)
                    next_vs_r = torch.cat([target_vs_r[1:], final_target_v_r.unsqueeze(0)], 0)
                    next_vs_c = torch.cat([target_vs_c[1:], final_target_v_c.unsqueeze(0)], 0)
                    baselines_r, baselines_c = target_vs_r, target_vs_c
                else:
                    final_v_r = reward_critic(final_x)
                    final_v_c = cost_critic(final_x)
                    next_vs_r = torch.cat([vs_r[1:], final_v_r.unsqueeze(0)], 0)
                    next_vs_c = torch.cat([vs_c[1:], final_v_c.unsqueeze(0)], 0)
                    baselines_r, baselines_c = vs_r, vs_c

                next_masks = utils.get_mask(1 - next_terms.to(next_rs.dtype))
                seq_masks = next_masks.cumulative_sequence(shifted=True)
                next_gammas = next_masks.values * self.gamma

                # Compute returns
                rets_r = utils.lambda_return(self.reward_act(next_rs), next_vs_r, next_gammas, self.lmbda)
                rets_c = utils.lambda_return(next_cs, next_vs_c, next_gammas, self.lmbda)

                # Compute advantages
                if return_norm is not None:
                    return_norm.update(rets_r, rets_c)
                    adv_r = return_norm.normalize_rewards(rets_r) - return_norm.normalize_rewards(vs_r)
                    adv_c = return_norm.normalize_costs(rets_c) - return_norm.normalize_costs(vs_c)
                else:
                    adv_r = rets_r - baselines_r
                    adv_c = rets_c - baselines_c

                # Flatten sequences
                x, a, vs_r, vs_c, ret_r, ret_c, adv_r, adv_c, seq_mask = [utils.flatten_seq(val) for val in (xs, as_, vs_r, vs_c, rets_r, rets_c, adv_r, adv_c, seq_masks)]

                batch_metrics = {
                    "rewards": next_rs.mean(),
                    "costs": next_cs.mean(),
                    "terminals": next_terms.float().mean(),
                    "reward_values": vs_r.mean(),
                    "cost_values": vs_c.mean(),
                    "reward_advantages": adv_r.mean(),
                    "cost_advantages": adv_c.mean(),
                    "r_returns": rets_r.mean(),
                    "c_returns": rets_c.mean(),
                }

        policy.train()
        metrics = self._optimize(x, a, vs_r, vs_c, ret_r, ret_c, adv_r, adv_c, seq_mask, it)

        if self.has_target:
            while (it - self._target_it) >= self.target_every:
                utils.ema_update(reward_critic, self.target_reward_critic, self.target_decay)
                utils.ema_update(cost_critic, self.target_cost_critic, self.target_decay)
                self._target_it += self.target_every

        metrics.update(batch_metrics)
        return metrics

    def compute_lambda_loss(self, mean_ep_cost: float) -> torch.Tensor:
        return -self.lagrangian_multiplier * (mean_ep_cost - self.cost_limit)
