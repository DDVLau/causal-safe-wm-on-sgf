# CPO with style of reinforce loss
# Reference: cpo.py from https://github.com/PKU-Alignment/omnisafe
import torch
import numpy as np
from torch import nn

import nets
import utils
from ac import Actor


class CPOPolicy(nn.Module):
    """CPO policy with actor and dual critics for reward and cost."""

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
        """Step function compatible with original CPO interface."""
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
        return CPOTrainer(self, **config, total_its=total_its, rng=rng, autocast=autocast, compile_=compile_)


class CPOTrainer:
    """CPO trainer implementing constraint policy optimization."""

    def __init__(
        self,
        policy,
        actor_optimizer,
        critic_optimizer,
        cpo_params,
        reward_act,
        return_norm,
        gamma,
        lmbda,
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
        self.return_norm = CPOReturnNorm(device=utils.device(policy)) if return_norm else None

        self.actor_optimizer = nets.Optimizer(policy.actor, **actor_optimizer, total_its=total_its, autocast=autocast)
        self.reward_critic_optimizer = nets.Optimizer(policy.reward_critic, **critic_optimizer, total_its=total_its, autocast=autocast)
        self.cost_critic_optimizer = nets.Optimizer(policy.cost_critic, **critic_optimizer, total_its=total_its, autocast=autocast)

        self.gamma = gamma
        self.lmbda = lmbda
        self.target_kl = cpo_params["target_kl"]
        self.cost_limit = cpo_params["cost_limit"]
        self.cg_iters = cpo_params["cg_iters"]
        self.backtrack_iters = cpo_params["backtrack_iters"]
        self.backtrack_coef = cpo_params["backtrack_coef"]

        self.total_its = total_its
        self.rng = rng
        self.autocast = autocast

        self._optimize = compile_(self._optimize)

    def get_flat_params_from(self, model):
        flat_params = []
        for _, param in model.named_parameters():
            if param.requires_grad:
                data = param.data
                data = data.view(-1)
                flat_params.append(data)
        assert flat_params, "No gradients were found in model parameters."
        return torch.cat(flat_params)

    def set_param_values_to_model(self, model, vals):
        assert isinstance(vals, torch.Tensor)
        i = 0
        for _, param in model.named_parameters():
            if param.requires_grad:
                orig_size = param.size()
                size = np.prod(list(param.size()))
                new_values = vals[i : int(i + size)]
                new_values = new_values.view(orig_size)
                param.data = new_values
                i += int(size)
        assert i == len(vals), f"Lengths do not match: {i} vs. {len(vals)}"

    def get_flat_gradients_from(self, model):
        grads = []
        for _, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                grad = param.grad
                grads.append(grad.view(-1))
        assert grads, "No gradients were found in model parameters."
        return torch.cat(grads)

    def conjugate_gradients(self, Avp, fvp_obs, b, nsteps, residual_tol=1e-10, eps=1e-6):
        x = torch.zeros_like(b)
        r = b.clone() - Avp(x, fvp_obs)
        p = r.clone()
        rdotr = torch.dot(r, r)

        for i in range(nsteps):
            _Avp = Avp(p, fvp_obs)
            alpha = rdotr / (torch.dot(p, _Avp) + eps)
            x += alpha * p
            r -= alpha * _Avp
            new_rdotr = torch.dot(r, r)
            mu = new_rdotr / (rdotr + eps)
            p = r + mu * p
            rdotr = new_rdotr
            if rdotr < residual_tol:
                break
        return x

    # Seems good until here
    def _optimize(self, x, a, target_v_r, target_v_c, ret_r, ret_c, adv_r, adv_c, seq_mask, it, ep_costs, xs):
        actor, reward_critic, cost_critic = self.policy.actor, self.policy.reward_critic, self.policy.cost_critic

        # Fisher Vector Product function
        def fvp(params, fvp_obs):
            actor.zero_grad()
            current_distribution = actor.get_stats(fvp_obs)
            with torch.no_grad():
                old_distribution = actor.get_stats(fvp_obs)
            kl_div = torch.distributions.kl.kl_divergence(old_distribution._dist, current_distribution._dist).mean()

            kl_grad = torch.autograd.grad(kl_div, actor.parameters(), create_graph=True)
            flat_kl_grad = torch.cat([grad.view(-1) for grad in kl_grad])
            kl_v = (flat_kl_grad * params).sum()
            grads = torch.autograd.grad(kl_v, actor.parameters())
            flat_grad_grad_kl = torch.cat([grad.contiguous().view(-1) for grad in grads])
            return flat_grad_grad_kl + params * 0.1

        # Store old policy parameters
        old_params = self.get_flat_params_from(actor)

        with self.autocast():
            # Get old policy distribution for KL computation
            old_stats = actor.get_stats(x, full_precision=True)

            # Compute policy losses
            actor_stats = actor.get_stats(x, full_precision=True)
            reward_loss = actor.reinforce_loss(actor_stats, a, adv_r, seq_mask)
            # minimal cost
            cost_loss = -actor.reinforce_loss(actor_stats, a, adv_c, seq_mask)

            # KL divergence
            kl_div = torch.distributions.kl.kl_divergence(old_stats._dist, actor_stats._dist).mean()

            metrics = {
                "reward_loss": reward_loss.item(),
                "cost_loss": cost_loss.item(),
                "kl_div": kl_div.item(),
            }

        # Reward gradients
        actor.zero_grad()
        reward_loss.backward(retain_graph=True)
        policy_grads = -self.get_flat_gradients_from(actor)  # Note this
        loss_reward_before = reward_loss.item()

        # Solve for step direction using CPO update rule
        reward_step = self.conjugate_gradients(fvp, x, policy_grads, self.cg_iters)
        xHx = torch.dot(reward_step, fvp(reward_step, x))
        lm = torch.sqrt(2 * self.target_kl / xHx)

        # Cost gradients
        actor.zero_grad()
        cost_loss.backward(retain_graph=True)
        cost_grads = self.get_flat_gradients_from(actor)
        loss_cost_before = cost_loss.item()

        # Compute constraint violation and cost gradients norm
        constraint_violation = ep_costs - self.cost_limit
        ep_costs_mean = constraint_violation.mean()
        cost_gradient_norm = torch.dot(cost_grads, cost_grads)

        if cost_gradient_norm <= 1e-6 and ep_costs_mean < 0:
            A = torch.zeros(1, device=x.device)
            B = torch.zeros(1, device=x.device)
            optim_case = 4
        else:
            cost_step = self.conjugate_gradients(fvp, x, cost_grads, self.cg_iters)
            r = torch.dot(policy_grads, cost_step)
            s = torch.dot(cost_grads, cost_step)

            A = xHx - r**2 / (s + 1e-8)
            B = 2 * self.target_kl - ep_costs_mean**2 / (s + 1e-8)

            if ep_costs_mean < 0 and B < 0:
                optim_case = 3
            elif ep_costs_mean < 0 <= B:
                optim_case = 2
            elif ep_costs_mean >= 0 and B >= 0:
                optim_case = 1
            else:
                optim_case = 0

        if optim_case in (3, 4):
            lm = torch.sqrt(2 * self.target_kl / (xHx + 1e-8))
            nu_star = torch.zeros(1, device=x.device)
            lambda_star = 1 / (lm + 1e-8)
            final_step = reward_step * lm

        elif optim_case in (1, 2):
            # Constrained cases - need quadratic optimization
            lambda_a = torch.sqrt(A / B)
            lambda_b = torch.sqrt(xHx / (2 * self.target_kl))

            r_val = r.item()
            eps_cost = ep_costs_mean + 1e-8

            if ep_costs_mean < 0:
                lambda_a_star = torch.clamp(lambda_a, torch.tensor(0.0, device=lambda_a.device), torch.tensor(r_val / eps_cost, device=lambda_a.device))
                lambda_b_star = torch.clamp(lambda_b, torch.tensor(r_val / eps_cost, device=lambda_b.device), torch.tensor(float("inf"), device=lambda_b.device))
            else:
                lambda_a_star = torch.clamp(lambda_a, torch.tensor(r_val / eps_cost, device=lambda_a.device), torch.tensor(float("inf"), device=lambda_a.device))
                lambda_b_star = torch.clamp(lambda_b, torch.tensor(0.0, device=lambda_b.device), torch.tensor(r_val / eps_cost, device=lambda_b.device))

            # Choose optimal lambda
            f_a = -0.5 * (A / (lambda_a_star + 1e-8) + B * lambda_a_star) - r * ep_costs_mean / (s + 1e-8)
            f_b = -0.5 * (xHx / (lambda_b_star + 1e-8) + 2 * self.target_kl * lambda_b_star)

            lambda_star = lambda_a_star if f_a >= f_b else lambda_b_star
            nu_star = torch.clamp(lambda_star * ep_costs_mean - r, min=0) / (s + 1e-8)

            final_step = (reward_step - nu_star * cost_step) / (lambda_star + 1e-8)

        else:
            nu_star = torch.sqrt(2 * self.target_kl / (s + 1e-8))
            lambda_star = torch.zeros(1, device=x.device)
            final_step = -nu_star * cost_step

        # Line search
        step_size = 1.0
        for step in range(self.backtrack_iters):
            new_params = old_params + step_size * final_step
            self.set_param_values_to_model(actor, new_params)
            acceptance_step = step + 1

            with torch.no_grad():
                try:
                    new_stats = actor.get_stats(x)
                    loss_r = actor.reinforce_loss(actor_stats, a, adv_r, seq_mask)
                except ValueError:
                    step_size *= self.backtrack_coef
                    continue

                new_stats = actor.get_stats(x)
                loss_c = -actor.reinforce_loss(actor_stats, a, adv_c, seq_mask)
                new_kl = torch.distributions.kl.kl_divergence(old_stats._dist, new_stats._dist).mean()
                loss_reward_improve = loss_reward_before - loss_r.item()
                loss_cost_diff = loss_c.item() - loss_cost_before

                if not torch.isfinite(new_kl):
                    continue

                if not (loss_reward_improve < 0 if optim_case > 1 else False) and not (loss_cost_diff > max(-ep_costs_mean, 0)) and not (new_kl > self.target_kl):
                    break
            step_size *= self.backtrack_coef
        else:
            final_step = torch.zeros_like(final_step)
            acceptance_step = 0

        new_params = old_params + step_size * final_step
        self.set_param_values_to_model(actor, new_params)

        # Update critics
        batch_size = x.shape[0]

        with self.autocast():
            reward_critic_stats = reward_critic.get_stats(x, full_precision=True)
            reward_critic_loss = reward_critic.loss(reward_critic_stats, ret_r, seq_mask)

            cost_critic_stats = cost_critic.get_stats(x, full_precision=True)
            cost_critic_loss = cost_critic.loss(cost_critic_stats, ret_c, seq_mask)

        reward_critic_norm = self.reward_critic_optimizer.step(reward_critic_loss, batch_size, it)
        cost_critic_norm = self.cost_critic_optimizer.step(cost_critic_loss, batch_size, it)

        metrics.update(
            {
                "reward_critic_loss": reward_critic_loss,
                "cost_critic_loss": cost_critic_loss,
                "constraint_violation": ep_costs_mean.item(),
                "lambda_star": lambda_star.item(),
                "acceptance_step": acceptance_step,
                "actor_grad_norm": torch.norm(final_step).mean().item(),
                **reward_critic_norm,
                **cost_critic_norm,
            }
        )

        return metrics

    def train(self, it, xs, final_x, as_, next_rs, next_cs, next_terms):
        """Training step for CPO."""
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

                # Compute final values
                final_v_r = reward_critic(final_x)
                final_v_c = cost_critic(final_x)

                next_vs_r = torch.cat([vs_r[1:], final_v_r.unsqueeze(0)], 0)
                next_vs_c = torch.cat([vs_c[1:], final_v_c.unsqueeze(0)], 0)

                next_masks = utils.get_mask(1 - next_terms.to(next_rs.dtype))
                seq_masks = next_masks.cumulative_sequence(shifted=True)
                next_gammas = next_masks.values * self.gamma

                # Compute returns
                rets_r = utils.lambda_return(self.reward_act(next_rs), next_vs_r, next_gammas, self.lmbda)
                rets_c = utils.lambda_return(next_cs, next_vs_c, next_gammas, self.lmbda)

                ep_costs = rets_c[0:1, :]

                # Compute advantages
                if return_norm is not None:
                    return_norm.update(rets_r, rets_c)
                    adv_r = return_norm.normalize_rewards(rets_r) - return_norm.normalize_rewards(vs_r)
                    adv_c = return_norm.normalize_costs(rets_c) - return_norm.normalize_costs(vs_c)
                else:
                    adv_r = rets_r - vs_r
                    adv_c = rets_c - vs_c

                # Flatten sequences
                x, a, ret_r, ret_c, adv_r, adv_c, seq_mask = [utils.flatten_seq(val) for val in (xs, as_, rets_r, rets_c, adv_r, adv_c, seq_masks)]

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
        metrics = self._optimize(x, a, vs_r, vs_c, ret_r, ret_c, adv_r, adv_c, seq_mask, it, ep_costs, xs)

        metrics.update(batch_metrics)
        return metrics


# This class should looks good
class CPOReturnNorm(nn.Module):
    """Adopted and revised from class `ReturnNorm` in ac.py"""

    def __init__(self, low_percentile=5.0, high_percentile=95.0, decay=0.99, maximum=1.0, device=None):
        super().__init__()
        self.register_buffer("inv_max", torch.tensor(1 / maximum, device=device))
        self.register_buffer("q", torch.tensor([low_percentile / 100, high_percentile / 100], device=device))
        self.register_buffer("decay", torch.tensor(decay, device=device))

        # Reward normalization parameters
        self.register_buffer("low_r", torch.zeros(1, device=device))
        self.register_buffer("high_r", torch.zeros(1, device=device))
        self.register_buffer("inv_scale_r", torch.zeros(1, device=device))

        # Cost normalization parameters
        self.register_buffer("low_c", torch.zeros(1, device=device))
        self.register_buffer("high_c", torch.zeros(1, device=device))
        self.register_buffer("inv_scale_c", torch.zeros(1, device=device))

    def update(self, ret_r, ret_c):
        """Update normalization statistics for both rewards and costs."""
        ret_r = ret_r.type(torch.float32)
        ret_c = ret_c.type(torch.float32)

        # Update reward normalization
        ret_low_r, ret_high_r = torch.quantile(ret_r.flatten(), self.q)
        decay = self.decay
        self.low_r.data = decay * self.low_r + (1 - decay) * ret_low_r
        self.high_r.data = decay * self.high_r + (1 - decay) * ret_high_r
        self.inv_scale_r.data = torch.maximum(self.inv_max, self.high_r - self.low_r)

        # Update cost normalization
        ret_low_c, ret_high_c = torch.quantile(ret_c.flatten(), self.q)
        self.low_c.data = decay * self.low_c + (1 - decay) * ret_low_c
        self.high_c.data = decay * self.high_c + (1 - decay) * ret_high_c
        self.inv_scale_c.data = torch.maximum(self.inv_max, self.high_c - self.low_c)

    def normalize_rewards(self, ret_r):
        """Normalize reward returns."""
        return (ret_r - self.low_r) / self.inv_scale_r

    def normalize_costs(self, ret_c):
        """Normalize cost returns."""
        return (ret_c - self.low_c) / self.inv_scale_c

    def forward(self, ret_r, ret_c):
        return self.normalize_rewards(ret_r), self.normalize_costs(ret_c)
