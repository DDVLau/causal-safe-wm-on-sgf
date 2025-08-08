import time
import torch
import gymnasium as gym
import nets
import utils

from typing import Tuple, List, Dict
from torch import nn


class PDAGLearning(nn.Module):
    def __init__(self, observation_space, single_action_space, y_dim, *, compile_, device=None):
        super().__init__()
        a_dim = single_action_space.shape[0]
        latent_nodes = 2 * y_dim + a_dim

        # LR model -> reward and cost
        self.lr_model = compile_(nn.Sequential(torch.nn.Linear(latent_nodes, 256), torch.nn.ReLU(), torch.nn.Linear(256, 2)).to(device))
        # IPS model -> reward and cost
        self.ips_model = compile_(nn.Sequential(torch.nn.Linear(latent_nodes, 256), torch.nn.ReLU(), torch.nn.Linear(256, 2)).to(device))
        # Propensity prediction model
        self.propensity_model = compile_(nn.Sequential(torch.nn.Linear(latent_nodes, 256), torch.nn.ReLU(), torch.nn.Linear(256, 1)).to(device))

    def get_treatment(self, yt, wm, agent, explore_agent) -> Dict[str, torch.Tensor]:
        """ """
        wm.eval()
        agent.eval()
        explore_agent.eval()

        def get_effects(wm, agent, yt):
            effect_preds = dict(reward=None, cost=None)
            next_ = wm.imagine(agent, horizon=1, start_y=yt)
            y = next_[0][0]
            a = next_[1][0]
            next_y = next_[3][0]
            effect_preds["reward"] = next_[4][0]
            effect_preds["cost"] = next_[5][0]
            feat = torch.cat([y, a, next_y], dim=-1)
            return effect_preds, feat

        effects, feat = get_effects(wm, agent, yt)
        treatment_effects, treatment_feat = get_effects(wm, explore_agent, yt)
        propensity = 0.5 * torch.ones((2 * yt.shape[0], 1), device=yt.device)

        return dict(feat=feat, effects=effects, treatment_feat=treatment_feat, treatment_effects=treatment_effects, propensity=propensity)

    def curiosity(self, y, a, flat_a, wm):
        """"""
        reward_predictor = wm.reward_predictor
        cost_predictor = wm.cost_predictor

        inp = torch.cat([y, flat_a], -1)
        next_y = y + wm.transition_predictor(inp)

        # get effect (r, c)
        inp_ips = torch.cat([y, a, next_y], -1)
        pred = self.ips_model(inp_ips)
        _, inp_r, inp_c = wm._get_predictor_inputs(y, flat_a, next_y)

        true_r = reward_predictor(inp_r)
        true_c = cost_predictor(inp_c)
        return (true_r - pred[:, 0], true_c - pred[:, 1])


class PDAGTrainer:
    """Trainer for a PDAG model."""

    def __init__(self, cdm, alpha, beta, C, constrain_coef, batch_size, num_epochs, cdm_optimizer, *, total_its, rng, autocast, compile_):
        self.cdm = cdm
        self.alpha = alpha
        self.beta = beta
        self.C = C
        self.constrain_coef = constrain_coef
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.total_its = total_its
        self.rng = rng
        self.autocast = autocast
        self.compile_ = compile_

        self.cdm_optimizer = nets.Optimizer(cdm, **cdm_optimizer, total_its=total_its, autocast=autocast)

    def ips_update_y(self, T, Y, weight):
        return T * Y / weight - (1 - T) * Y / (1 - weight)

    def train(self, yt, wm, agent, explore_agent, it) -> Dict[str, float]:
        device = yt.device
        # propensity should come from interventional action
        outputs = self.cdm.get_treatment(yt, wm, agent, explore_agent)
        X = torch.cat([outputs["feat"], outputs["treatment_feat"]], dim=0)
        Y_r = torch.cat([outputs["effects"]["reward"], outputs["treatment_effects"]["reward"]], dim=0)
        Y_c = torch.cat([outputs["effects"]["cost"], outputs["treatment_effects"]["cost"]], dim=0)
        Y = torch.stack([Y_r, Y_c], dim=-1)
        propensity = outputs["propensity"].to(device)

        num_sample = X.shape[0]
        batch_size = self.batch_size
        # T: half zeros, half ones
        T = torch.cat([torch.zeros(num_sample // 2, 1), torch.ones(num_sample - num_sample // 2, 1)], dim=0).to(device)
        metrics = {}

        with self.autocast():
            for _ in range(self.num_epochs):
                perm_idx = torch.randperm(num_sample)[:batch_size]
                sub_x = torch.Tensor(X[perm_idx])
                sub_y = torch.Tensor(Y[perm_idx])
                sub_t = torch.Tensor(T[perm_idx])
                sub_propensity = torch.Tensor(propensity[perm_idx, :])

                pred = self.cdm.lr_model(sub_x).reshape(batch_size, -1)
                pred_cate = self.cdm.ips_model(sub_x).reshape(batch_size, -1)
                xent_loss = nn.MSELoss()(pred, sub_y)

                # TODO: C should be prediction of normalised reward/cost
                C = self.C
                CATE_bound = torch.max(torch.sum(torch.clip(-C - pred_cate, 0, 10) + torch.clip(-C + pred_cate, 0, 10), dim=0))
                # TODO: this should be changed according to new sub_t and sub_propensity
                CATE_y = self.ips_update_y(sub_t.reshape(batch_size, -1), pred, sub_propensity.reshape(batch_size, -1))

                loss = self.beta * torch.mean((pred_cate - CATE_y) ** 2) + self.alpha * xent_loss + self.constrain_coef * CATE_bound

                grad_norm_dict = self.cdm_optimizer.step(loss, batch_size, it)

                pred_cate_stat = pred_cate.detach().mean(0).cpu().numpy()
                batch_metrics = {"pred_cate_reward": pred_cate_stat[0], "pred_cate_cost": pred_cate_stat[1], "loss": loss.item(), **grad_norm_dict}
                for k, v in batch_metrics.items():
                    if k not in metrics:
                        metrics[k] = []
                    metrics[k].append(float(v))

        for k in metrics:
            metrics[k] = torch.tensor(metrics[k]).mean().item()

        return metrics

    def _train_propensity(self, data, label) -> dict:
        metrics = {}
        # TBD
        with self.autocast():
            pass
        return metrics
