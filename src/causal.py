import torch
import nets

from typing import Dict
from torch import nn
from torch.nn import functional as F


class PDAGLearning(nn.Module):
    def __init__(self, observation_space, single_action_space, y_dim, *, compile_, device=None):
        super().__init__()
        self.y_dim = y_dim
        self.a_dim = single_action_space.shape[0]
        latent_nodes = 2 * y_dim + self.a_dim
        hidden_dim = 64  # TODO: add to args
        self.curiosity_method = None
        self.device = device

        # IPS model -> reward and cost
        self.ips_model = compile_(nn.Sequential(nn.Linear(latent_nodes, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 2)).to(device))
        # Propensity model
        propensity_model = {}
        for i in range(self.y_dim):
            propensity_model[f"nexty_{i}"] = nn.Sequential(nn.Linear(latent_nodes, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1), nn.Sigmoid())
        self.propensity_model = compile_(nn.ModuleDict(propensity_model).to(self.device))

    def create_propensity_input(self, feat, indices):
        bulk_input = torch.zeros_like(feat)  # [y, action, next_y]
        if len(indices) > 0:
            next_y_start = self.y_dim + self.a_dim
            adjusted_indices = [next_y_start + idx for idx in indices]
            bulk_input[..., adjusted_indices] = feat[..., adjusted_indices]
        return bulk_input

    def forward_propensity(self, feat, bg_info):
        prop_list = []
        for i in range(self.y_dim // 2):
            parent_set, sib_set = bg_info[f"nexty_{i}"]["parents_idx"], bg_info[f"nexty_{i}"]["siblings_idx"]
            bulk_input_feat = self.create_propensity_input(feat, parent_set + sib_set)
            prop = self.propensity_model[f"nexty_{i}"](bulk_input_feat)
            prop_list.append(prop)

        propensity = torch.cat(prop_list, dim=-1)
        propensity = torch.clip(propensity, min=0.2, max=0.8)
        return propensity

    def get_treatment(self, yt, wm, pdag, agent, explore_agent) -> Dict[str, torch.Tensor]:
        wm.eval()
        agent.eval()
        explore_agent.eval()

        def get_effects(wm, agent, yt):
            # One-step
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

        with torch.no_grad():
            bg_info = pdag.background()
            prop = self.forward_propensity(feat, bg_info)
            prop_treatment = self.forward_propensity(treatment_feat, bg_info)
            propensity = torch.cat([prop, prop_treatment], dim=0).mean(-1).unsqueeze(-1)
            propensity = torch.clip(propensity, min=0.2, max=0.8)

        return dict(feat=feat, effects=effects, treatment_feat=treatment_feat, treatment_effects=treatment_effects, propensity=propensity)

    def curiosity(self, y, a, flat_a, wm, eps=1e-2):
        inp = torch.cat([y, flat_a], -1)
        next_y = y + wm.transition_predictor(inp)
        inp_ips = torch.cat([y, a, next_y], -1)

        pred = self.ips_model(inp_ips)
        reward_scaled = torch.where(torch.abs(pred[:, 0]) > eps, torch.ones_like(pred[:, 0]), pred[:, 0])
        cost_scaled = torch.where(torch.abs(pred[:, 1]) > eps, torch.ones_like(pred[:, 1]), pred[:, 1])
        return (reward_scaled, cost_scaled)


class PDAGTrainer:
    """Trainer for a PDAG model."""

    def __init__(self, cdm, propensity_coef, beta, C, constrain_coef, batch_size, num_epochs, cdm_optimizer, *, total_its, rng, autocast, compile_):
        self.cdm = cdm
        self.propensity_coef = propensity_coef
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

    def train(self, yt, wm, pdag, agent, explore_agent, it) -> Dict[str, float]:
        device = yt.device
        # propensity should come from interventional action
        outputs = self.cdm.get_treatment(yt, wm, pdag, agent, explore_agent)
        X = torch.cat([outputs["feat"], outputs["treatment_feat"]], dim=0)
        Y_r = torch.cat([outputs["effects"]["reward"], outputs["treatment_effects"]["reward"]], dim=0)
        Y_c = torch.cat([outputs["effects"]["cost"], outputs["treatment_effects"]["cost"]], dim=0)
        Y = torch.stack([Y_r, Y_c], dim=-1)
        propensity = outputs["propensity"].to(device)

        num_sample = X.shape[0]
        batch_size = self.batch_size
        bg_info = pdag.background()
        # T: half zeros, half ones
        T = torch.cat([torch.zeros(num_sample // 2, 1), torch.ones(num_sample - num_sample // 2, 1)], dim=0).to(device)

        metrics = {}

        with self.autocast():
            for _ in range(self.num_epochs):
                perm_idx = torch.randperm(num_sample)[:batch_size]
                sub_x = torch.Tensor(X[perm_idx]).to(device)
                sub_y = torch.Tensor(Y[perm_idx]).to(device)
                sub_t = torch.Tensor(T[perm_idx]).to(device)
                sub_propensity = torch.Tensor(propensity[perm_idx, :]).to(device)

                # propensity loss
                pred_propensity = self.cdm.forward_propensity(sub_x, bg_info).mean(-1).unsqueeze(-1)
                xent_loss = nn.BCELoss()(pred_propensity, sub_t)

                # pred = self.cdm.lr_model(sub_x).reshape(batch_size, -1)
                pred_cate = self.cdm.ips_model(sub_x).reshape(batch_size, -1)

                # TODO: C should be prediction of normalised reward/cost
                C = self.C
                CATE_bound = torch.max(torch.sum(torch.clip(-C - pred_cate, 0, 1) + torch.clip(-C + pred_cate, 0, 1), dim=0))
                CATE_y = self.ips_update_y(sub_t.reshape(batch_size, -1), sub_y, sub_propensity.reshape(batch_size, -1))

                loss = xent_loss + self.beta * torch.mean((pred_cate - CATE_y) ** 2) + self.constrain_coef * CATE_bound

                grad_norm_dict = self.cdm_optimizer.step(loss, batch_size, it)
                pred_cate_stat = pred_cate.detach().mean(0).cpu().numpy()
                batch_metrics = {"pred_cate_reward": pred_cate_stat[0], "pred_cate_cost": pred_cate_stat[1], "loss": loss.item(), "cate_bound": CATE_bound.item(), **grad_norm_dict}

                for k, v in batch_metrics.items():
                    if k not in metrics:
                        metrics[k] = []
                    metrics[k].append(float(v))

        for k in metrics:
            metrics[k] = torch.tensor(metrics[k]).mean().item()

        with torch.no_grad():
            feat_stats_original = torch.split(outputs["feat"].cpu(), [self.cdm.y_dim, self.cdm.a_dim, self.cdm.y_dim], dim=1)
            feat_stats_treatment = torch.split(outputs["treatment_feat"].cpu(), [self.cdm.y_dim, self.cdm.a_dim, self.cdm.y_dim], dim=1)

            metrics["diff_feat_a_mse"] = F.l1_loss(feat_stats_treatment[1], feat_stats_original[1]).item()
            metrics["diff_feat_nexty_mse"] = F.l1_loss(feat_stats_treatment[2], feat_stats_original[2]).item()
            metrics["diff_eff_r"] = (outputs["effects"]["reward"] - outputs["treatment_effects"]["reward"]).abs().mean().item()
            metrics["diff_eff_c"] = (outputs["effects"]["cost"] - outputs["treatment_effects"]["cost"]).abs().mean().item()

        return metrics
