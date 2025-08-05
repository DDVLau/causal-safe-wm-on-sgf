import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
import wandb
from torch import nn
from torchvision.utils import make_grid

import nets
import utils
from mist import CLUB


class WorldModelDecomposed(nn.Module):
    """Decomposed world model of safety RL environment."""

    def __init__(
        self,
        observation_space,
        stacked_action_space,
        y_dim,
        z_dim,
        mist_hidden_dim,
        encoder,
        projector,
        predictor,
        transition_predictor,
        reward_predictor,
        cost_predictor,
        terminal_predictor,
        contrastive,
        sim_coef,
        var_coef,
        cov_coef,
        reward_coef,
        cost_coef,
        terminal_coef,
        mi_coef,
        *,
        compile_,
        device=None,
    ):
        super().__init__()

        if not (isinstance(observation_space, gym.spaces.Box) and len(observation_space.shape) == 4):
            raise ValueError('Observation space is not supported')

        # if not isinstance(stacked_action_space, gym.spaces.MultiDiscrete):
        #     raise ValueError('Action space is not supported')

        self.observation_space = observation_space
        self.stacked_action_space = stacked_action_space
        flat_stacked_action_space = gym.spaces.flatten_space(stacked_action_space)

        num_frames, h, w, c = observation_space.shape
        o_dim = num_frames * c
        o_res = h
        a_dim = flat_stacked_action_space.shape[0]

        self.o_dim = o_dim
        self.y_dim = y_dim  # this should be a dict
        self.z_dim = z_dim

        total_y_dim = sum(y_dim.values()) * 2

        self.encoder = compile_(nn.Sequential(*nets.cnn(o_dim, o_res, total_y_dim, **encoder, memory_format=torch.channels_last, device=device))) # y_reward + y_cost
        self.projector = compile_(nets.VectorMLP(total_y_dim, z_dim, **projector, device=device)) # y_reward + y_cost
        self.predictor = compile_(nets.VectorMLP(z_dim + a_dim, z_dim, **predictor, device=device))
        self.transition_predictor = compile_(nets.VectorMLP(total_y_dim + a_dim, total_y_dim, **transition_predictor, device=device))
        self.reward_predictor = compile_(nets.ScalarMLP(2 * y_dim['reward'] + a_dim, **reward_predictor, device=device)) # y_cost only
        self.terminal_predictor = compile_(nets.ScalarMLP(2 * total_y_dim + a_dim, **terminal_predictor, device=device)) # y_reward + y_cost
        if cost_predictor is not None:
            self.cost_predictor = compile_(nets.ScalarMLP(2 * y_dim['cost'] + a_dim, **cost_predictor, device=device)) # y_reward only

        self.mist_estimator = compile_(self._init_mist_estimator(y_dim, a_dim, mist_hidden_dim, device=device))

        self.contrastive = contrastive
        self.sim_coef = sim_coef
        self.var_coef = var_coef
        self.cov_coef = cov_coef
        self.reward_coef = reward_coef
        self.cost_coef = cost_coef
        self.terminal_coef = terminal_coef
        self.mi_coef = mi_coef
        self.device = device

    def _init_mist_estimator(self, y_dim, a_dim, mist_hidden_dim, device=None):
        """Initialize the MIST estimator."""
        estimators = nn.ModuleDict()
        estimators['reward'] = CLUB(x_dim=2 * y_dim['reward'] + a_dim, y_dim=1, hidden_size=mist_hidden_dim).to(device)
        estimators['reward_bar'] = CLUB(x_dim=2 * y_dim['reward'] + a_dim, y_dim=1, hidden_size=mist_hidden_dim).to(device)
        estimators['cost'] = CLUB(x_dim=2 * y_dim['cost'] + a_dim, y_dim=1, hidden_size=mist_hidden_dim).to(device)
        estimators['cost_bar'] = CLUB(x_dim=2 * y_dim['cost'] + a_dim, y_dim=1, hidden_size=mist_hidden_dim).to(device)
        return estimators

    def representation_modules(self):
        """Modules that are part of representation learning."""
        return (self.encoder, self.projector, self.predictor, self.reward_predictor, self.cost_predictor, self.terminal_predictor)

    def transition_modules(self):
        """Modules that are part of transition learning."""
        return (self.transition_predictor,)

    @torch.no_grad()
    def preprocess(self, o, dtype=None):
        """Preprocess a tensor of observations."""
        if dtype is None:
            dtype = torch.float

        # (N, F, H, W, C) -> (N, F * C, H, W) where F = num frames
        o = o.detach().permute(0, 1, 4, 2, 3).flatten(1, 2)

        if o.dtype != dtype:
            # Convert to float and normalize
            was_uint8 = o.dtype == torch.uint8
            o = o.to(dtype=dtype, memory_format=torch.channels_last)
            if was_uint8:
                o = o / 255.0
        else:
            o = o.to(memory_format=torch.channels_last)

        return o

    @torch.no_grad()
    def flatten_actions(self, stacked_a, dtype=None):
        """Flatten a stacked action tensor, to be used as input to a network."""
        if dtype is None:
            dtype = torch.float
        flat_a = utils.space_flatten(self.stacked_action_space, stacked_a, dtype=dtype)
        return flat_a

    @torch.no_grad()
    def encode(self, o, dtype=None):
        """Preprocess and encode an observation tensor."""
        o = self.preprocess(o, dtype=dtype)
        o = o.to()
        y = self.encoder(o)
        return y

    @torch.no_grad()
    def imagine(self, agent, horizon, start_y, start_cont_mask=None):
        """Synthesize a batch of trajectories."""
        assert not self.training

        y = start_y
        if start_cont_mask is not None:
            y = start_cont_mask.apply(y)
        agent_state = agent.start()
        cont_mask = start_cont_mask
        history = []

        for t in range(horizon):
            a, stacked_a, next_agent_state = agent.act(agent_state, cont_mask, y)

            flat_a = self.flatten_actions(stacked_a, dtype=y.dtype)  # dtype depends on autocast
            inp = torch.cat([y, flat_a], -1)

            # skip connection
            next_y = y + self.transition_predictor(inp)

            y_r, y_r_bar, y_c, y_c_bar = torch.split(y, [self.y_dim['reward'], self.y_dim['reward'], self.y_dim['cost'], self.y_dim['cost']], dim=-1)
            next_y_r, next_y_r_bar, next_y_c, next_y_c_bar = torch.split(next_y, [self.y_dim['reward'], self.y_dim['reward'], self.y_dim['cost'], self.y_dim['cost']], dim=-1)

            inp = torch.cat([y, flat_a, next_y], -1)
            inp_r = torch.cat([y_r, flat_a, next_y_r], -1)
            inp_c = torch.cat([y_c, flat_a, next_y_c], -1)

            inp = torch.cat([y, flat_a, next_y], -1)
            next_r = self.reward_predictor(inp_r)
            next_c = self.cost_predictor(inp_c) if hasattr(self, 'cost_predictor') else None
            next_term = self.terminal_predictor(inp)

            history.append((y, a, stacked_a, next_y, next_r, next_c, next_term))

            # use next_term to reset the next_y
            next_cont_mask = utils.get_mask(1 - next_term.float())
            y = next_cont_mask.apply(next_y)
            agent_state = next_agent_state
            cont_mask = next_cont_mask

        # ys, as, stacked_as, next_ys, next_rs, next_terms
        return tuple(map(tuple, zip(*history)))

    def representation_loss(self, ot, next_ot, flat_a, next_r, next_c, next_term):
        """Compute the representation loss."""
        assert self.training

        yt = self.encoder(ot)
        z = self.projector(yt)

        next_yt = self.encoder(next_ot)
        next_z = self.projector(next_yt)

        # variance-covariance regularization of z and next_z
        var_loss1, cov_loss1, std1 = utils.variance_covariance_loss(z, contrastive=self.contrastive)
        var_loss2, cov_loss2, std2 = utils.variance_covariance_loss(next_z, contrastive=self.contrastive)
        var_loss = (var_loss1 + var_loss2) / 2
        cov_loss = (cov_loss1 + cov_loss2) / 2

        # similarity between predicted z and next_z
        inp = torch.cat([z, flat_a], -1)
        pred_z = self.predictor(inp)
        sim_loss = self.predictor.loss(pred_z, next_z)

        # truncate the y_dim to match the input of the reward and terminal predictors
        yt_r, yt_r_bar, yt_c, yt_c_bar = torch.split(yt, [self.y_dim['reward'], self.y_dim['reward'], self.y_dim['cost'], self.y_dim['cost']], dim=-1)
        next_yt_r, next_yt_r_bar, next_yt_c, next_yt_c_bar = torch.split(next_yt, [self.y_dim['reward'], self.y_dim['reward'], self.y_dim['cost'], self.y_dim['cost']], dim=-1)

        inp = torch.cat([yt, flat_a, next_yt], -1)
        inp_reward = torch.cat([yt_r, flat_a, next_yt_r], -1)
        inp_cost = torch.cat([yt_c, flat_a, next_yt_c], -1)
        reward_stats = self.reward_predictor.get_stats(inp_reward, full_precision=True)
        reward_loss = self.reward_predictor.loss(reward_stats, next_r)
        cost_stats = self.cost_predictor.get_stats(inp_cost, full_precision=True)
        cost_loss = self.cost_predictor.loss(cost_stats, next_c)
        terminal_stats = self.terminal_predictor.get_stats(inp, full_precision=True)
        terminal_loss = self.terminal_predictor.loss(terminal_stats, next_term)

        inputs = self._get_mist_inputs(yt, flat_a, next_yt, next_r, next_c)
        mist_outer_loss = self.mi_outer_loss(inputs, self.mi_coef)

        representation_loss = (
            self.sim_coef * sim_loss
            + self.var_coef * var_loss
            + self.cov_coef * cov_loss
            + self.reward_coef * reward_loss
            + self.cost_coef * cost_loss
            + self.terminal_coef * terminal_loss
            + mist_outer_loss
        )

        pred_r = self.reward_predictor.predict(reward_stats)
        pred_term = self.terminal_predictor.predict(terminal_stats)
        if pred_term.dtype != torch.bool:
            pred_term = pred_term > 0.5

        metrics = {
            'z_sim_loss': sim_loss,
            'z_var_loss': var_loss,
            'z_cov_loss': cov_loss,
            'reward_loss': reward_loss,
            'cost_loss': cost_loss,
            'terminal_loss': terminal_loss,
            'representation_loss': representation_loss,
            'mist_outer_loss': mist_outer_loss,
            'z_std': ((std1 + std2) / 2).mean(),
            'reward_mae': (pred_r - next_r).abs().mean(),
            'terminal_acc': (pred_term == next_term).float().mean(),
        }
        return representation_loss, metrics, yt, next_yt

    def transition_loss(self, y, flat_a, next_y):
        """Compute the transition loss."""
        inp = torch.cat([y, flat_a], -1)
        # skip connection
        pred_y = y + self.transition_predictor(inp)
        transition_loss = self.transition_predictor.loss(pred_y, next_y)

        metrics = {
            'transition_loss': transition_loss,
            'y_mse': F.mse_loss(y, next_y),
            'y_norm': (torch.linalg.vector_norm(y, dim=-1).mean() + torch.linalg.vector_norm(next_y, dim=-1).mean()) / 2,
            'transition_mae': F.l1_loss(pred_y, next_y),
        }
        return transition_loss, metrics

    def mist_loss(self, y, flat_a, next_y, next_r, next_c):
        inputs = self._get_mist_inputs(y, flat_a, next_y, next_r, next_c)
        inner_loss = self.mi_inner_loss(inputs)

        metrics = {
            'mist_inner_loss': inner_loss,
        }
        return inner_loss, metrics

    def _get_mist_inputs(self, y, flat_a, next_y, next_r, next_c):
        y_splitted = torch.split(y, [self.y_dim['reward'], self.y_dim['reward'], self.y_dim['cost'], self.y_dim['cost']], dim=-1)
        next_y_splitted = torch.split(next_y, [self.y_dim['reward'], self.y_dim['reward'], self.y_dim['cost'], self.y_dim['cost']], dim=-1)
        # 0: reward, 1: reward_bar 2: cost, 3: cost_bar
        mine_reward_1 = (torch.cat([next_y_splitted[0], flat_a, y_splitted[0].detach()], dim=-1), next_r)
        mine_reward_2 = (torch.cat([next_y_splitted[1], flat_a, y_splitted[0].detach()], dim=-1), next_r)
        mine_cost_1 = (torch.cat([next_y_splitted[2], flat_a, y_splitted[2].detach()], dim=-1), next_c)
        mine_cost_2 = (torch.cat([next_y_splitted[3], flat_a, y_splitted[2].detach()], dim=-1), next_c)
        return (mine_reward_1, mine_reward_2, mine_cost_1, mine_cost_2)

    def mi_inner_loss(self, inputs):
        assert len(inputs) == 4
        losses = [
            self.mist_estimator[key].learning_loss(inp[0].detach(), inp[1].detach())
            for key, inp in zip(
                ["reward", "reward_bar", "cost", "cost_bar"],
                inputs
            )
        ]
        return sum(losses)

    def mi_outer_loss(self, inputs, mi_coef):
        assert len(inputs) == 4
        loss = -(
            mi_coef['reward'] * self.mist_estimator['reward'](inputs[0][0], inputs[0][1])
            - mi_coef['reward_bar'] * self.mist_estimator['reward_bar'](inputs[1][0], inputs[1][1])
        ) - (
            mi_coef['cost'] * self.mist_estimator['cost'](inputs[2][0], inputs[2][1])
            - mi_coef['cost_bar'] * self.mist_estimator['cost_bar'](inputs[3][0], inputs[3][1])
        )
        return loss


class WorldModel(nn.Module):
    """World model of an environment."""

    def __init__(
        self,
        observation_space,
        stacked_action_space,
        y_dim,
        z_dim,
        encoder,
        projector,
        predictor,
        transition_predictor,
        reward_predictor,
        terminal_predictor,
        contrastive,
        sim_coef,
        var_coef,
        cov_coef,
        reward_coef,
        terminal_coef,
        *,
        compile_,
        device=None,
    ):
        super().__init__()

        if not (isinstance(observation_space, gym.spaces.Box) and len(observation_space.shape) == 4):
            raise ValueError('Observation space is not supported')

        # if not isinstance(stacked_action_space, gym.spaces.MultiDiscrete):
        #     raise ValueError('Action space is not supported')

        self.observation_space = observation_space
        self.stacked_action_space = stacked_action_space
        flat_stacked_action_space = gym.spaces.flatten_space(stacked_action_space)

        num_frames, h, w, c = observation_space.shape
        o_dim = num_frames * c
        o_res = h
        a_dim = flat_stacked_action_space.shape[0]

        self.o_dim = o_dim
        self.y_dim = y_dim
        self.z_dim = z_dim

        self.encoder = compile_(nn.Sequential(*nets.cnn(o_dim, o_res, y_dim, **encoder, memory_format=torch.channels_last, device=device)))
        self.projector = compile_(nets.VectorMLP(y_dim, z_dim, **projector, device=device))
        self.predictor = compile_(nets.VectorMLP(z_dim + a_dim, z_dim, **predictor, device=device))
        self.transition_predictor = compile_(nets.VectorMLP(y_dim + a_dim, y_dim, **transition_predictor, device=device))
        self.reward_predictor = compile_(nets.ScalarMLP(2 * y_dim + a_dim, **reward_predictor, device=device))
        self.terminal_predictor = compile_(nets.ScalarMLP(2 * y_dim + a_dim, **terminal_predictor, device=device))

        self.contrastive = contrastive
        self.sim_coef = sim_coef
        self.var_coef = var_coef
        self.cov_coef = cov_coef
        self.reward_coef = reward_coef
        self.terminal_coef = terminal_coef
        self.device = device

    def representation_modules(self):
        """Modules that are part of representation learning."""
        return (self.encoder, self.projector, self.predictor, self.reward_predictor, self.terminal_predictor)

    def transition_modules(self):
        """Modules that are part of transition learning."""
        return (self.transition_predictor,)

    @torch.no_grad()
    def preprocess(self, o, dtype=None):
        """Preprocess a tensor of observations."""
        if dtype is None:
            dtype = torch.float

        # (N, F, H, W, C) -> (N, F * C, H, W) where F = num frames
        o = o.detach().permute(0, 1, 4, 2, 3).flatten(1, 2)

        if o.dtype != dtype:
            # Convert to float and normalize
            was_uint8 = o.dtype == torch.uint8
            o = o.to(dtype=dtype, memory_format=torch.channels_last)
            if was_uint8:
                o = o / 255.0
        else:
            o = o.to(memory_format=torch.channels_last)

        return o

    @torch.no_grad()
    def flatten_actions(self, stacked_a, dtype=None):
        """Flatten a stacked action tensor, to be used as input to a network."""
        if dtype is None:
            dtype = torch.float
        flat_a = utils.space_flatten(self.stacked_action_space, stacked_a, dtype=dtype)
        return flat_a

    @torch.no_grad()
    def encode(self, o, dtype=None):
        """Preprocess and encode an observation tensor."""
        o = self.preprocess(o, dtype=dtype)
        o = o.to()
        y = self.encoder(o)
        return y

    @torch.no_grad()
    def imagine(self, agent, horizon, start_y, start_cont_mask=None):
        """Synthesize a batch of trajectories."""
        assert not self.training

        y = start_y
        if start_cont_mask is not None:
            y = start_cont_mask.apply(y)
        agent_state = agent.start()
        cont_mask = start_cont_mask
        history = []

        for t in range(horizon):
            a, stacked_a, next_agent_state = agent.act(agent_state, cont_mask, y)

            flat_a = self.flatten_actions(stacked_a, dtype=y.dtype)  # dtype depends on autocast
            inp = torch.cat([y, flat_a], -1)

            # skip connection
            next_y = y + self.transition_predictor(inp)

            inp = torch.cat([y, flat_a, next_y], -1)
            next_r = self.reward_predictor(inp)
            next_term = self.terminal_predictor(inp)

            history.append((y, a, stacked_a, next_y, next_r, next_term))

            # use next_term to reset the next_y
            next_cont_mask = utils.get_mask(1 - next_term.float())
            y = next_cont_mask.apply(next_y)
            agent_state = next_agent_state
            cont_mask = next_cont_mask

        # ys, as, stacked_as, next_ys, next_rs, next_terms
        return tuple(map(tuple, zip(*history)))

    def representation_loss(self, ot, next_ot, flat_a, next_r, next_term):
        """Compute the representation loss."""
        assert self.training

        yt = self.encoder(ot)
        z = self.projector(yt)

        next_yt = self.encoder(next_ot)
        next_z = self.projector(next_yt)

        # variance-covariance regularization of z and next_z
        var_loss1, cov_loss1, std1 = utils.variance_covariance_loss(z, contrastive=self.contrastive)
        var_loss2, cov_loss2, std2 = utils.variance_covariance_loss(next_z, contrastive=self.contrastive)
        var_loss = (var_loss1 + var_loss2) / 2
        cov_loss = (cov_loss1 + cov_loss2) / 2

        # similarity between predicted z and next_z
        inp = torch.cat([z, flat_a], -1)
        pred_z = self.predictor(inp)
        sim_loss = self.predictor.loss(pred_z, next_z)

        inp = torch.cat([yt, flat_a, next_yt], -1)
        reward_stats = self.reward_predictor.get_stats(inp, full_precision=True)
        reward_loss = self.reward_predictor.loss(reward_stats, next_r)
        terminal_stats = self.terminal_predictor.get_stats(inp, full_precision=True)
        terminal_loss = self.terminal_predictor.loss(terminal_stats, next_term)

        representation_loss = (
            self.sim_coef * sim_loss
            + self.var_coef * var_loss
            + self.cov_coef * cov_loss
            + self.reward_coef * reward_loss
            + self.terminal_coef * terminal_loss
        )

        pred_r = self.reward_predictor.predict(reward_stats)
        pred_term = self.terminal_predictor.predict(terminal_stats)
        if pred_term.dtype != torch.bool:
            pred_term = pred_term > 0.5

        metrics = {
            'z_sim_loss': sim_loss,
            'z_var_loss': var_loss,
            'z_cov_loss': cov_loss,
            'reward_loss': reward_loss,
            'terminal_loss': terminal_loss,
            'representation_loss': representation_loss,
            'z_std': ((std1 + std2) / 2).mean(),
            'reward_mae': (pred_r - next_r).abs().mean(),
            'terminal_acc': (pred_term == next_term).float().mean(),
        }
        return representation_loss, metrics, yt, next_yt

    def transition_loss(self, y, flat_a, next_y):
        """Compute the transition loss."""
        inp = torch.cat([y, flat_a], -1)
        # skip connection
        pred_y = y + self.transition_predictor(inp)
        transition_loss = self.transition_predictor.loss(pred_y, next_y)

        metrics = {
            'transition_loss': transition_loss,
            'y_mse': F.mse_loss(y, next_y),
            'y_norm': (torch.linalg.vector_norm(y, dim=-1).mean() + torch.linalg.vector_norm(next_y, dim=-1).mean()) / 2,
            'transition_mae': F.l1_loss(pred_y, next_y),
        }
        return transition_loss, metrics


class WorldModelDecomposedTrainer:
    """Trainer for a decomposed world model."""

    def __init__(
        self,
        wm,
        replay_buffer,
        batch_size,
        augmentation,
        representation_optimizer,
        transition_optimizer,
        mist_optimizer,
        debug,
        init_steps,
        eval_mode,
        total_its,
        rng,
        *,
        autocast,
        compile_,
    ):

        if eval_mode not in ('none', 'decoder'):
            raise ValueError('World model eval_mode must be one of "none", "decoder"')

        self.wm = wm
        self.replay_buffer = replay_buffer
        self.batch_size = batch_size

        # Create augmentation module
        self.augmentation = compile_(nets.augmentation(augmentation))

        # Create optimizers
        self.representation_optimizer = nets.Optimizer(
            nn.ModuleList(wm.representation_modules()), **representation_optimizer, total_its=total_its, autocast=autocast
        )
        self.transition_optimizer = nets.Optimizer(
            nn.ModuleList(wm.transition_modules()), **transition_optimizer, total_its=total_its, autocast=autocast
        )
        self.mist_optimizer = nets.Optimizer(
            wm.mist_estimator, **mist_optimizer, total_its=total_its, autocast=autocast
        )

        self.init_steps = init_steps
        self.eval_mode = eval_mode
        self.total_its = total_its
        self.dream_horizon = debug['dream_horizon']
        self.rng = rng
        self.autocast = autocast

        total_y_dim = sum(wm.y_dim.values()) * 2

        if eval_mode == 'decoder':
            # Create decoder and optimizer for debugging/evaluation
            device = next(wm.parameters()).device
            self.decoder = compile_(
                nn.Sequential(*nets.transpose_cnn(total_y_dim, wm.o_dim, **debug['decoder'], memory_format=torch.channels_last, device=device))
            )
            self.decoder_optimizer = nets.Optimizer(self.decoder, **debug['optimizer'], total_its=total_its, autocast=autocast)
        elif eval_mode != 'none':
            raise ValueError('Invalid eval_mode')

        # Compile the optimization functions
        self._optimize = compile_(self._optimize)
        self._optimize_decoder = compile_(self._optimize_decoder)

    def _optimize(self, o, stacked_a, next_r, next_c, next_term, next_o, it):
        # This method is compiled with torch.compile
        wm = self.wm

        # Compute the representation loss
        autocast = self.autocast()
        with autocast:
            with torch.no_grad():
                dtype = torch.half if autocast._enabled else torch.float

                o = wm.preprocess(o, dtype=dtype)
                next_o = wm.preprocess(next_o, dtype=dtype)

                ot = self.augmentation(o)
                next_ot = self.augmentation(next_o)

                flat_a = wm.flatten_actions(stacked_a, dtype=dtype)

            representation_loss, representation_metrics, yt, next_yt = wm.representation_loss(ot, next_ot, flat_a, next_r, next_c, next_term)

        # Compute the transition loss
        with self.autocast():
            if isinstance(self.augmentation, nn.Identity):
                # don't need to reencode if not augmented
                y = yt.detach()
                next_y = next_yt.detach()
            else:
                # encode without augmentation
                wm.encoder.eval()
                with torch.no_grad():
                    y = wm.encoder(o)
                    next_y = wm.encoder(next_o)

            transition_loss, transition_metrics = wm.transition_loss(y, flat_a, next_y)

        # Compute the MIST loss
        with self.autocast():
            mist_total_loss, mist_metrics = wm.mist_loss(y, flat_a, next_y, next_r, next_c)

        # Optimize the world model
        rep_norm = self.representation_optimizer.step(representation_loss, self.batch_size, it)
        tran_norm = self.transition_optimizer.step(transition_loss, self.batch_size, it)
        mist_norm = self.mist_optimizer.step(mist_total_loss, self.batch_size, it)
        metrics = {**representation_metrics, **transition_metrics, **mist_metrics, **rep_norm, **tran_norm, **mist_norm}
        return metrics, y

    def _optimize_decoder(self, o, y, it):
        # This method is compiled with torch.compile
        autocast = self.autocast()
        with autocast:
            dtype = torch.half if autocast._enabled else torch.float
            o = self.wm.preprocess(o, dtype=dtype)
            recon = self.decoder(y)
            loss = F.mse_loss(recon, o, reduction='none').sum([-3, -2, -1]).mean()
            metrics = {'decoder_loss': loss}

        decoder_norm = self.decoder_optimizer.step(loss, o.shape[0], it)
        metrics.update(decoder_norm)
        return metrics

    def train(self, it):
        """Train the world model for one iteration."""
        idx = self.replay_buffer.sample_idx(self.batch_size, self.rng)
        target_device = self.wm.device
        o, stacked_a, next_r, next_c, next_term, next_o = self.replay_buffer.get(idx, 'o', 'a', 'next_r', 'next_c', 'next_term', 'next_o')
        o, stacked_a, next_r, next_c, next_term, next_o = (
            o.to(target_device),
            stacked_a.to(target_device),
            next_r.to(target_device),
            next_c.to(target_device),
            next_term.to(target_device),
            next_o.to(target_device),
        )

        self.wm.train()
        metrics, y = self._optimize(o, stacked_a, next_r, next_c, next_term, next_o, it)

        if self.eval_mode == 'decoder':
            self.decoder.train()
            decoder_metrics = self._optimize_decoder(o, y, it)
            metrics.update(decoder_metrics)

        return metrics, y

    @torch.no_grad()
    def evaluate(self, agent, seed):
        """Evaluate the world model."""
        if self.eval_mode != 'decoder':
            return {}

        wm, decoder, replay_buffer = self.wm, self.decoder, self.replay_buffer
        metrics = {}

        eval_rng = np.random.Generator(np.random.PCG64(seed))

        # Visualize reconstructions and dreams
        wm.eval()
        decoder.eval()
        with self.autocast():
            # Choose a few fixed and random observations from the buffer for visualization
            num_obs = 3
            fixed_idx = torch.linspace(0, self.init_steps, num_obs).long()
            fixed_o, fixed_term, fixed_trunc = replay_buffer.get(fixed_idx, 'next_o', 'next_term', 'next_trunc')
            random_idx = eval_rng.choice(len(replay_buffer), num_obs, replace=False)
            random_o, random_term, random_trunc = replay_buffer.get(random_idx, 'next_o', 'next_term', 'next_trunc')
            fixed_o, fixed_term, fixed_trunc = fixed_o.to(wm.device), fixed_term.to(wm.device), fixed_trunc.to(wm.device)
            random_o, random_term, random_trunc = random_o.to(wm.device), random_term.to(wm.device), random_trunc.to(wm.device)

            o = torch.cat([fixed_o, random_o], 0)
            term = torch.cat([fixed_term, random_term], 0)
            trunc = torch.cat([fixed_trunc, random_trunc], 0)
            cont_mask = utils.get_mask(~(term | trunc))

            # Encode and decode the observations
            y = wm.encode(o)
            ohat = decoder(y)
            # (N, F * C, H, W) -> (N, F, C, H, W)
            num_frames = wm.observation_space.shape[0]
            o = wm.preprocess(o, dtype=ohat.dtype)
            o, ohat = [x.unflatten(1, (num_frames, -1)) for x in (o, ohat)]

            # Create a visualization of the reconstructions
            img = torch.cat([utils.visualize_observations(o), utils.visualize_observations(ohat.clamp(0.0, 1.0))], 0)
            img = make_grid(img, nrow=num_obs * 2)
            img = (img.permute(1, 2, 0) * 255.0).byte().cpu().numpy()
            metrics['recons'] = wandb.Image(img)

            # Synthesize trajectories (dream)
            ys = wm.imagine(agent, self.dream_horizon, y, cont_mask)[0]
            ys = torch.stack(ys, 0)
            ohat = utils.apply_seq(decoder, ys)
            # (T, N, F * C, H, W) -> (T, N, F, C, H, W)
            ohat = ohat.unflatten(2, (num_frames, -1))

            # Create a video of the synthesized observations
            video = utils.apply_seq(utils.visualize_observations, ohat.clamp(0.0, 1.0))
            video = [make_grid(video[t], nrow=num_obs) for t in range(video.shape[0])]
            video.append(torch.zeros_like(video[-1]))  # append empty frame to visualize ends
            video = torch.stack(video, 0)
            video = (video * 255.0).byte().cpu().numpy()
            metrics['dream'] = wandb.Video(video, fps=10, format='gif')

        return metrics


class WorldModelTrainer:
    """Trainer for a world model."""

    def __init__(
        self,
        wm,
        replay_buffer,
        batch_size,
        augmentation,
        representation_optimizer,
        transition_optimizer,
        debug,
        init_steps,
        eval_mode,
        total_its,
        rng,
        *,
        autocast,
        compile_,
    ):

        if eval_mode not in ('none', 'decoder'):
            raise ValueError('World model eval_mode must be one of "none", "decoder"')

        self.wm = wm
        self.replay_buffer = replay_buffer
        self.batch_size = batch_size

        # Create augmentation module
        self.augmentation = compile_(nets.augmentation(augmentation))

        # Create optimizers
        self.representation_optimizer = nets.Optimizer(
            nn.ModuleList(wm.representation_modules()), **representation_optimizer, total_its=total_its, autocast=autocast
        )
        self.transition_optimizer = nets.Optimizer(
            nn.ModuleList(wm.transition_modules()), **transition_optimizer, total_its=total_its, autocast=autocast
        )

        self.init_steps = init_steps
        self.eval_mode = eval_mode
        self.total_its = total_its
        self.dream_horizon = debug['dream_horizon']
        self.rng = rng
        self.autocast = autocast

        if eval_mode == 'decoder':
            # Create decoder and optimizer for debugging/evaluation
            device = next(wm.parameters()).device
            self.decoder = compile_(
                nn.Sequential(*nets.transpose_cnn(wm.y_dim, wm.o_dim, **debug['decoder'], memory_format=torch.channels_last, device=device))
            )
            self.decoder_optimizer = nets.Optimizer(self.decoder, **debug['optimizer'], total_its=total_its, autocast=autocast)
        elif eval_mode != 'none':
            raise ValueError('Invalid eval_mode')

        # Compile the optimization functions
        self._optimize = compile_(self._optimize)
        self._optimize_decoder = compile_(self._optimize_decoder)

    def _optimize(self, o, stacked_a, next_r, next_c, next_term, next_o, it):
        # This method is compiled with torch.compile
        wm = self.wm

        # Compute the representation loss
        autocast = self.autocast()
        with autocast:
            with torch.no_grad():
                dtype = torch.half if autocast._enabled else torch.float

                o = wm.preprocess(o, dtype=dtype)
                next_o = wm.preprocess(next_o, dtype=dtype)

                ot = self.augmentation(o)
                next_ot = self.augmentation(next_o)

                flat_a = wm.flatten_actions(stacked_a, dtype=dtype)

            representation_loss, representation_metrics, yt, next_yt = wm.representation_loss(ot, next_ot, flat_a, next_r, next_c,next_term)

        # Compute the transition loss
        with self.autocast():
            if isinstance(self.augmentation, nn.Identity):
                # don't need to reencode if not augmented
                y = yt.detach()
                next_y = next_yt.detach()
            else:
                # encode without augmentation
                wm.encoder.eval()
                with torch.no_grad():
                    y = wm.encoder(o)
                    next_y = wm.encoder(next_o)

            transition_loss, transition_metrics = wm.transition_loss(y, flat_a, next_y)

        # Optimize the world model
        self.representation_optimizer.step(representation_loss, self.batch_size, it)
        self.transition_optimizer.step(transition_loss, self.batch_size, it)
        metrics = {**representation_metrics, **transition_metrics}
        return metrics, y

    def _optimize_decoder(self, o, y, it):
        # This method is compiled with torch.compile
        autocast = self.autocast()
        with autocast:
            dtype = torch.half if autocast._enabled else torch.float
            o = self.wm.preprocess(o, dtype=dtype)
            recon = self.decoder(y)
            loss = F.mse_loss(recon, o, reduction='none').sum([-3, -2, -1]).mean()
            metrics = {'decoder_loss': loss}

        self.decoder_optimizer.step(loss, o.shape[0], it)
        return metrics

    def train(self, it):
        """Train the world model for one iteration."""
        idx = self.replay_buffer.sample_idx(self.batch_size, self.rng)
        target_device = self.wm.device
        o, stacked_a, next_r, next_term, next_o = self.replay_buffer.get(idx, 'o', 'a', 'next_r', 'next_term', 'next_o')
        o, stacked_a, next_r, next_term, next_o = (
            o.to(target_device),
            stacked_a.to(target_device),
            next_r.to(target_device),
            next_term.to(target_device),
            next_o.to(target_device),
        )

        self.wm.train()
        metrics, y = self._optimize(o, stacked_a, next_r, next_term, next_o, it)

        if self.eval_mode == 'decoder':
            self.decoder.train()
            decoder_metrics = self._optimize_decoder(o, y, it)
            metrics.update(decoder_metrics)

        return metrics, y

    @torch.no_grad()
    def evaluate(self, agent, seed):
        """Evaluate the world model."""
        if self.eval_mode != 'decoder':
            return {}

        wm, decoder, replay_buffer = self.wm, self.decoder, self.replay_buffer
        metrics = {}

        eval_rng = np.random.Generator(np.random.PCG64(seed))

        # Visualize reconstructions and dreams
        wm.eval()
        decoder.eval()
        with self.autocast():
            # Choose a few fixed and random observations from the buffer for visualization
            num_obs = 3
            fixed_idx = torch.linspace(0, self.init_steps, num_obs).long()
            fixed_o, fixed_term, fixed_trunc = replay_buffer.get(fixed_idx, 'next_o', 'next_term', 'next_trunc')
            random_idx = eval_rng.choice(len(replay_buffer), num_obs, replace=False)
            random_o, random_term, random_trunc = replay_buffer.get(random_idx, 'next_o', 'next_term', 'next_trunc')
            fixed_o, fixed_term, fixed_trunc = fixed_o.to(wm.device), fixed_term.to(wm.device), fixed_trunc.to(wm.device)
            random_o, random_term, random_trunc = random_o.to(wm.device), random_term.to(wm.device), random_trunc.to(wm.device)

            o = torch.cat([fixed_o, random_o], 0)
            term = torch.cat([fixed_term, random_term], 0)
            trunc = torch.cat([fixed_trunc, random_trunc], 0)
            cont_mask = utils.get_mask(~(term | trunc))

            # Encode and decode the observations
            y = wm.encode(o)
            ohat = decoder(y)
            # (N, F * C, H, W) -> (N, F, C, H, W)
            num_frames = wm.observation_space.shape[0]
            o = wm.preprocess(o, dtype=ohat.dtype)
            o, ohat = [x.unflatten(1, (num_frames, -1)) for x in (o, ohat)]

            # Create a visualization of the reconstructions
            img = torch.cat([utils.visualize_observations(o), utils.visualize_observations(ohat.clamp(0.0, 1.0))], 0)
            img = make_grid(img, nrow=num_obs * 2)
            img = (img.permute(1, 2, 0) * 255.0).byte().cpu().numpy()
            metrics['recons'] = wandb.Image(img)

            # Synthesize trajectories (dream)
            ys = wm.imagine(agent, self.dream_horizon, y, cont_mask)[0]
            ys = torch.stack(ys, 0)
            ohat = utils.apply_seq(decoder, ys)
            # (T, N, F * C, H, W) -> (T, N, F, C, H, W)
            ohat = ohat.unflatten(2, (num_frames, -1))

            # Create a video of the synthesized observations
            video = utils.apply_seq(utils.visualize_observations, ohat.clamp(0.0, 1.0))
            video = [make_grid(video[t], nrow=num_obs) for t in range(video.shape[0])]
            video.append(torch.zeros_like(video[-1]))  # append empty frame to visualize ends
            video = torch.stack(video, 0)
            video = (video * 255.0).byte().cpu().numpy()
            metrics['dream'] = wandb.Video(video, fps=10, format='gif')

        return metrics
