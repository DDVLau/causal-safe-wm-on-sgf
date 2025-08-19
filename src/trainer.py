import time

import torch

import replay
import utils
import gymnasium as gym
from agent import AgentTrainer, AgentExplorationTrainer, AgentCPOTrainer
from wm import WorldModelTrainer, WorldModelDecomposedTrainer
from causal import PDAGTrainer
from cdm import CausalDynamicModel


class TrainerCausalWM:
    """Trainer for the agent and world model."""

    def __init__(
        self,
        env,
        game,
        algo,
        wm,
        cdm,
        agent,
        explore_agent,
        seed,
        env_steps,
        init_steps,
        env_epsilon,
        env_temperature,  # Not used
        wm_every,
        cdm_every,
        agent_every,
        pdag_every,
        log_every,
        eval_every,
        cdm_start,
        pdag_start,
        wm_trainer,
        cdm_trainer,
        agent_trainer,
        explore_agent_trainer,
        wm_eval,
        agent_eval,
        buffer_device,
        *,
        use_decom_wm,
        rng,
        autocast,
        compile_,
    ):
        self.env = env
        self.wm = wm
        self.cdm = cdm
        self.agent = agent
        self.explore_agent = explore_agent
        self.seed = seed
        self.env_steps = env_steps
        self.init_steps = init_steps
        self.env_epsilon = env_epsilon
        self.wm_every = wm_every
        self.cdm_every = cdm_every
        self.pdag_every = pdag_every
        self.agent_every = agent_every
        self.log_every = log_every
        self.eval_every = eval_every
        self.cdm_start = cdm_start
        self.pdag_start = pdag_start
        self.rng = rng
        self.autocast = autocast

        # Initialize the replay buffer
        start_o, _ = env.reset(seed=seed)

        buffer_class = replay.ReplayBufferSafeRL if use_decom_wm else replay.ReplayBuffer
        replay_buffer = buffer_class(env.observation_space, agent.stacked_action_space, env_steps, start_o, device=buffer_device)
        self.replay_buffer = replay_buffer

        # Initialize the world model and agent trainers
        if not use_decom_wm:
            self.wm_trainer = WorldModelTrainer(
                wm, replay_buffer, **wm_trainer, init_steps=init_steps, eval_mode=wm_eval, total_its=env_steps, rng=rng, autocast=autocast, compile_=compile_
            )
        else:
            self.wm_trainer = WorldModelDecomposedTrainer(
                wm, replay_buffer, **wm_trainer, init_steps=init_steps, eval_mode=wm_eval, total_its=env_steps, rng=rng, autocast=autocast, compile_=compile_
            )
        if algo == "cpo":
            self.agent_trainer = AgentCPOTrainer(
                game, agent, wm, replay_buffer, **agent_trainer, eval_mode=agent_eval, total_its=env_steps, rng=rng, autocast=autocast, compile_=compile_
            )
        else:
            self.agent_trainer = AgentTrainer(
                game, agent, wm, replay_buffer, **agent_trainer, eval_mode=agent_eval, total_its=env_steps, rng=rng, autocast=autocast, compile_=compile_
            )

        self.explore_agent_trainer = AgentExplorationTrainer(
            game, explore_agent, wm, cdm, replay_buffer, **explore_agent_trainer, eval_mode=agent_eval, total_its=env_steps, rng=rng, autocast=autocast, compile_=compile_
        )

        self.pdag_trainer = PDAGTrainer(cdm, **cdm_trainer, total_its=env_steps, rng=rng, autocast=autocast, compile_=compile_)
        self.pdag_evaluator = CausalDynamicModel(wm.y_dim, env.action_space.shape[0])

        self.it = -1
        self.wm_it = 0
        self.agent_it = 0
        self.explore_agent_it = 0
        self.cdm_it = 0
        self.pdag_it = 0
        self.wm_agg = utils.Aggregator(op="mean")
        self.agent_agg = utils.Aggregator(op="mean")
        self.explore_agent_agg = utils.Aggregator(op="mean")
        self.cdm_agg = utils.Aggregator(op="mean")
        self.pdag_agg = utils.Aggregator(op="mean")
        self.train_time = 0.0
        self.wm_time = 0.0
        self.agent_time = 0.0
        self.cdm_time = 0.0
        self.last_log = -1
        self.last_eval = -1

        # Initialize the agent state and prefill the replay buffer
        agent.eval()
        with autocast():
            with torch.no_grad():
                self.agent_state = agent.start()
                # self.cont_mask = utils.get_mask(torch.ones(1, dtype=torch.bool, device=replay_buffer.device))
                self.cont_mask = utils.get_mask(torch.ones(1, dtype=torch.bool, device=self.wm.device))
                for _ in range(init_steps):
                    a, stacked_a, self.agent_state = agent.act_randomly(self.agent_state, self.cont_mask, self.rng)
                    self._env_step(a, stacked_a)

    def close(self):
        self.agent_trainer.close()

    def is_finished(self):
        return self.it >= (self.env_steps - self.init_steps - 1)

    def _env_step(self, a, stacked_a):
        env = self.env
        action = a.detach().cpu().numpy()
        if isinstance(env.action_space, gym.spaces.Discrete) or isinstance(env.action_space, gym.spaces.MultiDiscrete):
            action = action.item()
        else:
            action = action.squeeze(0)
        next_o, next_r, next_term, next_trunc, info = env.step(action)
        next_c = info.get("cost", None)
        cont_o = next_o
        if next_term or next_trunc:
            cont_o, _ = env.reset()

        next_r, next_c, next_term, next_trunc, next_o = self.replay_buffer.step(stacked_a, next_o, next_r, next_c, next_term, next_trunc, cont_o)
        next_term, next_trunc = next_term.to(self.wm.device), next_trunc.to(self.wm.device)
        self.cont_mask = utils.get_mask(~(next_term | next_trunc))

    def train(self):
        """Train the agent and world model for one iteration."""

        wm, agent, replay_buffer = self.wm, self.agent, self.replay_buffer

        it = self.it + 1

        # Select an action using the agent
        agent.eval()
        wm.eval()
        with self.autocast():
            with torch.no_grad():
                o = self.replay_buffer.cont_o
                o = o.to(wm.device)
                y = wm.encode(o)
                if self.rng.random() < self.env_epsilon:
                    a, stacked_a, self.agent_state = agent.act_randomly(self.agent_state, self.cont_mask, self.rng)
                else:
                    a, stacked_a, self.agent_state = agent.act(self.agent_state, self.cont_mask, y)

        # Take a step in the environment
        self._env_step(a, stacked_a)

        # Train the world model
        start_time = time.time()
        start_y = None

        while self.wm_it <= it:
            wm_metrics, start_y = self.wm_trainer.train(it)
            self.wm_agg.append(wm_metrics)
            self.wm_it += self.wm_every

        wm_end_time = time.time()

        # Train the cdm
        while self.cdm_it <= it:
            if self.cdm_start <= it:
                cdm_metrics = self.pdag_trainer.train(start_y, wm, agent, self.explore_agent, it)
                self.cdm_agg.append(cdm_metrics)
            self.cdm_it += self.cdm_every

        cdm_end_time = time.time()

        # Train the agent
        while self.agent_it <= it:
            agent_metrics = self.agent_trainer.train(it, start_y)
            self.agent_agg.append(agent_metrics)
            self.agent_it += self.agent_every

        # Train the exploration agent
        while self.explore_agent_it <= it:
            explore_agent_metrics = self.explore_agent_trainer.train(it, start_y)
            self.explore_agent_agg.append(explore_agent_metrics)
            self.explore_agent_it += self.agent_every

        while self.pdag_it <= it:
            if self.pdag_start <= it:
                pdag_metrics = self.pdag_evaluator.compute(replay_buffer, wm, self.seed)
                self.pdag_agg.append(pdag_metrics)
            self.pdag_it += self.pdag_every

        end_time = time.time()  # (includes debug training time)
        self.train_time += end_time - start_time
        self.wm_time += wm_end_time - start_time
        self.cdm_time += cdm_end_time - wm_end_time
        self.agent_time += end_time - cdm_end_time

        # Create the metrics for logging
        metrics = {}
        is_first = it == 0
        is_final = it == self.env_steps - self.init_steps - 1

        if is_first or is_final or (it - self.last_log >= self.log_every):
            stats = {
                **replay_buffer.get_stats(),
                "train_time": self.train_time,
                "wm_time": self.wm_time,
                "cdm_time": self.cdm_time,
                "agent_time": self.agent_time,
            }
            metrics.update({f"stats/{k}": v for k, v in stats.items()})
            metrics.update({f"wm/{k}": v for k, v in self.wm_agg.aggregate().items()})
            metrics.update({f"cdm/{k}": v for k, v in self.cdm_agg.aggregate().items()})
            metrics.update({f"agent/{k}": v for k, v in self.agent_agg.aggregate().items()})
            metrics.update({f"explore_agent/{k}": v for k, v in self.explore_agent_agg.aggregate().items()})
            self.last_log = it

        if is_first or is_final or (it - self.last_eval >= self.eval_every):
            wm.eval()
            agent.eval()
            wm_eval_metrics = self.wm_trainer.evaluate(agent, self.seed)
            agent_eval_metrics = self.agent_trainer.evaluate(is_final, self.seed)
            pdag_metrics = self.pdag_evaluator.evaluate()  # Visualising pdag
            metrics.update({f"eval/{k}": v for k, v in wm_eval_metrics.items()})
            metrics.update({f"eval/{k}": v for k, v in agent_eval_metrics.items()})
            metrics.update({f"eval/{k}": v for k, v in pdag_metrics.items()})
            self.last_eval = it

        self.it = it
        return metrics


class Trainer:
    """Trainer for the agent and world model."""

    def __init__(
        self,
        env,
        game,
        wm,
        agent,
        seed,
        env_steps,
        init_steps,
        env_epsilon,
        env_temperature,  # Not used
        wm_every,
        agent_every,
        log_every,
        eval_every,
        wm_trainer,
        agent_trainer,
        wm_eval,
        agent_eval,
        buffer_device,
        *,
        use_decom_wm,
        rng,
        autocast,
        compile_,
    ):
        self.env = env
        self.wm = wm
        self.agent = agent
        self.seed = seed
        self.env_steps = env_steps
        self.init_steps = init_steps
        self.env_epsilon = env_epsilon
        self.wm_every = wm_every
        self.agent_every = agent_every
        self.log_every = log_every
        self.eval_every = eval_every
        self.rng = rng
        self.autocast = autocast

        # Initialize the replay buffer
        start_o, _ = env.reset(seed=seed)
        replay_buffer = replay.ReplayBuffer(env.observation_space, agent.stacked_action_space, env_steps, start_o, use_cost=use_decom_wm, device=buffer_device)
        self.replay_buffer = replay_buffer

        # Initialize the world model and agent trainers
        if not use_decom_wm:
            self.wm_trainer = WorldModelTrainer(
                wm, replay_buffer, **wm_trainer, init_steps=init_steps, eval_mode=wm_eval, total_its=env_steps, rng=rng, autocast=autocast, compile_=compile_
            )
        else:
            self.wm_trainer = WorldModelDecomposedTrainer(
                wm, replay_buffer, **wm_trainer, init_steps=init_steps, eval_mode=wm_eval, total_its=env_steps, rng=rng, autocast=autocast, compile_=compile_
            )

        self.agent_trainer = AgentTrainer(game, agent, wm, replay_buffer, **agent_trainer, eval_mode=agent_eval, total_its=env_steps, rng=rng, autocast=autocast, compile_=compile_)

        self.it = -1
        self.wm_it = 0
        self.agent_it = 0
        self.wm_agg = utils.Aggregator(op="mean")
        self.agent_agg = utils.Aggregator(op="mean")
        self.train_time = 0.0
        self.wm_time = 0.0
        self.agent_time = 0.0
        self.last_log = -1
        self.last_eval = -1

        # Initialize the agent state and prefill the replay buffer
        agent.eval()
        with autocast():
            with torch.no_grad():
                self.agent_state = agent.start()
                # self.cont_mask = utils.get_mask(torch.ones(1, dtype=torch.bool, device=replay_buffer.device))
                self.cont_mask = utils.get_mask(torch.ones(1, dtype=torch.bool, device=self.wm.device))
                for _ in range(init_steps):
                    a, stacked_a, self.agent_state = agent.act_randomly(self.agent_state, self.cont_mask, self.rng)
                    self._env_step(a, stacked_a)

    def close(self):
        self.agent_trainer.close()

    def is_finished(self):
        return self.it >= (self.env_steps - self.init_steps - 1)

    def _env_step(self, a, stacked_a):
        env = self.env
        action = a.detach().cpu().numpy()
        if isinstance(env.action_space, gym.spaces.Discrete) or isinstance(env.action_space, gym.spaces.MultiDiscrete):
            action = action.item()
        else:
            action = action.squeeze(0)
        next_o, next_r, next_term, next_trunc, info = env.step(action)
        next_c = info.get("cost", None)
        cont_o = next_o
        if next_term or next_trunc:
            cont_o, _ = env.reset()

        next_r, next_c, next_term, next_trunc, next_o = self.replay_buffer.step(stacked_a, next_o, next_r, next_c, next_term, next_trunc, cont_o)
        next_term, next_trunc = next_term.to(self.wm.device), next_trunc.to(self.wm.device)
        self.cont_mask = utils.get_mask(~(next_term | next_trunc))

    def train(self):
        """Train the agent and world model for one iteration."""

        wm, agent, replay_buffer = self.wm, self.agent, self.replay_buffer

        it = self.it + 1

        # Select an action using the agent
        agent.eval()
        wm.eval()
        with self.autocast():
            with torch.no_grad():
                o = self.replay_buffer.cont_o
                o = o.to(wm.device)
                y = wm.encode(o)
                if self.rng.random() < self.env_epsilon:
                    a, stacked_a, self.agent_state = agent.act_randomly(self.agent_state, self.cont_mask, self.rng)
                else:
                    a, stacked_a, self.agent_state = agent.act(self.agent_state, self.cont_mask, y)

        # Take a step in the environment
        self._env_step(a, stacked_a)

        # Train the world model
        start_time = time.time()
        start_y = None

        while self.wm_it <= it:
            wm_metrics, start_y = self.wm_trainer.train(it)
            self.wm_agg.append(wm_metrics)
            self.wm_it += self.wm_every

        wm_end_time = time.time()

        # Train the agent
        while self.agent_it <= it:
            agent_metrics = self.agent_trainer.train(it, start_y)
            self.agent_agg.append(agent_metrics)
            self.agent_it += self.agent_every

        end_time = time.time()  # (includes debug training time)
        self.train_time += end_time - start_time
        self.wm_time += wm_end_time - start_time
        self.agent_time += end_time - wm_end_time

        # Create the metrics for logging
        metrics = {}
        is_first = it == 0
        is_final = it == self.env_steps - self.init_steps - 1

        if is_first or is_final or (it - self.last_log >= self.log_every):
            stats = {**replay_buffer.get_stats(), "train_time": self.train_time, "wm_time": self.wm_time, "agent_time": self.agent_time}
            metrics.update({f"stats/{k}": v for k, v in stats.items()})
            metrics.update({f"wm/{k}": v for k, v in self.wm_agg.aggregate().items()})
            metrics.update({f"agent/{k}": v for k, v in self.agent_agg.aggregate().items()})
            self.last_log = it

        if is_first or is_final or (it - self.last_eval >= self.eval_every):
            wm.eval()
            agent.eval()
            wm_eval_metrics = self.wm_trainer.evaluate(agent, self.seed)
            agent_eval_metrics = self.agent_trainer.evaluate(is_final, self.seed)
            metrics.update({f"eval/{k}": v for k, v in wm_eval_metrics.items()})
            metrics.update({f"eval/{k}": v for k, v in agent_eval_metrics.items()})
            self.last_eval = it

        self.it = it
        return metrics
