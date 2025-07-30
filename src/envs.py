from functools import partial

import gymnasium as gym
import numpy as np


def make_env(env_name, make, env_config):
    suite, game = env_name.split(':', 1)
    if suite == 'dm_control':
        env = dmc(game=game, make=make, **env_config)
    elif suite == 'atari':
        env = atari(game=game, make=make, **env_config)
    elif suite == 'safety_gym':
        raise NotImplementedError('Safety Gym environments are not supported yet.')
    else:
        raise ValueError(f'Unsupported: {suite}')
    return env


def safety_gymnasium(game, make, size):
    import safety_gymnasium as safetygym

    version = 'v0'
    env_id = f'{game}-{version}'
    wrappers = []

    kwargs = dict(render_mode="rgb_array", camera_name="vision", width=size[0], height=size[1])
    if make:
        env = safetygym.make(env_id, **kwargs)
        for wrapper in wrappers:
            env = wrapper(env)
        return env
    else:
        return env_id, wrappers, kwargs


def dmc(game, make, camera_id, width, height, frame_stack, action_repeat):
    import shimmy

    gym.register_envs(shimmy)

    domain, task = game.split("_", 1)
    version = 'v0'
    env_id = f"dm_control/{domain}-{task}-{version}"
    if camera_id is None:
        camera_id = dict(quadruped=2).get(domain, 0)
    render_kwargs = dict(camera_id=camera_id, width=width, height=height)

    wrappers = [
        partial(DMCWrapper, action_repeat=action_repeat),
        partial(gym.wrappers.FrameStackObservation, stack_size=frame_stack),
    ]
    kwargs = dict(render_mode="rgb_array", render_kwargs=render_kwargs)

    if make:
        env = gym.make(env_id, render_mode="rgb_array", render_kwargs=render_kwargs)
        for wrapper in wrappers:
            env = wrapper(env)
        return env
    else:
        return env_id, wrappers, kwargs


def atari(
    game,
    make,
    sticky,
    full_action_space,
    max_frames,
    noop_max,
    resolution,
    grayscale,
    frame_skip,
    frame_stack,
    episodic_life,
):
    """Create an Atari environment."""
    if max_frames > 108000:
        raise NotImplementedError('NoFrameskip-v4 environments do not support max_frames > 108000')

    version = 'v0' if sticky else 'v4'
    env_id = f'ale_py:{game}NoFrameskip-{version}'
    wrappers = [
        partial(gym.wrappers.TimeLimit, max_episode_steps=max_frames),
        partial(
            gym.wrappers.AtariPreprocessing,
            noop_max=noop_max,
            frame_skip=frame_skip,
            screen_size=resolution,
            terminal_on_life_loss=False,
            grayscale_obs=grayscale,
            grayscale_newaxis=True,
        ),
        partial(gym.wrappers.FrameStackObservation, stack_size=frame_stack),
    ]

    if episodic_life:
        wrappers.append(EpisodicLifeWrapper)

    if game == 'Breakout':
        # fire on reset for Breakout
        if episodic_life:
            wrappers.append(FireResetWrapper)
        else:
            wrappers.append(FireLifeWrapper)

    kwargs = dict(full_action_space=full_action_space)

    if make:
        env = gym.make(env_id, **kwargs)
        for wrapper in wrappers:
            env = wrapper(env)
        return env
    else:
        return env_id, wrappers, kwargs


# region DMC wrappers


class DMCWrapper(gym.Wrapper):
    def __init__(self, env, action_repeat=1):
        super().__init__(env)
        self._action_repeat = action_repeat
        width = self.env.spec.kwargs.get('render_kwargs').get('width')
        height = self.env.spec.kwargs.get('render_kwargs').get('height')
        self.observation_space = gym.spaces.Box(0, 255, (height, width, 3), dtype=np.uint8)

    def reset(self, **kwargs):
        _, info = self.env.reset(**kwargs)
        o = self.env.render()
        return o, info

    def step(self, action):
        reward = 0
        for _ in range(self._action_repeat):
            _, next_r, next_term, next_trunc, info = self.env.step(action)
            reward += next_r
            if next_term or next_trunc:
                break

        next_o = self.env.render()
        return next_o, reward, next_term, next_trunc, info


# endregion


# region Atari wrappers


class EpisodicLifeWrapper(gym.Wrapper):
    # different from AtariPreprocessing: real reset only when game over

    def __init__(self, env):
        super().__init__(env)
        self.lives = 0
        self.game_over = True

    def _ale_lives(self):
        return self.env.unwrapped.ale.lives()

    def reset(self, **kwargs):
        if self.game_over:
            o, info = self.env.reset(**kwargs)
        else:
            # noop after lost life
            o, _, _, _, info = self.env.step(0)
        self.lives = self._ale_lives()
        return o, info

    def step(self, action):
        next_o, next_r, next_term, next_trunc, info = self.env.step(action)
        self.game_over = next_term or next_trunc or self.game_over
        lives = self._ale_lives()
        if lives < self.lives and lives > 0:
            next_term = True
        self.lives = lives
        return next_o, next_r, next_term, next_trunc, info


class FireResetWrapper(gym.Wrapper):
    # adopted from https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/atari_wrappers.py

    def __init__(self, env):
        super().__init__(env)
        action_meanings = env.unwrapped.get_action_meanings()
        self.fire = len(action_meanings) >= 3 and action_meanings[1] == 'FIRE'
        self.reset_reward = None

    def reset(self, **kwargs):
        if not self.fire:
            return self.env.reset(**kwargs)

        self.env.reset(**kwargs)

        _, next_r, next_term, next_trunc, _ = self.env.step(1)
        self.reset_reward = next_r
        if next_term or next_trunc:
            self.env.reset(**kwargs)
            self.reset_reward = 0

        o, next_r, next_term, next_trunc, info = self.env.step(2)
        self.reset_reward += next_r
        if next_term or next_trunc:
            o, info = self.env.reset(**kwargs)
            self.reset_reward = None

        return o, info

    def step(self, action):
        next_o, next_r, next_term, next_trunc, info = self.env.step(action)
        if self.reset_reward is not None:
            next_r += self.reset_reward
            self.reset_reward = None
        return next_o, next_r, next_term, next_trunc, info


class FireLifeWrapper(gym.Wrapper):
    # adopted from https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/atari_wrappers.py

    def __init__(self, env):
        super().__init__(env)
        action_meanings = env.unwrapped.get_action_meanings()
        self.fire = len(action_meanings) >= 3 and action_meanings[1] == 'FIRE'
        self.lives = 0
        self.reset_reward = None

    def _ale_lives(self):
        return self.env.unwrapped.ale.lives()

    def reset(self, **kwargs):
        if not self.fire:
            return self.env.reset(**kwargs)

        self.env.reset(**kwargs)

        _, next_reward, next_term, next_trunc, _ = self.env.step(1)
        self.reset_reward = next_reward
        if next_term or next_trunc:
            self.env.reset(**kwargs)
            self.reset_reward = 0

        o, next_reward, next_term, next_trunc, info = self.env.step(2)
        self.reset_reward += next_reward
        if next_term or next_trunc:
            o, info = self.env.reset(**kwargs)
            self.reset_reward = None

        self.lives = self._ale_lives()
        return o, info

    def step(self, action):
        next_o, next_r, next_term, next_trunc, info = self.env.step(action)
        if self.reset_reward is not None:
            next_r += self.reset_reward
            self.reset_reward = None

        if not self.fire:
            return next_o, next_r, next_term, next_trunc, info

        lives = self._ale_lives()
        if not (next_term or next_trunc) and lives < self.lives and lives > 0:
            _, next_r_, next_term, next_trunc, _ = self.env.step(1)
            self.reset_reward = next_r_
            if not (next_term or next_trunc):
                _, next_r_, next_term, next_trunc, _ = self.env.step(2)
                self.reset_reward += next_r_
        self.lives = lives
        return next_o, next_r, next_term, next_trunc, info


# endregion


if __name__ == "__main__":
    from shimmy.registration import DM_CONTROL_SUITE_ENVS

    env_ids = [f"dm_control/{'-'.join(item)}-v0" for item in DM_CONTROL_SUITE_ENVS]
    # Example usage
    env = dmc("acrobot_swingup", make=True, camera_id=None, width=64, height=64, action_repeat=2)
    env.reset(seed=42)
    print(env.render())
    print(env.observation_space())
    env.close()
