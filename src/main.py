from argparse import ArgumentParser
from pathlib import Path

import ruamel.yaml as yaml
import torch
import wandb
import logging

import envs
import utils
from agent import Agent
from policy.ac import ActorCriticPolicy
from policy.ac_lagrange import ActorCriticLagrangePolicy
from policy.cpo import CPOPolicy
from wm import WorldModel, WorldModelDecomposed
from causal import PDAGLearning
from trainer import Trainer, TrainerCausalWM

logging.basicConfig(level=logging.DEBUG)


def main():
    # Parse command line arguments
    parser = ArgumentParser()
    parser.add_argument("--device", type=str, required=True, help="The device used for training")
    parser.add_argument("--buffer_device", type=str, required=False, help="The device used for buffer")
    parser.add_argument("--game", type=str, required=True, help='The Atari game, e.g., "Breakout"')
    parser.add_argument("--seed", type=int, required=True, help="The random seed to use for reproducibility")
    parser.add_argument("--config", type=str, required=True, help="The configuration file")
    parser.add_argument("--mode", type=str, nargs="?", help="The W&B mode")
    parser.add_argument("--project", type=str, nargs="?", help="The W&B project")
    parser.add_argument("--notes", type=str, nargs="?", help="The W&B notes")
    parser.add_argument("--wm_eval", type=str, default="none", help='The type of world model evaluation, one of "none", "decoder"')
    parser.add_argument("--agent_eval", type=str, default="all", help='The type of agent evaluation, one of "none", "all", "final"')
    parser.add_argument("--amp", default=False, action="store_true", help="Whether to use automatic mixed precision")
    parser.add_argument("--compile", default=False, action="store_true", help="Whether to use torch.compile")
    parser.add_argument("--save", default=False, action="store_true", help="Whether to save the models after training")
    parser.add_argument("--use_decom_wm", type=bool, default=True, help="Whether to use the decomposed world model")
    args = parser.parse_args()

    # Load the configuration file
    with open(args.config, "r") as f:
        config = yaml.YAML(typ="safe", pure=True).load(f)

    # Update the configuration with the command line arguments
    config = {
        **config,
        "config": args.config,
        "game": args.game,
        "seed": args.seed,
        "wm_eval": args.wm_eval,
        "agent_eval": args.agent_eval,
        "amp": args.amp,
        "compile": args.compile,
        "save": args.save,
    }

    # Initialize W&B
    wandb.init(project=args.project, mode=args.mode, notes=args.notes, config=config)
    config = wandb.config

    # Setup the device, torch.autocast, and torch.compile
    device = torch.device(args.device)
    buffer_device = torch.device(args.buffer_device if hasattr(args, "buffer_device") and args.buffer_device else args.device)
    autocast = lambda: torch.autocast(device_type=device.type, enabled=config.amp)
    # The arguments of torch.compile can be set here
    compile_ = lambda mod: torch.compile(mod, dynamic=True, disable=not config.compile)

    # Initialize the environment, policy, agent, and world model
    seed = (config.seed + 17) * 13
    rng = utils.seed_everything(seed)
    env = envs.make_env(config.game, make=True, env_config=config.env)

    y_dim = config.wm["y_dim"] if type(config.wm["y_dim"]) is not dict else sum(config.wm["y_dim"].values()) * 2
    a_dim = env.action_space.n if hasattr(env.action_space, "n") else env.action_space.shape[0]
    if config.policy["algo"] == "ac":
        policy = ActorCriticPolicy(y_dim, a_dim, config.policy["actor"], config.policy["critic"], compile_=compile_, device=device)
    elif config.policy["algo"] == "ac_lag":
        policy = ActorCriticLagrangePolicy(y_dim, a_dim, config.policy["actor"], config.policy["reward_critic"], config.policy["cost_critic"], compile_=compile_, device=device)
    elif config.policy["algo"] == "cpo":
        policy = CPOPolicy(y_dim, a_dim, config.policy["actor"], config.policy["reward_critic"], config.policy["cost_critic"], compile_=compile_, device=device)
    else:
        raise NotImplementedError("algo not implemented")
    agent = Agent(policy, env.action_space, config.action_stack)

    # Exploration agent
    explore_policy = ActorCriticPolicy(y_dim, a_dim, config.explore_policy["actor"], config.explore_policy["critic"], compile_=compile_, device=device)
    explore_agent = Agent(explore_policy, env.action_space, config.action_stack)

    wm = (
        WorldModel(env.observation_space, agent.stacked_action_space, **config.wm, compile_=compile_, device=device)
        if not args.use_decom_wm
        else WorldModelDecomposed(env.observation_space, agent.stacked_action_space, **config.wm, compile_=compile_, device=device)
    )

    # Causal dynamics model
    cdm = (
        PDAGLearning(observation_space=env.observation_space, single_action_space=agent.single_action_space, y_dim=y_dim, compile_=compile_, device=device)
        if args.use_decom_wm
        else None
    )

    # Initialize the trainer
    trainer = (
        Trainer(
            env,
            config.game,
            wm,
            agent,
            seed,
            **config.trainer,
            use_decom_wm=args.use_decom_wm,
            wm_eval=config.wm_eval,
            agent_eval=config.agent_eval,
            buffer_device=buffer_device,
            rng=rng,
            autocast=autocast,
            compile_=compile_,
        )
        if not args.use_decom_wm
        else TrainerCausalWM(
            env,
            config.game,
            config.policy["algo"],
            wm,
            cdm,
            agent,
            explore_agent,
            seed,
            **config.trainer,
            use_decom_wm=args.use_decom_wm,
            wm_eval=config.wm_eval,
            agent_eval=config.agent_eval,
            buffer_device=buffer_device,
            rng=rng,
            autocast=autocast,
            compile_=compile_,
        )
    )

    print(f"Starting... (seed: {seed})")
    print(f"World Model # params: {utils.count_params(wm)}")
    for attr in dir(wm):
        if any(attr.endswith(suffix) for suffix in ("projector", "predictor", "encoder", "decoder")):
            print(f"  {attr:<20}: {utils.count_params(getattr(wm, attr))}")
    print(f"Agent # params: {utils.count_params(agent)}")

    # Train the agent and world model
    while not trainer.is_finished():
        metrics = trainer.train()

        if len(metrics) > 0:
            wandb.log(metrics, step=trainer.it)

        if trainer.it == 0:
            print("Training...")  # good, everything worked up to this point

    # Save the models, if necessary
    if config.save:
        torch.save(wm.state_dict(), Path(wandb.run.dir) / "wm.pt")
        torch.save(agent.state_dict(), Path(wandb.run.dir) / "agent.pt")
        wandb.save("wm.pt")
        wandb.save("agent.pt")
        if config.wm_eval == "decoder":
            torch.save(trainer.wm_trainer.decoder.state_dict(), Path(wandb.run.dir) / "decoder.pt")
            wandb.save("decoder.pt")

    # Cleanup
    trainer.close()
    wandb.finish()


if __name__ == "__main__":
    main()
