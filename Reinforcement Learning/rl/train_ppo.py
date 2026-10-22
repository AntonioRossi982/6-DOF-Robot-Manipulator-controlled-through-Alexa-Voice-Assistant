import argparse
import os

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv

from rl import config as C
from rl.arduinobot_reach_env import ArduinobotReachEnv


def main():
    
    # Parse command line arguments for training configuration
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=200_000)
    p.add_argument("--envs", type=int, default=4)
    p.add_argument("--name", type=str, default="ppo_v1")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    # Define folders for saving logs/valuation (first folder) 
    # and models/checkpoints (second folder)
    os.makedirs(C.RUNS_DIR, exist_ok=True)
    os.makedirs(C.MODELS_DIR, exist_ok=True)

    # Create vectorized environments for training and a single environment for evaluation
    # Environments are going to update the same robot's policy in parallel.
    env = make_vec_env(ArduinobotReachEnv, n_envs=args.envs, seed=args.seed,
                       vec_env_cls=SubprocVecEnv)
    eval_env = Monitor(ArduinobotReachEnv())

    # Create the PPO (proximal policy optimization) model with specified hyperparameters for training
    model = PPO(
        "MlpPolicy", env, device="cpu", seed=args.seed, verbose=1,
        n_steps=1024, batch_size=256, n_epochs=10, gamma=0.98, learning_rate=3e-4,
        policy_kwargs=dict(net_arch=[128, 128]),
        tensorboard_log=C.RUNS_DIR,
    )

    # Define callbacks for evaluation and checkpointing during training
    callbacks = [
        EvalCallback(eval_env,
                     best_model_save_path=os.path.join(C.MODELS_DIR, args.name + "_best"),
                     log_path=os.path.join(C.RUNS_DIR, args.name + "_eval"),
                     eval_freq=max(20_000 // args.envs, 1), n_eval_episodes=30, deterministic=True),
        CheckpointCallback(save_freq=max(100_000 // args.envs, 1),
                           save_path=os.path.join(C.MODELS_DIR, args.name + "_ckpt")),
    ]

    # Train the model for the specified number of steps and save the final model
    model.learn(total_timesteps=args.steps, callback=callbacks, tb_log_name=args.name)
    model.save(os.path.join(C.MODELS_DIR, args.name + "_final"))



if __name__ == "__main__":
    main()