import gym
from stable_baselines3 import DQN
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor


def make_env():
    env = gym.make("BreakoutNoFrameskip-v4")
    env = AtariWrapper(env)
    env = Monitor(env)
    return env


def main():
    env = DummyVecEnv([make_env])
    model = DQN(
        "CnnPolicy",
        env,
        verbose=1,
        buffer_size=50000,
        learning_starts=10000,
        target_update_interval=1000,
        exploration_fraction=0.1,
    )
    model.learn(total_timesteps=100000)
    model.save("dqn_breakout")


if __name__ == "__main__":
    main()
