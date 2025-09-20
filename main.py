# ==============================================================================
# STEP 1: IMPORT NECESSARY LIBRARIES
# ==============================================================================

import os  # Used for creating directories to save our model and logs.
import gymnasium as gym  # The core library for reinforcement learning environments.
import retro  # The library that allows us to use classic game environments like Super Mario Bros.
import numpy as np  # Used for numerical operations, especially for the random number generator.
from gymnasium.wrappers import TimeLimit  # A wrapper to end an episode after a certain number of steps.
from stable_baselines3 import PPO  # The Proximal Policy Optimization (PPO) algorithm we will use to train our agent.
from stable_baselines3.common.atari_wrappers import ClipRewardEnv, WarpFrame  # Pre-built wrappers for preprocessing game environments.
from stable_baselines3.common.callbacks import CheckpointCallback  # A tool to save our model's progress during training.
from stable_baselines3.common.vec_env import (
    SubprocVecEnv,      # A tool to run multiple environments in parallel in separate CPU processes.
    VecFrameStack,      # A wrapper to stack consecutive frames together to give the agent a sense of motion.
    VecTransposeImage,  # A wrapper to re-order the dimensions of the image data for the neural network.
    VecMonitor,         # A vectorized version of the Monitor wrapper to log statistics across multiple environments.
)

# ==============================================================================
# STEP 2: CREATE THE CUSTOM FRAME-SKIPPING WRAPPER
# ==============================================================================

# We created a StochasticFrameSkip class that will now inherit everything from gym.Wrapper.
# A wrapper is a special class that "wraps around" an environment to modify its behavior.
# In this case, we are modifying how the 'step' function works to implement frame skipping
# with a bit of randomness (stochasticity) to make the agent's learning more robust
class StochasticFrameSkip(gym.Wrapper):
    # The __init__ method is the constructor for this class.
    # It takes the environment to wrap ('env') and two parameters, 'n' and 'stickprob'.
    def __init__(self, env, n, stickprob):
        # We must call the parent Wrapper's constructor to properly set up the wrapper.
        gym.Wrapper.__init__(self, env)
        # 'n' is the number of frames to repeat the same action for. This is our frame skip amount.
        self.n = n
        # 'stickprob' is the probability of the action "sticking" (repeating) by chance.
        self.stickprob = stickprob
        # 'current_action' will store the current action being repeated across the skipped frames.
        self.current_action = None
        # This creates an isolated random number generator for this wrapper to use.
        self.rng = np.random.RandomState()
        # This is a safe check to see if the underlying environment supports a faster, non-rendering step.
        # hasattr() checks if an object has an attribute without causing an error.
        self.supports_want_render = hasattr(env, "supports_want_render")

    # The reset method is called at the beginning of every new game/episode.
    # **kwargs allows passing any extra arguments to the underlying environment's reset method.
    def reset(self, **kwargs):
        # We reset the current action to None because the episode is starting over.
        self.current_action = None
        # We then call the reset method of the environment we are wrapping ('self.env')
        # and return whatever it gives back. This is called "delegation".
        return self.env.reset(**kwargs)

    # The step method is called every time the agent chooses a new action.
    def step(self, ac):
        # Initialize variables for the frame-skipping loop.
        total_reward = 0
        # This loop runs 'n' times (e.g., 4 times) for each single action 'ac' from the agent.
        for i in range(self.n):
            # The logic here determines if we should use the new action or stick with the old one.
            if self.current_action is None:
                self.current_action = ac
            elif i == 0:
                if self.rng.rand() > self.stickprob:
                    self.current_action = ac
            elif i == 1:
                self.current_action = ac
            
            # This is the optimization we checked earlier. If the feature exists and this is
            # one of the first 3 frames, we tell the emulator to run the game logic but not to
            # waste time drawing the screen (that is why want_render = False).
            if self.supports_want_render and i < self.n - 1:
                obs, rew, terminated, truncated, info = self.env.step(self.current_action, want_render=False)
            else: # Otherwise we do a normal step that also renders.
                obs, rew, terminated, truncated, info = self.env.step(self.current_action)
            
            # Add the reward from this single frame to our total for the 4-frame step.
            total_reward += rew
            # If the game ends (terminated) or the time limit is hit (truncated), we must stop immediately.
            if terminated or truncated:
                break
        # Return the final observation, the summed reward over 4 frames, and the game status.
        return obs, total_reward, terminated, truncated, info

# ==============================================================================
# STEP 3: THE MAIN FUNCTION WHERE EVERYTHING COMES TOGETHER
# ==============================================================================

def main():
    # --- Configuration ---
    GAME_NAME = "SuperMarioBros-Nes"
    GAME_STATE = "Level1-1"
    # Set the number of parallel environments. For my Ryzen 5 5600H (6 Cores, 12 Threads).
    # A value of 12 is a great choice to keep CPU busy gathering data for GPU.
    NUM_ENVS = 12
    TOTAL_TIMESTEPS = 10_000_000

    # --- Directory Setup ---
    # Not using ./ will also works as os.makedirs() handles it
    models_dir = "./models/PPO" 
    log_dir = "./tensorboard"
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # --- Define the Environment Blueprint Function ---
    # This is an "inner function" that acts as a recipe for creating a single, fully-wrapped environment.
    # It's a clean way to bundle all the setup steps.
    def make_env():
        # 1. Create the base retro environment from the game's ROM.
        env = retro.make(game=GAME_NAME, state=GAME_STATE)

        # 2. Apply our custom frame-skipping wrapper. 'n=4' is a standard value from the DeepMind paper.
        env = StochasticFrameSkip(env, n=4, stickprob=0.25)

        # 3. Add a time limit. This prevents an episode from running forever if the agent gets stuck.
        env = TimeLimit(env, max_episode_steps=4500)

        # 4. Apply the DeepMind-style wrappers for preprocessing.
        # WarpFrame resizes the image to 84x84 and converts it to grayscale to reduce complexity.
        env = WarpFrame(env, width=84, height=84)
        # ClipRewardEnv clips all positive rewards to +1 and negative rewards to -1, which helps stabilize training.
        env = ClipRewardEnv(env)
        
        return env

    # --- Vectorize the Environment ---
    # This is where the parallelism happens.
    # We create a list of 'recipes' (function references) and give it to SubprocVecEnv.
    # It then spawns a separate CPU process for each recipe, running them all at the same time.
    env = SubprocVecEnv([make_env for _ in range(NUM_ENVS)])
    # VecMonitor is a vectorized version of the Monitor wrapper.
    # It logs important statistics like episode rewards and lengths across all parallel environments.
    env = VecMonitor(env)
    # We now apply final wrappers that work on the vectorized environment.
    # VecFrameStack stacks 4 consecutive frames together into one observation (shape: 4, 84, 84).
    # This is crucial for the agent to perceive motion and velocity.
    env = VecFrameStack(env, n_stack=4)
    # The neural network (PyTorch) expects image data in the order (Channels, Height, Width).
    # The environment provides it as (Height, Width, Channels). This wrapper fixes the order.
    env = VecTransposeImage(env)

    # --- Callback for Saving Models ---
    # This sets up a tool to automatically save the model's progress during training.
    checkpoint_callback = CheckpointCallback(
        # We divide by NUM_ENVS because the total 
        # steps are counted across all parallel environments.
        # // means integer division (floor division),
        # e.g. 50000 // 12 = 4166 instead of 4166.666...
        save_freq=50000 // NUM_ENVS,
        save_path=models_dir,
        name_prefix="SMB",
    )
    
    # --- PPO Model Definition ---
    # This is where we define our agent using the PPO algorithm.
    model = PPO(
        policy="CnnPolicy", # 'CnnPolicy' is a Convolutional Neural Network, perfect for image-based inputs.
        env=env, # The vectorized and wrapped environment we just created.
        learning_rate=2.5e-5, # A standard, stable learning rate for PPO on Atari games.
        n_steps=512, # The number of steps each environment runs before a model update. (512*12 = 6144 total steps per update).
        batch_size=512, # The size of the data chunks used during a learning update.
        n_epochs=4, # How many times the model goes over the collected data during each update.
        gamma=0.99, # DEFAULT: The discount factor. A value close to 1 makes the agent value long-term rewards.
        gae_lambda=0.95, # DEFAULT: A parameter for the GAE algorithm, which helps estimate the advantage of actions.
        clip_range=0.05, # The core of PPO. It limits how much the policy can change, ensuring stability.
        ent_coef=0.01, # The entropy coefficient. It encourages exploration by rewarding the agent for taking uncertain actions.
        verbose=1, # 'verbose=1' tells Stable Baselines to print out training progress to the console.
        tensorboard_log=log_dir, # Specifies the directory to save logs for TensorBoard visualization.
        device="cuda", # This crucial line tells the model to use your RTX 3060 GPU for the heavy lifting (the neural network updates).
    )

    # --- Start Training ---
    # This is the command that starts the entire training process.
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS, # The total number of simulation steps to run.
        callback=checkpoint_callback, # Tells the learning process to use our automatic model saver.
        tb_log_name="PPO", # The name for this specific training run in TensorBoard.
    )
    
    # --- Save Final Model ---
    model.save(f"{models_dir}/SMB_final") 

    # Always a good idea to clean up and close the environment when done.
    # it is optional here as the script ends, but a good habit.
    env.close() 

if __name__ == "__main__":
    main()