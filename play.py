# ==============================================================================
# STEP 1: IMPORT NECESSARY LIBRARIES
# ==============================================================================

import os
import gymnasium as gym
import retro
import numpy as np
import cv2  # OpenCV is used to create the window for rendering
from gymnasium.wrappers import TimeLimit
from stable_baselines3 import PPO
from stable_baselines3.common.atari_wrappers import ClipRewardEnv, WarpFrame
from stable_baselines3.common.vec_env import (
    DummyVecEnv,        # Use DummyVecEnv for evaluation as it's simpler for a single environment
    VecFrameStack,
    VecTransposeImage,
)
from stable_baselines3.common.monitor import Monitor # Use the standard Monitor for single env logging

# ==============================================================================
# STEP 2: RE-CREATE THE CUSTOM FRAME-SKIPPING WRAPPER
# This class must be identical to the one in your training script.
# ==============================================================================

class StochasticFrameSkip(gym.Wrapper):
    def __init__(self, env, n, stickprob):
        gym.Wrapper.__init__(self, env)
        self.n = n
        self.stickprob = stickprob
        self.current_action = None
        self.rng = np.random.RandomState()
        self.supports_want_render = hasattr(env, "supports_want_render")

    def reset(self, **kwargs):
        self.current_action = None
        return self.env.reset(**kwargs)

    def step(self, ac):
        total_reward = 0
        for i in range(self.n):
            if self.current_action is None:
                self.current_action = ac
            elif i == 0:
                if self.rng.rand() > self.stickprob:
                    self.current_action = ac
            elif i == 1:
                self.current_action = ac
            
            if self.supports_want_render and i < self.n - 1:
                obs, rew, terminated, truncated, info = self.env.step(self.current_action, want_render=False)
            else:
                obs, rew, terminated, truncated, info = self.env.step(self.current_action)
            
            total_reward += rew
            if terminated or truncated:
                break
        return obs, total_reward, terminated, truncated, info

# ==============================================================================
# STEP 3: MAIN FUNCTION FOR INFERENCE AND EVALUATION
# ==============================================================================

def main():
    # --- Configuration ---
    GAME_NAME = "SuperMarioBros-Nes"
    GAME_STATE = "Level1-1"
    
    # IMPORTANT: Update this path to point to the specific model you want to test.
    # Your training script saves checkpoints (e.g., SMB_50000_steps.zip) and a final model.
    # The 'SMB_final.zip' created by the callback is often a good choice.
    MODEL_PATH = "./models/PPO/SMB_final.zip"

    # --- Environment Setup Function ---
    # This must be identical to your training script, with one key change for rendering.
    def make_env():
        # 1. Create the base retro environment.
        #    KEY CHANGE: Set render_mode to 'rgb_array'. This makes the env.render() method
        #    return the pixel data as an array, instead of opening its own window.
        env = retro.make(game=GAME_NAME, state=GAME_STATE, render_mode='rgb_array')
        
        # 2. Add the Monitor wrapper. This lets us see the final score in the terminal when Mario dies.
        # This is for evaluation only, to check performance.
        env = Monitor(env)

        # 3. Apply all the same wrappers in the same order as in training.
        env = StochasticFrameSkip(env, n=4, stickprob=0.25)
        env = TimeLimit(env, max_episode_steps=4500)
        env = WarpFrame(env, width=84, height=84)
        env = ClipRewardEnv(env)
        
        return env

    # --- Create and Wrap the Environment for Inference ---
    # We use DummyVecEnv for simplicity when running a single environment for viewing.
    # DummyVecEnv handles the 5 values returned by env.step() to 4 values.
    env = DummyVecEnv([make_env])
    env = VecFrameStack(env, n_stack=4)
    env = VecTransposeImage(env)

    # --- Load the Trained Model ---
    model = PPO.load(MODEL_PATH, env=env)

    # --- Run the Agent in a Loop ---
    obs = env.reset()
    print("Press 'q' in the game window to quit.")
    while True:
        # model.predict() uses the trained neural network to choose the best action.
        # It outputs, (array([1, 0, 0, 0, 1, 0, 1, 1, 0]), None), that is why we have action, _
        # Why is that? Because the designer kept it a tuple for other policies that might return states.
        # 'deterministic=True' makes the agent always choose the best action, disabling exploration.
        action, _ = model.predict(obs, deterministic=True)
        
        # Take the action in the environment.
        # dones is a boolean array indicating if the episode has ended.
        # e.g., dones = [False] or dones = [True], case of single env.
        # info is a list containing dictionaries that may contain episode statistics when an episode ends.
        # e.g., info = [{'episode': {'r': 123.0, 'l': 456, 't': 7890.0}}] --> [{key:value}]
        obs, rewards, dones, info = env.step(action)

        # Render the game frame in our own controlled window.
        frame = env.render()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) # Convert color format for OpenCV
        cv2.imshow("Super Mario Bros. - Trained Agent", frame)
        
        # --- ONLY RUNS ONCE PER EPISODE ---
        # If the episode is over, print the score from the Monitor wrapper and reset.
        if dones[0]:
            # episode_info returns value from [{key:value}] with episode statistics.
            episode_info = info[0].get("episode")
            if episode_info:
                print(f"Episode finished! Score: {episode_info['r']:.2f}, Length: {episode_info['l']} steps")
            obs = env.reset()

        # Check if the 'q' key is pressed to close the window.
        # cv2.waitKey(1) ensures the window updates and checks for key presses every 1 ms.
        # 0xFF gives the last 8 bits of the key code, ensuring compatibility across systems.
        # so (cv2.waitKey(1) & 0xFF) together gives the correct ASCII code of the key pressed.
        # ord('q') gets the ASCII code for 'q'.
        # ord means "ordinal", it converts a character to its integer ASCII value.
        if (cv2.waitKey(1) & 0xFF) == ord('q'):
            break

    # Clean up when the loop is exited.
    # env.close() closes the environment.
    env.close()
    # cv2.destroyAllWindows() closes the OpenCV window.
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()