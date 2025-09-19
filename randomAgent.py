import retro
import time

def main():
    """
    This script runs a random agent in the Super Mario Bros. environment.
    
    Its purpose is to verify that your ROM and environment are working correctly
    before we start a long training session.
    """
    
    # Creates the Super Mario Bros. environment.
    # render_mode="human" creates a window for you to see the game.
    env = retro.make(game="SuperMarioBros-Nes", state="Level1-1", render_mode="human")
    
    # Reset the environment to get the initial state.
    env.reset()
    
    # Loop forever to keep playing.
    while True:

        # env.action_space.sample() picks a random valid action.
        action = env.action_space.sample()
        
        # Apply the action to the game and get the results (returns 5 values).
        observation, reward, terminated, truncated, info = env.step(action)

        # If the episode is over (Mario dies or level ends), reset the environment.
        if terminated or truncated:
            env.reset()

        # Add a small delay to make the game watchable at a human speed. e.g. 120 FPS.
        time.sleep(1/120)
            
if __name__ == "__main__":
    main()