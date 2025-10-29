from matplotlib import pyplot as plt
import numpy as np
import rooms
from REINFORCE_sol import ReinforceAgent
np.random.seed(123)


def episode(env, agent, discount_return_factor=1):
    rewards = []
    states = []
    actions = []
    time_step = 0
    discounted_return = 0
    done = False
    state = env.reset()
    while not done:
        states.append(state)
        action = agent.choose_action(state)
        next_state, reward, done, _ = env.step(action)
        actions.append(action)
        rewards.append(reward)
        state = next_state
        discounted_return += reward * (discount_return_factor ** time_step)
        time_step +=1
    return states, actions, rewards, discounted_return


def train(env, agent, nr_episodes, discount_return_factor=1, plot=True):
    returns = []
    alpha_start = agent.alpha
    test_returns = []
    for i in range(nr_episodes):
        states, actions, rewards, episode_return = episode(env, agent, discount_return_factor)
        returns.append(episode_return)
        agent.update_montecarlo(states, actions, rewards)
        agent.alpha = agent.alpha - alpha_start / nr_episodes
        if i % (nr_episodes // 50) == 0:
            print("episode {:5d}, alpha {:0.6f}, return {}".format(i, agent.alpha, episode_return))
            test_returns.append(sum([episode(env, agent, discount_return_factor)[3] for _ in range(10)]) / 10.)

    if plot:
        fig, (ax1, ax2) = plt.subplots(2, 1)
        ax1.plot(returns)
        ax1.set_title("Training returns")
        ax1.set_xlabel('episode')
        ax2.plot(test_returns)
        ax2.set_title("Test average returns")
        ax2.set_xlabel('episode')
        plt.show()

    return returns, test_returns


if __name__ == '__main__':
    # If you get an error "imageio.ffmpeg.download() has been deprecated",
    # run: pip install imageio==2.4.1 in your terminal and try again

    env = rooms.load_env("rooms_9_9_4.txt", "rooms.mp4")

    # Set the parameters and train the agent
    alpha = 0.01
    gamma = 0.99
    discount_rf = 0.99  # corresponds to cumulative rewards
    nr_episodes = 2000  # > 20!

    agent = ReinforceAgent(env, alpha, gamma)
    train(env, agent, nr_episodes, discount_rf, plot=True)
    # env.save_video()
