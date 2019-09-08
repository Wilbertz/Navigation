from collections import deque
import torch
import numpy as np


class DQNLearning:
    """Class used to run agent within reinforcement learning environment."""
    def __init__(self,
                 agent,
                 env,
                 brain_name,
                 n_episodes,
                 max_timesteps,
                 eps_start,
                 eps_end,
                 eps_decay,
                 weights_file,
                 target_reward=13):
        """
            Initialize the DQLLearning class.
            Args:
                agent: The agent used to solve the environment.
                env: The environment
                brain_name: The name of the currently used brain.
                n_episodes: The maximum number of episodes to run.
                max_timesteps: The maximum number of timesteps within an episode.
                eps_start: The start value of epsilon (Control exploration vs. exploitation)
                eps.end: The end value of epsilon.
                eps_decay: The decay of epsilon during each time step
                weights_file: The name of the file used to store the weights.
                target_reward: The target to be reached in order tpo solve the problem.
        """
        self.agent = agent
        self.env = env
        self.brain_name = brain_name
        self.n_episodes = n_episodes
        self.max_timesteps = max_timesteps
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.eps = eps_start
        self.score = 0
        self.scores = []  # A list containing rhe scores from all episodes
        self.scores_window = deque(maxlen=100)  # Only the last 100 scores are stored
        self.weights_file = weights_file
        self.target_reward = target_reward

    def learning(self):
        """
            Learn to solve the environment.
            Returns:
                scores (List of floats): The scores during the learning process.

        """
        self.scores = []
        self.scores_window = deque(maxlen=100)
        self.eps = self.eps_start

        for i_episode in range(1, self.n_episodes + 1):  # run n_episodes,
            # Reset the environment, use the training mode
            env_info = self.env.reset(train_mode=True)[self.brain_name]  # Reset the environment
            self.score = 0  # initialize the score
            state = env_info.vector_observations[0]  # Get the initial observation from the environment

            for t in range(self.max_timesteps):  # Run for at most of max_t timesteps
                # Get action, next state, rewards and whether episode is done from environment
                action = self.agent.get_action(state, self.eps)
                env_info = self.env.step(action)[self.brain_name]
                next_state = env_info.vector_observations[0]
                reward = env_info.rewards[0]
                done = env_info.local_done[0]

                # Execute action and train the agent.
                self.agent.step(state, action, reward, next_state, done)
                self.score += reward  # update the score

                state = next_state  # update the state
                if done:  # break if episode is complete
                    break

            self.eps = max(self.eps_end, self.eps * self.eps_decay)  # Decrease epsilon according schedule

            if self.update_and_print_scores(i_episode):
                break

        return self.scores

    def update_and_print_scores(self, index_episode: int) -> bool:
        """
            Print scores to the console. Checks whether target score has been reached.
            Args:
                index_episode (int): The current episode (1 based index).
            Returns:
                A boolean flag indicating the target reward has been reached.

        """
        self.scores_window.append(self.score)  # Update the scores queue
        self.scores.append(self.score)  # Add the current score to the list of scores
        average_score = np.mean(self.scores_window)  # Compute the average score over the window

        print('\rEpisode {} \tAverage score: {: .2f}'.format(index_episode, average_score), end="")

        if index_episode % 100 == 0:
            print('\rEpisode {} \tAverage score: {: .2f}'.format(index_episode, average_score))
            return False

        if average_score >= self.target_reward:  # Check whether the target reward has been reached.
            print('\nEnvironment solved in {: d} episodes!\tAverage Score: {: .2f}'.format(index_episode - 100,
                                                                                           average_score))
            torch.save(self.agent.qnetwork_local.state_dict(), self.weights_file)
            return True
