
import random
from collections import namedtuple, deque
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional

GAMMA = 0.99            # Discount factor for rewards
TAU = 1e-3              # Soft update of target parameters
BATCH_SIZE = 64         # Batch size
BUFFER_SIZE = 100000    # Experience buffer size
LEARNING_RATE = 1e-3    # Learning rate (used by ADAM optimizer)
UPDATE_EVERY = 5        # Frequency of updating the network

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

ExperienceTuple = namedtuple('ExperienceTuple', field_names=['state', 'action', 'reward', 'next_state', 'done'])
""""A named tuple used to collect the different fields within the experience buffer"""


class ExperienceBuffer:
    """Fixed-size queue used to store experience tuples."""
    def __init__(self, action_size: int, buffer_size: int, batch_size: int) -> None:
        """
            Initialize an ExperienceBuffer object.
            Args:
                action_size (int): The dimension of each action
                buffer_size (int): The maximum size (number of tuples) of the buffer
                batch_size (int): The size of each training batch
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size

    def append_experience(self, state: int, action: int, reward: float, next_state: int, done: bool) -> None:
        """
            Add a new experience tuple to the experience queue.
            Args:
                state (int): The current state
                action (int): The selected action
                reward (float): The received reward after selecting the action
                next_state (int): The new state reached after executing the action
                done (bool): A boolean flag indicating the episode has ended.
        """
        self.memory.append(ExperienceTuple(state, action, reward, next_state, done))

    def get_experiences(self):
        """
            Sample a collection of not None experiences from the experience_buffer.
            Returns:
                A tuple of torch tensors containing
                states, actions, rewards, next_states and dones from the experience buffer.
        """
        experiences = [e for e in random.sample(self.memory, k=self.batch_size) if e is not None]

        states = torch.from_numpy(np.vstack([e.state for e in experiences])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences]).astype(np.uint8)).float().to(device)

        return states, actions, rewards, next_states, dones

    def __len__(self) -> int:
        """
            Return the current number of samples within the experience_buffer..
            Returns:
               The current number (int) of samples within the experience buffer.
        """
        return len(self.memory)


class Agent:
    """The agent used to solve the reinforcement learning problem."""
    def __init__(self,
                 dqnetwork: torch.nn.Module,
                 state_size: int,
                 action_size: int,
                 update_type: str,
                 seed: int = 42,
                 gamma: float = GAMMA,
                 tau: float = TAU,
                 batch_size: int = BATCH_SIZE,
                 buffer_size: int = BUFFER_SIZE,
                 learning_rate: float = LEARNING_RATE,
                 update_every: int = UPDATE_EVERY
                 ):
        """
            Initialize an agent object, this includes:
                - Build an DQ network
                - Create an Experience Buffer
            Args:
                dqnetwork (torch.nn.Module): model to use as the function approximation.
                state_size (int): The dimension of the state space.
                action_size (int): The dimension of the action space.
                update_type (str): Either 'dqn' or 'double-dqn'.
                seed (int): The random seed.
                gamma (float): The discount factor for rewards.
                tau (float): The interpolation parameter controlling the amount of updating the target model.
                batch_size(int): The batch size used for learning.
                learning_rate (float): The learning rate used by the ADAM optimizer.
                update_every (int): The frequency of updating the network.
        """
        self.state_size = state_size
        self.action_size = action_size
        self.update_type = update_type
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.time_step = 0
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.learning_rate = learning_rate
        self.update_every = update_every

        # DQ-Network
        self.qnetwork_local = dqnetwork(state_size, action_size, seed).to(device)
        self.qnetwork_target = dqnetwork(state_size, action_size, seed).to(device)
        # We use an ADAM optimizer with the configured learning rate
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=self.learning_rate)

        # Experience Buffer
        self.experience_buffer = ExperienceBuffer(action_size, self.buffer_size, self.batch_size)

    def step(self, state, action, reward, next_state, done):
        """
            This method is used within a learning loop after the agent has executed
            an action against the environment.

            The experience buffer is updated and a learning step is executed every update_every steps.
            Args:
                state (int): The current state
                action (int): The selected action
                reward (float): The received reward after selecting the action
                next_state (int): The new state reached after executing the action
                done (bool): A boolean flag indicating the episode has ended.
        """
        # Save experience in experience buffer
        self.experience_buffer.append_experience(state, action, reward, next_state, done)

        # Learning is done only every update_every steps
        self.time_step = (self.time_step + 1) % self.update_every
        if self.time_step == 0:
            # Check whether there are enough samples available within the experience buffer.
            if len(self.experience_buffer) > self.batch_size:
                # Get the experiences and update the value parameters.
                self.update_value_parameters(self.experience_buffer.get_experiences(), self.gamma)

    def get_action(self, state, eps: float = 0.):
        """
            This method is used within a learning to select an action to be executed against the environment.
            Args:
                state (int): The current state
                eps (int): A parameter controlling exploration versus exploitation (epsilon-greedy action selection)
                During a typical learning loop epsilon is rapidly decreased.
        """
        # The state is moved to the device (either gpu or cpu).
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        # The network is used to compute the actions.
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection among the computed actions
        if random.random() > eps:
            # Get best action
            return np.argmax(action_values.cpu().data.numpy())
        else:
            # Explore new actions by selecting random action
            return random.choice(np.arange(self.action_size))

    def update_value_parameters(self, experiences, gamma):
        """
            This method is used within the step method to update the value parameters.
            Args:
                experiences: The tuple of torch tensors. containing states, actions, rewards, next_states and dones.
                gamma (float): The discount factor for rewards.
        """
        states, actions, rewards, next_states, dones = experiences

        # Use the target model to get Q values.
        if self.update_type == 'dqn':
            q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)

        elif self.update_type == 'double_dqn':
            best_local_actions = self.qnetwork_local(states).max(1)[1].unsqueeze(1)
            double_dqn_targets = self.qnetwork_target(next_states)
            q_targets_next = torch.gather(double_dqn_targets, 1, best_local_actions)
        else:
            raise ValueError("Unknown update type, only dqn or double_dqn are supported.")

        # Compute Q targets for current states
        q_targets = rewards + (gamma * q_targets_next * (1 - dones))

        # Get expected Q values from local model
        q_expected = self.qnetwork_local(states).gather(1, actions)

        # The goal is to minimize the mean square error between expected and actual q values.
        loss = torch.nn.functional.mse_loss(q_expected, q_targets)

        # Minimize the loss using SGD
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update the target model parameters
        self.update_model_parameters(self.qnetwork_local, self.qnetwork_target, self.tau)

    @staticmethod
    def update_model_parameters(local_model, target_model, tau: float) -> None:
        """
           Update the model parameters according to this formula:
           θ_target = τ*θ_local + (1 - τ)*θ_target

           Args:
               local_model (PyTorch model): weights will be copied from this model
               target_model (PyTorch model): weights will be copied to this model
               tau (float): interpolation parameter, tau = 1 results in complete overwrite
        """

        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)


if __name__ == "__main__":
    eb = ExperienceBuffer(1, 100, 5)
    for i in range(10):
        eb.append_experience(random.randint(1, 10), random.randint(1, 5), 1.0 * random.randint(1, 100),
                             random.randint(1, 10), False)

    states, actions, rewards, next_states, dones = eb.get_experiences()
    print(states)
