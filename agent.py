
import random
from collections import namedtuple, deque
from typing import Tuple
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional

GAMMA = 0.99            # Discount factor for rewards
TAU = 1e-3              # Soft update of target parameters
BATCH_SIZE = 64         # Batch size
BUFFER_SIZE = 100000    # Experience buffer size
LEARNING_RATE = 1e-3    # Learning rate
UPDATE_EVERY = 5        # Frequency of updating the network

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

ExperienceTuple = namedtuple('ExperienceTuple', field_names=['state', 'action', 'reward', 'next_state', 'done'])


class ExperienceBuffer:
    """Fixed-size queue used to store experience tuples."""
    def __init__(self, action_size: int, buffer_size: int, batch_size: int) -> None:
        """Initialize a ExperienceBuffer object.
        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size

    def append_experience(self, state: int, action: int, reward: float, next_state: int, done: bool) -> None:
        """Add a new experience tuple to queue."""
        self.memory.append(ExperienceTuple(state, action, reward, next_state, done))

    def get_experiences(self):
        """Sample a collection of not None experiences from experience_buffer."""
        experiences = [e for e in random.sample(self.memory, k=self.batch_size) if e is not None]

        states = torch.from_numpy(np.vstack([e.state for e in experiences])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences]).astype(np.uint8)).float().to(device)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        """Return the current number of samples within the experience_buffer."""
        return len(self.memory)


class Agent:
    def __init__(self, dqnetwork: torch.nn.Module, state_size: int, action_size: int, update_type: str, seed: int):
        """Initialize an Agent object.
        Params
        ======
            qnetwork (torch.nn.Module): model to use as the function approximation
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            update_type (str): 'dqn' or 'double-dqn'
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.update_type = update_type

        # DQ-Network
        self.qnetwork_local = dqnetwork(state_size, action_size, seed).to(device)
        self.qnetwork_target = dqnetwork(state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LEARNING_RATE)

        # Experience Buffer
        self.experience_buffer = ExperienceBuffer(action_size, BUFFER_SIZE, BATCH_SIZE)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

    def step(self, state, action, reward, next_state, done):
        # Save experience in experience buffer
        self.experience_buffer.append_experience(state, action, reward, next_state, done)

        # Learning is done only every UPDATE_EVERY steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # Check whether there are enough sample available within the experience buffer.
            if len(self.experience_buffer) > BATCH_SIZE:
                # Get the experiences and update the value parameters.
                experiences = self.experience_buffer.get_experiences()
                self.update_value_parameters(experiences, GAMMA)

    def get_action(self, state, eps: float = 0.):
        """Returns actions for given state as per current policy.
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            # Get best action
            return np.argmax(action_values.cpu().data.numpy())
        else:
            # Explore new actions by selecting random action
            return random.choice(np.arange(self.action_size))

    def update_value_parameters(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.
        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        q_targets_next = None
        # Get max predicted Q values (for next states) from target model
        # In case of double dqn, this
        if self.update_type == 'dqn':
            q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)

        elif self.update_type == 'double_dqn':
            best_local_actions = self.qnetwork_local(states).max(1)[1].unsqueeze(1)
            double_dqn_targets = self.qnetwork_target(next_states)
            q_targets_next = torch.gather(double_dqn_targets, 1, best_local_actions)

        # Compute Q targets for current states
        q_targets = rewards + (gamma * q_targets_next * (1 - dones))

        # Get expected Q values from local model
        q_expected = self.qnetwork_local(states).gather(1, actions)

        # Compute the loss using mse between expected and current q values.
        loss = torch.nn.functional.mse_loss(q_expected, q_targets)

        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update the target model
        self.update_model_parameters(self.qnetwork_local, self.qnetwork_target, TAU)

    @staticmethod
    def update_model_parameters(local_model, target_model, tau: float) -> None:
        """
           Update the model parameters according to this formula:
           θ_target = τ*θ_local + (1 - τ)*θ_target

           Args:
               local_model (PyTorch model): weights will be copied from
               target_model (PyTorch model): weights will be copied to
               tau (float): interpolation parameter
        """

        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)


if __name__ == "__main__":
    eb = ExperienceBuffer(1, 100, 5)
    for i in range(10):
        eb.append_experience(random.randint(1, 10), random.randint(1, 5), 1.0 * random.randint(1, 100), random.randint(1, 10), False)

    states, actions, rewards, next_states, dones = eb.get_experiences()
    print(states)
