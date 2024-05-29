# Implementation Report for Deep Q-Learning Agent

## Introduction

In this project, we train an agent to navigate and collect bananas in a large, square world using deep reinforcement learning techniques. The agent receives a reward of +1 for collecting a yellow banana and a penalty of -1 for collecting a blue banana. The goal is to maximize the collection of yellow bananas while avoiding blue ones.

## Environment Description

The environment is characterized by a state space of 37 dimensions, which includes the agent's velocity and ray-based perception of objects around its forward direction. The action space consists of four discrete actions:
- `0` - move forward
- `1` - move backward
- `2` - turn left
- `3` - turn right

The task is episodic, and the environment is considered solved when the agent achieves an average score of +13 over 100 consecutive episodes.

## Learning Algorithm

The agent is trained using a Deep Q-Network (DQN) algorithm

### Q-Network Architecture

The Q-Network is a neural network used as a function approximator. The architecture consists of:
- An input layer of size 37 (state size)
- Two hidden layers with 64 units each
- An output layer of size 4 (action size)

The network uses ReLU activation functions for the hidden layers.

```python
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, seed):
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)
```
## Hyperparameters
The following hyperparameters were used for training the agent:

- Replay buffer size: 100,000
- Minibatch size: 64
- Discount factor (gamma): 0.99
- Soft update factor (tau): 0.001
- Learning rate (alpha): 0.0005
- Update frequency: 4 steps

## Training Process
The agent's experience is stored in a replay buffer. During training, batches of experiences are sampled from the buffer to update the Q-Network. The target network is updated periodically to stabilize training.

### Training Results
The agent was able to solve the environment in 543 episodes. The plot below shows the reward per episode, illustrating the learning progress.

### Rewards per Episode
Plot showing the score per episode over all the episodes. The environment was solved in **543** episodes (currently).

![Results](https://github.com/quockhanh0212/udacity-deep-reinforcement-learning/blob/main/p1_navigation/results/dqn_scores.png)

## Future Improvements
To further improve the agent's performance, the following strategies can be considered:

- Prioritized Experience Replay: Prioritizing important transitions to be replayed more frequently.
- Rainbow DQN: Integrating multiple DQN enhancements, including prioritized experience replay, dueling networks, and multi-step learning.
- Parameter Noise: Adding noise to network parameters to encourage exploration.
- Curiosity-driven Exploration: Implementing intrinsic motivation to explore less-visited states.

## Conclusion
This report outlines the implementation of a DQN agent for navigating a simulated environment to collect bananas. The agent successfully learns to achieve an average reward of +13 over 500 episodes. Future improvements can be made by incorporating advanced techniques and further tuning the hyperparameters.