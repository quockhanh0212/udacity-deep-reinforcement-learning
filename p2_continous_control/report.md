# Report on Continuous Control Using DDPG
## Introduction
This report outlines the implementation of the Deep Deterministic Policy Gradient (DDPG) algorithm to solve the Reacher environment, which involves controlling a double-jointed arm to move towards target locations. The goal of the agent is to maintain its position at the target location for as many time steps as possible, receiving a reward of +0.1 for each step the agent's hand is in the goal location.

## Implementation Details
### Learning Algorithm: DDPG
DDPG is an actor-critic, model-free algorithm based on the deterministic policy gradient that can operate over continuous action spaces. It uses four neural networks: two for the actor and two for the critic, each with its local and target networks. The actor network selects actions, while the critic network evaluates them. The target networks provide stability to the learning process by slowly tracking the learned networks.

### Hyperparameters
The hyperparameters chosen for this implementation are as follows:

- Replay buffer size: 1e6
- Batch size: 1024
- Discount factor (γ): 0.99
- Soft update parameter (τ): 1e-3
- Learning rate (Actor): 1e-4
- Learning rate (Critic): 3e-4
- Weight decay: 1e-4
- Noise parameters: θ = 0.15, σ = 0.2
### Model Architectures
#### Actor Network
The actor network is designed to map states to actions. It consists of three fully connected layers with the following architecture:

- Input layer: size = state size (33)
- Hidden layer 1: size = 256, activation = Leaky ReLU
- Hidden layer 2: size = 128, activation = Leaky ReLU
- Output layer: size = action size (4), activation = Tanh
- Batch normalization is applied to the input layer to normalize the state inputs.

#### Critic Network
The critic network evaluates the state-action pairs. It consists of four fully connected layers with the following architecture:

- Input layer: size = state size (33)
- Hidden layer 1: size = 256, activation = Leaky ReLU
- Hidden layer 2: size = 256 + action size (260), activation = Leaky ReLU
- Hidden layer 3: size = 128, activation = Leaky ReLU
- Output layer: size = 1
Batch normalization is also applied to the input layer.

## Training Process
The agent interacts with the environment, storing experiences in a replay buffer. Periodically, the agent samples a batch of experiences from the replay buffer to update the actor and critic networks. The critic is updated to minimize the mean squared error between the predicted Q-values and the target Q-values. The actor is updated using the policy gradient, which maximizes the expected return. Soft updates are applied to the target networks to ensure smooth updates.

The training process continues until the agent achieves an average score of +30 over 100 consecutive episodes.

## Results
The agent was able to solve the environment in 243 episodes, achieving an average score of +30.04 over the last 100 episodes. The plot below shows the rewards per episode, illustrating the learning progress of the agent.
![Trained Agent](https://github.com/quockhanh0212/udacity-deep-reinforcement-learning/blob/main/p2_continous_control/scores_plot.png)


## Future Improvements
Several improvements can be made to enhance the agent's performance:

- Hyperparameter Tuning: Experimenting with different values for the learning rates, batch size, and noise parameters can help improve stability and convergence speed.
- Network Architecture: Modifying the architecture of the actor and critic networks, such as adding more layers or using different activation functions, could enhance the learning capacity.
- Prioritized Experience Replay: Implementing a prioritized replay buffer where experiences are sampled based on their importance can lead to more efficient learning.
- Multi-Agent Training: Exploring training with multiple agents in parallel can help gather diverse experiences and improve the robustness of the policy.
- Exploration Strategies: Investigating alternative exploration strategies, such as parameter noise, can lead to better exploration of the action space.
By incorporating these improvements, the agent's performance can be further enhanced, leading to faster convergence and higher rewards.

