# Project Report: Collaboration and Competition
## Introduction
This project involves working with the Tennis environment, where two agents control rackets to bounce a ball over a net. The goal is to keep the ball in play, with rewards given for successful hits and penalties for letting the ball hit the ground or hitting it out of bounds. The environment is considered solved when the agents achieve an average score of +0.5 over 100 consecutive episodes, taking the maximum score between the two agents for each episode.

## Learning Algorithm
The algorithm implemented for this task is the Multi-Agent Deep Deterministic Policy Gradient (MADDPG). This algorithm uses a combination of actor-critic methods to learn policies in continuous action spaces. Each agent has its own actor and critic networks, which are updated based on experience tuples sampled from a replay buffer.

## Hyperparameters
- Buffer Size: 100,000
- Batch Size: 128
- Discount Factor (Gamma): 0.99
- Soft Update Factor (Tau): 0.001
- Learning Rate (Actor): 0.0001
- Learning Rate (Critic): 0.0001

## Model Architecture
### Actor Network
- Input Layer: State size (8 dimensions)
- Hidden Layer 1: 512 units, ReLU activation
- Hidden Layer 2: 256 units, ReLU activation
- Output Layer: Action size (2 dimensions), Tanh activation
### Critic Network
- Input Layer: State size (8 dimensions)
- Hidden Layer 1: 512 units, ReLU activation
- Hidden Layer 2: 256 units (state + action dimensions), ReLU activation
- Output Layer: 1 unit (Q-value)

## Implementation
The training process involved running episodes where the agents interact with the environment, collect experiences, and update their networks. The agents' actions are determined by their current policy (actor network), and the critics evaluate the actions by providing Q-values. The agents explore the action space using Ornstein-Uhlenbeck noise to encourage exploration.

### Training Loop
The training loop runs for a maximum of 2500 episodes, updating the policy networks based on experiences sampled from the replay buffer. The agents' performance is monitored, and training stops once the average score over 100 episodes reaches at least +0.5.

## Performance
The environment was solved in 1739 episodes, with an average score of 0.51 over 100 consecutive episodes.

![Training Progress](https://github.com/quockhanh0212/udacity-deep-reinforcement-learning/blob/main/p3_collab_compet/scores.png)

## Future Improvements
To further improve the agent's performance, the following strategies could be explored:

- Parameter Tuning: Adjusting hyperparameters such as learning rates, batch sizes, and exploration noise parameters.
- Network Architecture: Experimenting with deeper or more complex network architectures for the actor and critic networks.
- Advanced Exploration Techniques: Implementing advanced exploration strategies like parameter space noise or intrinsic motivation methods.
- Multi-Agent Coordination: Enhancing coordination between agents by sharing information or using centralized training with decentralized execution.
- Transfer Learning: Utilizing pre-trained models or transfer learning techniques to bootstrap the learning process.

## Conclusion
The MADDPG algorithm successfully solved the Tennis environment, demonstrating the effectiveness of actor-critic methods in multi-agent reinforcement learning scenarios. The agents achieved an average score of +0.5 within 1739 episodes, showcasing the potential of this approach for complex tasks requiring collaboration and competition.