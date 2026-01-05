## Featured Project: Soft Actor-Critic (SAC) Implementation

The primary project in this portfolio is a from-scratch implementation of the **Soft Actor-Critic (SAC)** reinforcement learning algorithm, trained on continuous-control environments using TensorFlow and Keras.

I built this project to better understand how theoretical ideas—entropy maximization, stochastic policies, and value estimation—translate into real learning systems. Rather than relying on a prebuilt library implementation, I implemented the actor, twin critics, target networks, replay buffer, and entropy tuning manually.

### Key Components
- Stochastic Gaussian policy with learned mean and log standard deviation  
- Twin Q-networks with Polyak-averaged target networks  
- Replay buffer for off-policy learning  
- Automatic entropy tuning via a learned temperature parameter  
- Training and evaluation on Gymnasium continuous-control environments  

### Why this project
This project was especially meaningful to me because it required moving fluidly between:
- mathematical reasoning (objective functions, entropy terms, Bellman backups)
- implementation details (numerical stability, randomness, optimizer behavior)
- system-level debugging (framework behavior and training dynamics)

Diagnosing why the agent initially failed to learn—despite a correct conceptual implementation—taught me how small, hidden details in real systems can dominate outcomes.

## Development Environment

Most of this work was done independently using:
- A web browser for research and references
- PyCharm as the primary development environment
- TensorFlow / Keras for model construction and training

I did not work in a formal lab or makerspace; instead, I built and tested ideas wherever circumstances allowed.

## Attribution & References

This work was inspired by and references:
- Haarnoja et al., *Soft Actor-Critic Algorithms and Applications*, 2018 (arXiv:1812.05905)
- OpenAI Spinning Up (conceptual guidance and reference structure)

All implementation decisions, architecture choices, and debugging were done independently.
