# ChokoPPO: Deep Reinforcement Learning for Strategic Board Game Play

A reinforcement learning system that trains autonomous agents to play Choko, a two-player abstract strategy game, using Proximal Policy Optimization (PPO) and Double Deep Q-Networks (DDQN). This project was developed as part of CS 224R: Deep Reinforcement Learning at Stanford University.

## Executive Summary

This project demonstrates the application of modern deep reinforcement learning techniques to master Choko, a complex two-player board game with a 2,625-dimensional discrete action space. Key contributions include:

- **Custom Gym Environment:** Complete implementation of Choko game mechanics with action masking and observation symmetries
- **Population-Based Self-Play:** Novel training approach maintaining 20 historical agent snapshots to prevent policy collapse
- **Dual Algorithm Comparison:** Parallel implementations of PPO and Double DQN enabling empirical performance analysis
- **Strong Empirical Results:** Trained agents achieve 70% win rate against depth-3 minimax search baseline

The project encompasses 2,000+ lines of production-quality Python code including environment design, agent architectures, training infrastructure, and evaluation pipelines.

## Table of Contents

- [Game Description](#game-description)
- [Technical Overview](#technical-overview)
- [Installation and Setup](#installation-and-setup)
- [Training](#training)
- [Interactive Play](#interactive-play)
- [Project Structure](#project-structure)
- [Architecture and Design Decisions](#architecture-and-design-decisions)
- [Hyperparameter Configuration](#hyperparameter-configuration)
- [Experimental Results](#experimental-results)
- [Limitations and Considerations](#limitations-and-considerations)
- [Future Research Directions](#future-research-directions)
- [References](#references)

## Game Description

Choko is a two-player zero-sum perfect information game played on a 5×5 board. Each player controls 12 pieces with the objective of capturing all opponent pieces. The game features two distinct phases that create a rich strategic landscape:

### Placement Phase
Players alternate placing pieces on empty board positions. The first player to place a piece acquires "drop initiative," which requires them to place all remaining pieces before transitioning to movement actions. This mechanism introduces strategic depth regarding timing and positioning.

### Movement Phase
After completing piece placement (or when drop initiative allows), players may execute one of two move types per turn:

- **Standard Move:** Translate a piece one square in any cardinal direction (up, down, left, right) to an adjacent empty square.

- **Jump Move:** Leap over an adjacent opponent piece to land on an empty square two spaces away, capturing both the jumped piece and one additional opponent piece anywhere on the board.

### Terminal Conditions
The game terminates when either player captures all 12 opponent pieces (win condition) or after 200 moves elapse (draw condition).

## Technical Overview

This project implements a complete reinforcement learning pipeline including a custom Gym-compatible environment, two distinct RL algorithms, and a comprehensive training infrastructure with self-play capabilities.

### PPO Implementation

The PPO agent employs an actor-critic architecture with the following key features:

- **Self-play training** with a population-based approach maintaining 20 frozen agent checkpoints
- **Generalized Advantage Estimation (GAE)** with λ=0.95 for variance reduction
- **Action masking** to efficiently handle the large discrete action space
- **Observation symmetry** leveraging board state transformations for improved sample efficiency

### Double DQN Implementation

The Q-learning agent implements Double DQN with several enhancements:

- **Epsilon-greedy exploration** (ε=0.1) for balanced exploration-exploitation
- **Target network** updated every 100 training steps for stability
- **n-step returns** (n=5) for improved temporal credit assignment
- **Experience replay buffer** with capacity of 20,000 transitions

### Action Space Complexity

The environment presents a combinatorially large action space comprising 2,625 discrete actions:
- 25 placement actions (one per board square)
- 100 standard move actions (25 pieces × 4 directions)
- 2,500 jump actions (25 pieces × 4 directions × 25 capture targets)

Action masking is employed to restrict the policy to valid actions at each state, significantly improving learning efficiency.

## Installation and Setup

The project uses Conda for dependency management. To set up the environment:

```bash
conda env create -f environment.yml
conda activate choko-ppo
```

Alternatively, dependencies can be installed via pip:
```bash
pip install torch numpy gym stable-baselines3 tensorboard tqdm
```

**Hardware Requirements:** The implementation supports CPU, CUDA, and Apple Metal Performance Shaders (MPS). The system automatically detects and utilizes available accelerators. Training on GPU/MPS is recommended for significantly reduced training time.

## Training

### PPO Training

To train a PPO agent:

```bash
python scripts/train.py
```

Training configuration:
- **Iterations:** 8,000 training iterations
- **Samples per iteration:** 8,192 environment transitions
- **Checkpoint frequency:** Every 250 iterations (saved to `checkpoints/ppo/run_13/`)
- **Evaluation frequency:** Every 10 iterations against frozen policy snapshots
- **Logging:** TensorBoard metrics logged to `logs/ppo/run_13/`

Monitor training progress via TensorBoard:
```bash
tensorboard --logdir logs/ppo/
```

### Double DQN Training

To train a Q-learning agent:

```bash
python scripts/train.py --use_q_learning
```

Training configuration:
- **Iterations:** 200,000 training steps
- **Exploration:** Epsilon-greedy with ε=0.1
- **Target network update frequency:** Every 100 steps
- **Evaluation frequency:** Every 200 iterations
- **Checkpoint frequency:** Every 1,000 iterations (saved to `checkpoints/q_learning/run_3/`)

## Interactive Play

To evaluate trained agents through human gameplay:

```bash
python scripts/play.py
```

**Note:** The checkpoint path must be configured in `play.py` to load the desired trained agent.

### Input Format

The interface accepts moves in the following format:

```
place 2 3        # Placement action: place piece at position (row=2, col=3)
move 1 2 right   # Standard move: move piece at (1,2) one square right
jump 2 2 up 0 4  # Jump move: jump from (2,2) upward, additionally capture piece at (0,4)
```

### Board Representation

The game state is rendered using the following notation:
- `.` — Empty square
- `X` — Player 1 pieces
- `O` — Player 2 pieces

## Project Structure

```
ChokoPPO/
├── agents/
│   ├── ppo_agent.py        # Actor-critic PPO implementation
│   ├── q_agent.py          # Double DQN implementation
│   ├── minimax_agent.py    # Minimax baseline (depth-3 search)
│   ├── minimax_ppo.py      # Hybrid approach
│   └── random_agent.py     # Random baseline
├── envs/
│   └── choko_env.py        # Gym environment for Choko
├── infrastructure/
│   ├── replay_buffer.py    # Replay buffers for both PPO and Q-learning
│   ├── utils.py            # Helper functions (GAE, observation flipping)
│   └── board_editor.py     # Tool for setting up board positions
├── scripts/
│   ├── train.py            # Main training loop
│   ├── play.py             # Human vs agent interface
│   └── experiments.py      # Experiment configurations
└── config.py               # Hyperparameters
```

## Architecture and Design Decisions

### State Representation
The board state is encoded as a 26-dimensional vector comprising 25 board positions (flattened 5×5 grid) plus one additional feature for drop initiative status. A critical optimization applies observation symmetry: the state representation for player 2 is transformed such that both players perceive themselves as player 1. This symmetry exploitation significantly improves sample efficiency by effectively doubling training data.

### Action Masking
Invalid actions are masked via logit manipulation, setting invalid action logits to -10^10 prior to softmax normalization. This technique is essential given that the majority of actions are invalid in any given state, preventing the policy from wasting probability mass on impossible moves.

### Population-Based Self-Play
Rather than training exclusively against the current policy, the system maintains a pool of 20 frozen agent checkpoints sampled uniformly across training. This approach mitigates cyclical behavior and policy collapse, promoting robust strategy development against diverse opponents.

### Reward Engineering
The reward structure employs sparse terminal rewards (±2 for win/loss) augmented with shaped intermediate rewards (+0.1 per piece captured). The capture reward undergoes linear decay over training to gradually shift the agent's objective from exploration to exploitation and efficient winning strategies.

### Advantage Estimation
Generalized Advantage Estimation with λ=0.95 provides an effective bias-variance trade-off for this domain, balancing between Monte Carlo (high variance, low bias) and TD learning (low variance, high bias).

## Hyperparameter Configuration

All hyperparameters are centralized in `config.py`:

### PPO Hyperparameters
- **Learning rate:** 1e-4
- **Clip range (ε):** 0.2
- **Batch size:** 64
- **Rollout length:** 8,192 transitions
- **Training epochs:** 6 per iteration
- **Value loss coefficient (c₁):** 0.5
- **Entropy coefficient (c₂):** 0.01
- **GAE lambda (λ):** 0.95
- **Discount factor (γ):** 0.99

### Double DQN Hyperparameters
- **Learning rate:** 1e-3
- **Exploration rate (ε):** 0.1
- **n-step returns:** 5
- **Replay buffer capacity:** 20,000 transitions
- **Target network update frequency:** 100 steps
- **Discount factor (γ):** 0.99

These hyperparameters were determined through systematic empirical evaluation, with lower learning rates and larger batch sizes demonstrating improved training stability.

## Experimental Results

### Training Performance

The PPO agent demonstrates strong strategic play after approximately 3,000 training iterations (≈24M environment transitions). Learned behaviors include:

- **Spatial control:** Prioritizing central board positions during placement phase
- **Tactical planning:** Constructing multi-step jump sequences for efficient captures
- **Initiative management:** Strategic exploitation of drop initiative mechanics
- **Material evaluation:** Executing favorable piece trades based on positional advantage

### Benchmark Evaluation

Against a minimax search baseline with depth limit of 3:
- **PPO win rate:** 70%
- **Q-learning win rate:** Comparable performance achieved with extended training

The PPO agent exhibits faster convergence relative to Double DQN, likely attributable to its stochastic policy parameterization enabling more effective exploration in the large action space.

## Limitations and Considerations

### Current Limitations
- **Draw conditions:** Defensive play patterns occasionally result in drawn games at the 200-move limit
- **Environment flexibility:** Board dimensions are fixed at 5×5 (not parameterized)
- **Computational requirements:** CPU-only training requires approximately 10 hours for 3,000 PPO iterations
- **Code artifacts:** Legacy implementations (`choko_env_old.py`, `PPOAgentOld`) retained for backward compatibility

## Future Research Directions

Potential extensions to this work include:

1. **AlphaZero-style training:** Integration of Monte Carlo Tree Search with learned value and policy networks for enhanced planning
2. **Curriculum learning:** Progressive training on smaller board configurations before scaling to 5×5
3. **Architectural improvements:** Attention mechanisms over piece positions to better capture spatial relationships
4. **Parallelization:** Distributed environment rollouts for improved computational efficiency
5. **Population management:** Tournament-based selection for frozen agent pool rather than uniform sampling

## References

This implementation draws upon game mechanics and strategic analysis from the ChokoZero paper (included in repository). While all code was developed independently, the paper provided foundational insights into game dynamics and rule specifications.

### Key Algorithms
- **Proximal Policy Optimization:** Schulman et al. (2017) - "Proximal Policy Optimization Algorithms"
- **Double Q-Learning:** van Hasselt et al. (2015) - "Deep Reinforcement Learning with Double Q-learning"
- **Generalized Advantage Estimation:** Schulman et al. (2015) - "High-Dimensional Continuous Control Using Generalized Advantage Estimation"

---

**Project Information**  
Developed for CS 224R: Deep Reinforcement Learning, Stanford University  


