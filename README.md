# Temporal Difference Agent for Backgammon Disengaged Bearoff Game

## Introduction

This project involves implementing a Temporal Difference (TD) learning agent for the game of backgammon, specifically focusing on the disengaged bearoff configuration. The TD agent will replace the existing random decision maker in the game environment.

## Environment

- The environment represents the disengaged bearoff configuration in backgammon.
- The state space is characterized by a vector of 16 features that describe the board state.
- The agent's task is to provide a single scalar value (evaluation function) for each feature vector, indicating the likelihood of winning for white.
- The game engine rolls the dice, presents legal board configurations to the agent, and the agent ranks and selects the best move.

## Temporal Difference Formulation

The TD agent uses the following temporal difference formulation:

\[
\tilde{w}_{t+1} = \tilde{w}_t + \alpha \left(V(\tilde{s}_{t+1}, \tilde{w}_t) - V(\tilde{s}_t, \tilde{w}_t)\right) \tilde{z}_t
\]

where
\[
\tilde{z}_t = \gamma \tilde{z}_{t-1} + \nabla V_{\tilde{w}}(\tilde{s}_t, \tilde{w}_t)
\]

- \(V(\cdot, \cdot)\) is the predicted value of being in a state given the neural model.
- \(\tilde{z}_t\) is the eligibility trace, initialized at \(\tilde{z}_{-1} = 0\).
- \(\alpha\) is the learning rate, and \(\gamma\) is the discount factor.

## Implementation

The provided random decision maker in the algorithm will be replaced with the TD agent at Steps 7, 12, and 27. The agent will evaluate each legal board configuration, learn from the temporal difference, and make informed moves.

## Usage

1. Set up the backgammon environment.
2. Train the TD agent by running the algorithm, replacing the random agent with the TD agent.
3. Evaluate the trained TD agent's performance in the backgammon disengaged bearoff game.

## Dependencies

- Python
- Neural network library (e.g., TensorFlow, PyTorch)
- Backgammon environment (provided)

## Contributors

- Tasneem Hoque
- Egil Carlsen

## License

This project is licensed under the [License Name] - see the [LICENSE.md](LICENSE.md) file for details.