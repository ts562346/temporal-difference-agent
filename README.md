# Temporal Difference Agent for Backgammon Disengaged Bearoff Game

## Environment

• We will assume the case of backgammon in the disengaged bearoff configuration as our environment.
• YChallenge is to modify your multi-layer perceptron to support the temporal difference formulation for ‘error’ with ‘eligibility traces’ (~zt) or
~wt+1 = ~wt + α (V (~st+1, ~wt) − V (~st, ~wt)) ~zt (1)
where V (·, ·) is the predicted value of being in a state given a (neural) model and ~zt is defined by,
~zt = γ~zt−1 + ∇V~wt (~st, ~wt) (2) where ~zt=−1= 0 and ∇~wtV (·, ·) is the partial derivative with respect to ~wt.
• An implementation for the task environment with a random decision maker is already provided. You
will replace the random decision maker with your TD-agent (‘blue’ highlighted elements in Algorithm 1).
• The agent experiences the environment as a vector of 16 features characterizing board state. You can preprocess these features before providing to your multi-layer perceptron as you see fit.
• Your agent will provide a single scalar value over the unit interval that acts as an evaluation function
for each feature vector (1 (0) implies win (loss) for white).
• The game engine (environment) ‘rolls’ the dice and presents each legal configuration of the board
(that your ‘agent’ evaluates, Steps 7 and 12) before implementing the move that your agent ranks
the highest (Step 22).
• The environment also detects ‘end of game’ and returns a corresponding non-zero reward if white
won (Step 25).
• A random agent is provided, which you are expected to replace with your TD learning agent at Steps
7, 12 and 27.