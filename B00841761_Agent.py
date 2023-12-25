from B00841761_Model import TD_agent
from B00841761_Model import normalize_board
import pickle
import numpy as np
from backgammon import backgammon


def train_agent(n_games, track_agent_performance=False):
    train_features = []
    train_targets = []
    agent_performance = []

    for i in range(n_games):
        print("Training game", i)
        game = backgammon()
        game_moves_count = 0
        game_score = []

        while game.get_winner() is None:
            moves = game.get_moves()
            game_moves_count += len(moves)

            for move in moves:
                train_features.append(normalize_board(move))

            chosen_move = np.argmax(np.random.uniform(size=len(moves)))
            score = np.random.uniform()
            game_score.append(score)
            game.score_move(moves[chosen_move], score)

        if game.get_winner() == "WHITE":
            train_targets.extend([[1]] * game_moves_count)
            agent_wins = 1
        else:
            train_targets.extend([[0]] * game_moves_count)
            agent_wins = 0

        if track_agent_performance:
            agent_performance.append({"game": i, "agent_wins": agent_wins, "average_score": np.mean(game_score)})

    train_features = np.array(train_features)
    train_targets = np.array(train_targets)
    agent = TD_agent(learning_rate=0.01, gamma=0.5, hidden_neurons=20)  # Adjust hyperparameters
    agent.train(train_features, train_targets)

    if track_agent_performance:
        return agent, agent_performance
    else:
        return agent


def play_and_evaluate(agent):
    print("New game")
    new_game = backgammon()

    while new_game.get_winner() is None:
        moves = new_game.get_moves()
        features = np.zeros((len(moves), 16))

        for i, move in enumerate(moves):
            features[i] = normalize_board(move)

        evaluations = agent.evaluate(features)
        new_game.score_move(moves[np.argmax(evaluations)], np.max(evaluations))

    print(f"Winner: {new_game.get_winner()}")


try:
    agent, agent_performance = train_agent(n_games=1000, track_agent_performance=True)
    print("Agent Performance during training:")
    for perf in agent_performance[-10:]:
        print(perf)

    play_and_evaluate(agent)

    # Dump the agent object into a pickled file
    with open('B00841761_Model.pkl', 'wb') as file:
        pickle.dump(agent, file)

    print("Model saved successfully as B00841761_Model.pkl")

except Exception as e:
    print("An error occurred while running the code:")
    print(str(e))