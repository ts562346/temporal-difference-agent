import numpy as np

def normalize_board(board):
    max_checker_count = 15
    max_off_count = 15
    normalized_board = (
        [checker_count / max_checker_count for checker_count in board[:12]]
        + [board[12] / max_off_count, board[13] / max_off_count]
        + [1 if board[14] == 1 else -1]
        + [1]
    )
    return normalized_board

def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def sigmoid_derivative(z):
    return z * (1 - z)


def train_mlp(inputs, targets, hidden_neurons, epochs, learning_rate, gamma):
    input_neurons, output_neurons = inputs.shape[1], targets.shape[1]
    weights_input_hidden = np.random.uniform(size=(input_neurons, hidden_neurons))
    weights_hidden_output = np.random.uniform(size=(hidden_neurons, output_neurons))

    # Initialize eligibility traces
    eligibility_traces_input_hidden = np.zeros_like(weights_input_hidden)
    eligibility_traces_hidden_output = np.zeros_like(weights_hidden_output)

    # Train the network
    for epoch in range(epochs):
        for i in range(inputs.shape[0]):
            input_vector, target = inputs[i, :], targets[i, :]

            hidden_layer_activation = sigmoid(np.dot(input_vector, weights_input_hidden))
            output_layer_activation = sigmoid(np.dot(hidden_layer_activation, weights_hidden_output))

            # Compute TD error
            error = target - output_layer_activation
            td_error = error * sigmoid_derivative(output_layer_activation)

            # Update eligibility traces
            eligibility_traces_hidden_output = gamma * eligibility_traces_hidden_output + np.outer(hidden_layer_activation, td_error)
            eligibility_traces_input_hidden = gamma * eligibility_traces_input_hidden + np.outer(input_vector, hidden_layer_activation * np.dot(td_error, weights_hidden_output.T) * sigmoid_derivative(hidden_layer_activation))

            # Update weights using the eligibility traces and TD error
            weights_hidden_output += learning_rate * eligibility_traces_hidden_output
            weights_input_hidden += learning_rate * eligibility_traces_input_hidden

    return weights_input_hidden, weights_hidden_output

def predict(input_data, weights_input_hidden, weights_hidden_output):
    hidden_layer_activation = sigmoid(np.dot(input_data, weights_input_hidden))
    output_layer_activation = sigmoid(np.dot(hidden_layer_activation, weights_hidden_output))
    return output_layer_activation