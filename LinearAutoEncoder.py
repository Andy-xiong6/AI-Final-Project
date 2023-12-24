import numpy as np

class linearautoencoder:
    def __init__(self, hidden_layer_sizes ,input_size, output_size):
        self.hidden_layer_sizes = hidden_layer_sizes # a list of number. the length refers the layer numbers, and the element refers the hidden units
        self.weights = []
        self.biases = []
        self.weights = [np.random.randn(input_size, output_size)]
        self.biases = [np.random.randn(1, output_size)]

    def forward_propagation(self, X):
        activations = [X]
        for i in range(len(self.weights)):
            z = activations[-1] @ self.weights[i] + self.biases[i]
            a = z # linear activation function
            activations.append(a)
        return activations

    def loss(self, X, y):
        out = self.forward_propagation(X)[-1]
        return np.mean((out - y) ** 2)
    
    def train(self, X, y, learning_rate, num_epochs):
        for epoch in range(num_epochs):
            activations = self.forward_propagation(X)
            out = activations[-1]
            loss = np.mean((out - y) ** 2)
            error = out - y

            #backward_propogation and update
            deltas = [error]
            for i in range(len(self.weights)-1, 0, -1):
                delta = (deltas[0] @ self.weights[i].T)
                deltas.insert(0, delta) # add to the first
                
            for i in range(len(self.weights)):
                dW = activations[i].T @ deltas[i]
                dB = np.sum(deltas[i], axis=0, keepdims=True)
                self.weights[i] -= learning_rate * dW
                self.biases[i] -= learning_rate * dB
    