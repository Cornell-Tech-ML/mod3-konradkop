"""
Be sure you have minitorch installed in you Virtual Env.
>>> pip install -Ue .
"""

import random
import numpy as np
import minitorch


class Network(minitorch.Module):
    def __init__(self, hidden_layers):
        super().__init__()
        self.layer1 = Linear(2, hidden_layers)
        self.layer2 = Linear(hidden_layers, hidden_layers)
        self.layer3 = Linear(hidden_layers, 1)

    def forward(self, x):
        middle = [h.relu() for h in self.layer1.forward(x)]
        end = [h.relu() for h in self.layer2.forward(middle)]
        return self.layer3.forward(end)[0].sigmoid()

class Linear(minitorch.Module):
    def __init__(self, in_size, out_size):
        super().__init__()
        self.weights = []
        self.bias = []

        xavier_weights = Linear.get_xavier_weights(in_size, out_size)

        for i in range(in_size):
            self.weights.append([])
            for j in range(out_size):
                self.weights[i].append(
                    self.add_parameter(
                        f"weight_{i}_{j}", minitorch.Scalar(xavier_weights[i * out_size + j])
                    )
                )
        for j in range(out_size):
            self.bias.append(
                self.add_parameter(
                    f"bias_{j}", minitorch.Scalar(2 * (random.random() - 0.5))
                )
            )

    @staticmethod
    # This get_xavier_weights came from Zhuohao Yang's post on ED discussion
    def get_xavier_weights(fan_in: int, fan_out: int):
        n = fan_in * fan_out
        random_weights = np.random.uniform(low=-1.0, high=1.0, size=n)

        # Adjust the mean to be exactly 0
        actual_mean = np.mean(random_weights)
        xavier_weights = random_weights - actual_mean

        # Calculate desired variance
        desired_variance = 2/ (fan_in + fan_out)

        # Adjust the variance to be the desired variance
        actual_variance = np.var(xavier_weights)
        scaling_factor = np.sqrt(desired_variance / actual_variance)
        xavier_weights = xavier_weights * scaling_factor

        return xavier_weights

    def forward(self, inputs):
        """
        Compute the output of the linear layer given the input.

        The linear layer performs the operation:
        y = Wx + b,
        where:
        - W is the weight matrix,
        - x is the input vector, and
        - b is the bias vector.

        Args:
            inputs (list of minitorch.Scalar): A list of input values for the layer.
                                                The length should match the number of weights (fan-in).

        Returns:
            list of minitorch.Scalar: The output values after applying the linear transformation.
        """

        # Initialize the output vector with bias values
        output = [b.value for b in self.bias]
        # Iterate over each input value
        for node in range(len(inputs)):
            # Iterate over each output unit
            for z in range(len(output)):
                # Update the output by adding the weighted input
                output[z] = output[z] + inputs[node] * self.weights[node][z].value

        return output



def default_log_fn(epoch, total_loss, correct, losses):
    print("Epoch ", epoch, " loss ", total_loss, "correct", correct)


class ScalarTrain:
    def __init__(self, hidden_layers):
        self.hidden_layers = hidden_layers
        self.model = Network(self.hidden_layers)

    def run_one(self, x):
        return self.model.forward(
            (minitorch.Scalar(x[0], name="x_1"), minitorch.Scalar(x[1], name="x_2"))
        )

    def train(self, data, learning_rate, max_epochs=500, log_fn=default_log_fn):
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.model = Network(self.hidden_layers)
        optim = minitorch.SGD(self.model.parameters(), learning_rate)

        losses = []
        for epoch in range(1, self.max_epochs + 1):
            total_loss = 0.0
            correct = 0
            optim.zero_grad()

            # Forward
            loss = 0
            for i in range(data.N):
                x_1, x_2 = data.X[i]
                y = data.y[i]
                x_1 = minitorch.Scalar(x_1)
                x_2 = minitorch.Scalar(x_2)
                out = self.model.forward((x_1, x_2))

                if y == 1:
                    prob = out
                    correct += 1 if out.data > 0.5 else 0
                else:
                    prob = -out + 1.0
                    correct += 1 if out.data < 0.5 else 0
                loss = -prob.log()
                (loss / data.N).backward()
                total_loss += loss.data

            losses.append(total_loss)

            # Update
            optim.step()

            # Logging
            if epoch % 10 == 0 or epoch == max_epochs:
                log_fn(epoch, total_loss, correct, losses)


if __name__ == "__main__":
    PTS = 50
    HIDDEN = 2
    DATASET = minitorch.datasets["Simple"](PTS)
    RATE = 0.5
    # ScalarTrain(HIDDEN).train(data, RATE)

    # Simple
    # PTS = 50
    # HIDDEN = 2
    # RATE = 0.1

    # Diag
    # PTS = 50
    # HIDDEN = 2
    # RATE = 1

    # Split
    # PTS = 50
    # HIDDEN = 3
    # RATE = 1

    # XOR
    # PTS = 50
    # HIDDEN = 10
    # RATE = 0.5

    # konradChangedThisFile