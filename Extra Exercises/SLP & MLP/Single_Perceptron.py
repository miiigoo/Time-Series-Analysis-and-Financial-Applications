import numpy as np

# Step 1: Define the data (binary images of "1" and "0")
digit_1 = [
    [0, 0, 1, 0, 0,  # Pattern 1
     0, 0, 1, 0, 0,
     0, 0, 1, 0, 0,
     0, 0, 1, 0, 0,
     0, 0, 1, 0, 0],
    #Pattern 2
    [0, 1, 1, 0, 0,
     0, 0, 1, 0, 0,
     0, 0, 1, 0, 0,
     0, 0, 1, 0, 0,
     0, 1, 1, 1, 0],
    #Pattern 3
    [0, 0, 1, 0, 0,
     0, 1, 1, 0, 0,
     0, 0, 1, 0, 0,
     0, 0, 1, 0, 0,
     0, 0, 1, 0, 0],
    #Pattern 4
    [0, 0, 0, 1, 0,
     0, 0, 1, 1, 0,
     0, 0, 1, 0, 0,
     0, 1, 1, 0, 0,
     0, 1, 0, 0, 0],
    #Pattern 5
    [0, 0, 0, 0, 1,
     0, 0, 0, 1, 0,
     0, 0, 1, 0, 0,
     0, 1, 0, 0, 0,
     1, 0, 0, 0, 0],
    #Pattern 6
    [0, 0, 0, 0, 1,
     0, 0, 1, 0, 0,
     0, 0, 1, 0, 0,
     0, 0, 1, 0, 0,
     0, 0, 1, 0, 0],
]

digit_0 = [
    [0, 1, 1, 1, 0,  #Pattern 1
     0, 1, 0, 1, 0,
     0, 1, 0, 1, 0,
     0, 1, 0, 1, 0,
     0, 1, 1, 1, 0],
    #Pattern 2
    [0, 0, 1, 0, 0,
     0, 1, 0, 1, 0,
     0, 1, 0, 1, 0,
     0, 1, 0, 1, 0,
     0, 0, 1, 0, 0],
    #Pattern 3
    [0, 0, 1, 0, 0,
     0, 1, 0, 1, 0,
     0, 1, 0, 1, 0,
     0, 1, 0, 1, 0,
     0, 1, 1, 1, 0],
    #Pattern 4
    [0, 0, 1, 1, 0,
     0, 1, 0, 0, 1,
     0, 1, 0, 0, 1,
     0, 1, 0, 1, 0,
     0, 1, 1, 0, 0],
    #Pattern 5
    [1, 1, 1, 1, 1,
     1, 0, 0, 0, 1,
     1, 0, 1, 0, 1,
     1, 0, 0, 0, 1,
     1, 1, 1, 1, 1],
    #Pattern 6
    [0, 1, 1, 1, 0,
     1, 0, 0, 0, 1,
     1, 0, 0, 0, 1,
     1, 0, 0, 0, 1,
     0, 1, 1, 1, 0],
]

# Step 2: Prepare data
# Convert to numpy arrays and split into training and test sets
data_1 = np.array(digit_1)
data_0 = np.array(digit_0)
train_1 = data_1[:4]
train_0 = data_0[:4]
test_1 = data_1[4:]
test_0 = data_0[4:]

train_data = np.vstack((train_1, train_0))
train_labels = np.array([1] * 4 + [-1] * 4)
test_data = np.vstack((test_1, test_0))
test_labels = np.array([1] * 2 + [-1] * 2)

# Step 3: Initialize perceptron
def initialize_perceptron(input_dim):
    weights = np.random.rand(input_dim) * 0.01
    bias = np.random.rand() * 0.01
    return weights, bias

# Step 4: Train perceptron
def train_perceptron(train_data, train_labels, alpha=0.1, epochs=10):
    weights, bias = initialize_perceptron(train_data.shape[1])
    for epoch in range(epochs):
        total_error = 0
        for x, d in zip(train_data, train_labels):
            y = np.sign(np.dot(weights, x) + bias)
            error = d - y
            total_error += abs(error)
            weights += alpha * error * x
            bias += alpha * error
        if total_error == 0:
            break
    return weights, bias

# Step 5: Test perceptron
def test_perceptron(test_data, test_labels, weights, bias):
    predictions = [np.sign(np.dot(weights, x) + bias) for x in test_data]
    accuracy = np.mean(predictions == test_labels)
    return accuracy

# Train and test
weights, bias = train_perceptron(train_data, train_labels)
accuracy = test_perceptron(test_data, test_labels, weights, bias)
print(f"Test Accuracy: {accuracy * 100:.2f}%")
