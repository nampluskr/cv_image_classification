import os
import gzip
import numpy as np

#################################################################
# Dataset
#################################################################
def load_images(data_dir, split="train"):
    filename = "train-images-idx3-ubyte.gz" if split == "train" else "t10k-images-idx3-ubyte.gz"
    filepath = os.path.join(data_dir, filename)
    with gzip.open(filepath, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)
    return data.reshape(-1, 28, 28).copy()

def load_labels(data_dir, split="train"):
    filename = "train-labels-idx1-ubyte.gz" if split == "train" else "t10k-labels-idx1-ubyte.gz"
    filepath = os.path.join(data_dir, filename)
    with gzip.open(filepath, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=8)
    return data.copy()

def one_hot(x, num_classes):
    return np.eye(num_classes)[x]

def to_binary_label(x, positive_class=0):
    binary = (x == positive_class)
    return binary.reshape(-1, 1)

#################################################################
# Functions
#################################################################
def sigmoid(x):
    return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))

def sigmoid_grad(x):
    return x * (1 - x)

def binary_cross_entropy(preds, targets):
    # preds/targets: (N, 1)
    preds = np.clip(preds, 1e-8, 1 - 1e-8)
    return -np.mean(targets * np.log(preds) + (1 - targets) * np.log(1 - preds))

def binary_accuracy(preds, targets, threshold=0.5):
    # preds/targets: (N, 1)
    pred_label = (preds >= threshold).astype(np.float32)
    return (pred_label == targets).mean()

if __name__ == "__main__":
    print(f">> {os.path.basename(__file__)}")

    #################################################################
    # Hyperparameters
    #################################################################
    DATA_DIR = "/mnt/d/deep_learning/datasets/mnist"
    SEED = 42
    BATCH_SIZE = 64
    LEARNING_RATE = 1e-3
    NUM_EPOCHS = 10
    NUM_SAMPLES = 10

    np.random.seed(SEED)

    #################################################################
    # Data loading
    #################################################################
    x_train = load_images(DATA_DIR, "train")    # (60000, 28, 28)
    y_train = load_labels(DATA_DIR, "train")    # (60000,)
    x_test = load_images(DATA_DIR, "test")      # (10000, 28, 28)
    y_test = load_labels(DATA_DIR, "test")      # (10000,)

    #################################################################
    # Data Preprocessing
    #################################################################
    x_train = x_train.reshape(-1, 784).astype(np.float32) / 255.0           # (60000, 784)
    y_train = to_binary_label(y_train, positive_class=0).astype(np.float32) # (60000, 1)
    x_test = x_test.reshape(-1, 784).astype(np.float32) / 255.0             # (10000, 784)
    y_test = to_binary_label(y_test, positive_class=0).astype(np.float32)   # (10000, 1)

    #################################################################
    # Modeling
    #################################################################
    w1 = np.random.randn(784, 256)
    b1 = np.zeros(256)
    w2 = np.random.randn(256, 128)
    b2 = np.zeros(128)
    w3 = np.random.randn(128, 1)
    b3 = np.zeros(1)

    #################################################################
    # Training
    #################################################################
    print(f"\n>> Training:")

    for epoch in range(1, NUM_EPOCHS + 1):
        total_loss = 0
        total_acc = 0
        total_size = 0

        indices = np.random.permutation(len(x_train))

        for idx in range(0, len(x_train), BATCH_SIZE):
            x = x_train[indices[idx: idx + BATCH_SIZE]]
            y = y_train[indices[idx: idx + BATCH_SIZE]]
            batch_size = len(x)
            total_size += batch_size

            # Forward propagation
            z1 = np.dot(x, w1) + b1                     # (N, 256)
            a1 = sigmoid(z1)                            # (N, 256)
            z2 = np.dot(a1, w2) + b2                    # (N, 128)
            a2 = sigmoid(z2)                            # (N, 128)
            z3 = np.dot(a2, w3) + b3                    # (N, 1)
            preds = sigmoid(z3)                         # (N, 1)

            loss = binary_cross_entropy(preds, y)              # (N, 1), (N, 1)
            acc = binary_accuracy(preds, y)                    # (N, 1), (N, 1)

            # Backward propagation (manual)
            dout = (preds - y) / batch_size             # (N, 1)
            grad_z3 = dout                              # (N, 1)
            grad_w3 = np.dot(a2.T, grad_z3)             # (128, 1)
            grad_b3 = np.sum(grad_z3, axis=0)           # (1, )

            grad_a2 = np.dot(grad_z3, w3.T)             # (N, 128)
            grad_z2 = sigmoid_grad(a2) * grad_a2        # (N, 128)
            grad_w2 = np.dot(a1.T, grad_z2)             # (256, 128)
            grad_b2 = np.sum(grad_z2, axis=0)           # (128,)

            grad_a1 = np.dot(grad_z2, w2.T)             # (N, 256)
            grad_z1 = sigmoid_grad(a1) * grad_a1        # (N, 256)
            grad_w1 = np.dot(x.T, grad_z1)              # (784, 256)
            grad_b1 = np.sum(grad_z1, axis=0)           # (256,)

            # Update weights (in-place)
            w1 -= LEARNING_RATE * grad_w1
            b1 -= LEARNING_RATE * grad_b1
            w2 -= LEARNING_RATE * grad_w2
            b2 -= LEARNING_RATE * grad_b2
            w3 -= LEARNING_RATE * grad_w3
            b3 -= LEARNING_RATE * grad_b3

            total_loss += loss * batch_size
            total_acc += acc * batch_size

        print(f"[{epoch:>2}/{NUM_EPOCHS}] "
              f"loss:{total_loss/total_size:.3f} acc:{total_acc/total_size:.3f}")

    #################################################################
    # Evaluaiton
    #################################################################
    print(f"\n>> Evaluation:")

    total_loss = 0.0
    total_acc = 0.0
    total_size = 0

    for idx in range(0, len(x_test), BATCH_SIZE):
        x = x_test[idx:idx + BATCH_SIZE]
        y = y_test[idx:idx + BATCH_SIZE]
        batch_size = x.shape[0]
        total_size += batch_size

        # Forward propagation
        z1 = np.dot(x, w1) + b1
        a1 = sigmoid(z1)
        z2 = np.dot(a1, w2) + b2
        a2 = sigmoid(z2)
        z3 = np.dot(a2, w3) + b3
        preds = sigmoid(z3)

        loss = binary_cross_entropy(preds, y)
        acc = binary_accuracy(preds, y)

        total_loss += loss * batch_size
        total_acc += acc * batch_size

    print(f"loss:{total_loss/total_size:.3f} acc:{total_acc/total_size:.3f}")

    #################################################################
    # Prediction
    #################################################################
    print(f"\n>> Prediction:")

    x = x_test[:NUM_SAMPLES]
    y = y_test[:NUM_SAMPLES]

    z1 = np.dot(x, w1) + b1
    a1 = sigmoid(z1)
    z2 = np.dot(a1, w2) + b2
    a2 = sigmoid(z2)
    z3 = np.dot(a2, w3) + b3
    preds = sigmoid(z3)

    for i in range(NUM_SAMPLES):
        print(f"Target: {int(y[i, 0])} | Prediction: {int(preds[i, 0] >= 0.5)} (Prob: {preds[i, 0]:.3f})")