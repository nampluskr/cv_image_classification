import os
import gzip
import numpy as np
import random

import torch


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benhmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)

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

#################################################################
# Functions
#################################################################
def sigmoid(x):
    return torch.where(x >= 0, 1 / (1 + torch.exp(-x)), torch.exp(x) / (1 + torch.exp(x)))

def sigmoid_grad(x):
    return x * (1 - x)

def softmax(x):
    # x: (N, num_classes)
    x_max = torch.max(x, dim=1, keepdims=True).values
    e_x = torch.exp(x - x_max)
    return e_x / torch.sum(e_x, dim=1, keepdims=True)

def cross_entropy(preds, targets):
    # preds/targets: (N, num_classes)
    probs = torch.sum(preds * targets, dim=1)
    return -torch.mean(torch.log(probs))

def accuracy(preds, targets):
    # preds/targets: (N, num_classes)
    targets = targets.argmax(dim=1)
    return (preds.argmax(dim=1) == targets).float().mean()


if __name__ == "__main__":
    print(f">> {os.path.basename(__file__)}")

    #################################################################
    # Hyperparameters
    #################################################################
    DATA_DIR = "/mnt/d/deep_learning/datasets/mnist"
    SEED = 42
    BATCH_SIZE = 64
    LEARNING_RATE = 1e-2
    NUM_EPOCHS = 10
    NUM_SAMPLES = 10

    set_seed(SEED)

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
    x_train_np = x_train.reshape(-1, 784).astype(np.float32) / 255.0
    y_train_np = one_hot(y_train, num_classes=10).astype(np.float32)
    x_test_np = x_test.reshape(-1, 784).astype(np.float32) / 255.0
    y_test_np = one_hot(y_test, num_classes=10).astype(np.float32)

    x_train = torch.from_numpy(x_train_np)          # (60000, 784)
    y_train = torch.from_numpy(y_train_np)          # (60000, 10)
    x_test = torch.from_numpy(x_test_np)            # (10000, 784)
    y_test = torch.from_numpy(y_test_np)            # (10000, 10)

    #################################################################
    # Modeling
    #################################################################
    w1 = torch.randn(784, 256)
    b1 = torch.zeros(256)
    w2 = torch.randn(256, 128)
    b2 = torch.zeros(128)
    w3 = torch.randn(128, 10)
    b3 = torch.zeros(10)

    params = [w1, b1, w2, b2, w3, b3]
    for param in params:
        param.requires_grad_(True)

    #################################################################
    # Training
    #################################################################
    print(f"\n>> Training:")

    for epoch in range(1, NUM_EPOCHS + 1):
        total_loss = 0
        total_acc = 0
        total_size = 0

        indices = torch.randperm(len(x_train))

        for idx in range(0, len(x_train), BATCH_SIZE):
            x = x_train[indices[idx:idx + BATCH_SIZE]]  # (N, 784)
            y = y_train[indices[idx:idx + BATCH_SIZE]]  # (N, 10)
            batch_size = x.size(0)
            total_size += batch_size

            # Forward propagation
            z1 = torch.matmul(x, w1) + b1               # (N, 256)
            a1 = sigmoid(z1)                            # (N, 256)
            z2 = torch.matmul(a1, w2) + b2              # (N, 128)
            a2 = sigmoid(z2)                            # (N, 128)
            z3 = torch.matmul(a2, w3) + b3              # (N, 10)
            preds = softmax(z3)                         # (N, 10)

            loss = cross_entropy(preds, y)              # (N, 10), (N, 10)
            acc = accuracy(preds, y)                    # (N, 10), (N, 10)

            # Backward propagation (autograd)
            loss.backward()

            # Update weights (no grad)
            with torch.no_grad():
                for param in params:
                    param -= LEARNING_RATE * param.grad
                    param.grad.zero_()

            total_loss += loss.item() * batch_size
            total_acc += acc.item() * batch_size

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
        batch_size = x.size(0)
        total_size += batch_size

        # Forward propagation
        z1 = torch.matmul(x, w1) + b1
        a1 = sigmoid(z1)
        z2 = torch.matmul(a1, w2) + b2
        a2 = sigmoid(z2)
        z3 = torch.matmul(a2, w3) + b3
        preds = softmax(z3)

        loss = cross_entropy(preds, y)
        acc = accuracy(preds, y)

        total_loss += loss.item() * batch_size
        total_acc += acc.item() * batch_size

    print(f"loss:{total_loss/total_size:.3f} acc:{total_acc/total_size:.3f}")

    #################################################################
    # Prediction
    #################################################################
    print(f"\n>> Prediction:")

    x = x_test[:NUM_SAMPLES]
    y = y_test[:NUM_SAMPLES]

    z1 = torch.matmul(x, w1) + b1
    a1 = sigmoid(z1)
    z2 = torch.matmul(a1, w2) + b2
    a2 = sigmoid(z2)
    z3 = torch.matmul(a2, w3) + b3
    preds = softmax(z3)

    for i in range(NUM_SAMPLES):
        print(f"Target: {y[i].argmax()} | Prediction: {preds[i].argmax()}")