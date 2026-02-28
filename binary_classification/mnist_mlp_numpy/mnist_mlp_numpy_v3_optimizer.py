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

#################################################################
# Modules
#################################################################
class Module:
    def __init__(self):
        self.params = []
        self.grads = []

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        raise NotImplementedError

class Linear(Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.w = np.random.randn(in_features, out_features)
        self.b = np.zeros(out_features)
        self.grad_w = np.zeros_like(self.w)
        self.grad_b = np.zeros_like(self.b)

        self.params.extend([self.w, self.b])
        self.grads.extend([self.grad_w, self.grad_b])
        self.x = None

    def forward(self, x):
        self.x = x
        return np.dot(x, self.w) + self.b

    def backward(self, dout):
        self.grad_w[...] = np.dot(self.x.T, dout)
        self.grad_b[...] = np.sum(dout, axis=0)
        return np.dot(dout, self.w.T)

class Sigmoid(Module):
    def forward(self, x):
        self.out = sigmoid(x)
        return self.out

    def backward(self, dout):
        return dout * self.out * (1 - self.out)

class Sequential(Module):
    def __init__(self, *layers):
        super().__init__()
        self.layers = list(layers)

        for layer in self.layers:
            self.params.extend(layer.params)
            self.grads.extend(layer.grads)

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, dout):
        for layer in reversed(self.layers):
            dout = layer.backward(dout)

#####################################################################
# Optimizers
#####################################################################
class SGD:
    def __init__(self, model, lr):
        self.params = model.params
        self.grads = model.grads
        self.lr = lr

    def step(self):
        for param, grad in zip(self.params, self.grads):
            param -= self.lr * grad

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
    model = Sequential(
        Linear(784, 256),
        Sigmoid(),
        Linear(256, 128),
        Sigmoid(),
        Linear(128, 1),
    )
    optimizer = SGD(model, lr=LEARNING_RATE)

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
            logits = model(x)                           # (N, 10)
            preds = sigmoid(logits)                     # (N, 10)

            loss = binary_cross_entropy(preds, y)              # (N, 10), (N, 10)
            acc = binary_accuracy(preds, y)                    # (N, 10), (N, 10)

            # Backward propagation (manual)
            dout = (preds - y) / batch_size             # (N, 10)
            model.backward(dout)

            # Update weights (in-place)
            optimizer.step()

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
        logits = model(x)
        preds = sigmoid(logits)

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

    logits = model(x)
    preds = sigmoid(logits)

    for i in range(NUM_SAMPLES):
        print(f"Target: {int(y[i, 0])} | Prediction: {int(preds[i, 0] >= 0.5)} (Prob: {preds[i, 0]:.3f})")