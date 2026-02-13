import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from PIL import Image

# ==================== CONFIGURATION ====================
BATCH_SIZE = 64
EPOCHS = 12
LR = 1e-3

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Device: {torch.cuda.get_device_name(0)}")

device = "cpu"

# ==================== DATA LOADING ====================
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)

train_dataloader = DataLoader(training_data, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE)

classes = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

# ==================== NUMPY IMPLEMENTATION ====================

class Linear:
    """PyTorch-compatible linear layer with Kaiming initialization"""
    def __init__(self, in_features, out_features):
        # Match PyTorch's kaiming_normal_ for fan_in mode with ReLU
        std = np.sqrt(2.0 / in_features)
        self.weight = np.random.randn(in_features, out_features).astype(np.float32) * std
        self.bias = np.zeros(out_features, dtype=np.float32)
        
        self.grad_weight = np.zeros_like(self.weight)
        self.grad_bias = np.zeros_like(self.bias)
        self.input = None

    def forward(self, x):
        self.input = x
        return x @ self.weight + self.bias

    def backward(self, grad_output):
        # Gradient accumulation without batch averaging (handled in loss)
        self.grad_weight = self.input.T @ grad_output
        self.grad_bias = np.sum(grad_output, axis=0)
        return grad_output @ self.weight.T

    def step(self, lr):
        # Optimizer-agnostic parameter update
        self.weight -= lr * self.grad_weight
        self.bias -= lr * self.grad_bias


class ReLU:
    """Rectified Linear Unit activation"""
    def __init__(self):
        self.input = None

    def forward(self, x):
        self.input = x
        return np.maximum(0, x)

    def backward(self, grad_output):
        return grad_output * (self.input > 0)


class NeuralNetworkNumpy:
    """Modular network with layer composition"""
    def __init__(self):
        self.layers = [
            Linear(28*28, 512),
            ReLU(),
            Linear(512, 512),
            ReLU(),
            Linear(512, 10)
        ]

    def forward(self, x):
        x = x.reshape(x.shape[0], -1)
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, grad_output):
        grad = grad_output
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def step(self, lr):
        # Optimizer logic decoupled from layer implementation
        for layer in self.layers:
            if hasattr(layer, 'step'):
                layer.step(lr)


def log_softmax(x):
    """Numerically stable log-softmax computation"""
    x_max = np.max(x, axis=1, keepdims=True)
    shifted = x - x_max
    log_sum_exp = np.log(np.sum(np.exp(shifted), axis=1, keepdims=True))
    return shifted - log_sum_exp


def cross_entropy_loss(logits, targets):
    """
    Combines log-softmax and NLL loss with numerical stability.
    Returns mean loss and gradient (already batch-averaged).
    """
    m = targets.shape[0]
    log_probs = log_softmax(logits)
    
    # Negative log-likelihood for target classes
    loss = -np.mean(log_probs[np.arange(m), targets])
    
    # Gradient derivation: ∂L/∂z = softmax(z) - one_hot
    probs = np.exp(log_probs)  # Recover probabilities from log-space
    one_hot = np.zeros_like(probs)
    one_hot[np.arange(m), targets] = 1
    grad = (probs - one_hot) / m
    
    return loss, grad


def train_numpy(dataloader, model, lr):
    size = len(dataloader.dataset)
    train_losses = []
    
    for batch, (X, y) in enumerate(dataloader):
        X = X.numpy()
        y = y.numpy()
        
        logits = model.forward(X)
        loss, grad = cross_entropy_loss(logits, y)
        
        model.backward(grad)
        model.step(lr)
        
        train_losses.append(loss)
        
        if batch % 100 == 0:
            current = (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    
    return np.mean(train_losses)


def test_numpy(dataloader, model):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0
    
    for X, y in dataloader:
        X = X.numpy()
        y = y.numpy()
        
        pred = model.forward(X)
        loss, _ = cross_entropy_loss(pred, y)
        test_loss += loss
        correct += np.sum(np.argmax(pred, axis=1) == y)
    
    test_loss /= num_batches
    accuracy = 100 * correct / size
    print(f"Test Error: \n Accuracy: {accuracy:>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return test_loss, accuracy


# ==================== PYTORCH IMPLEMENTATION ====================

class NeuralNetworkPyTorch(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )
        # Match NumPy initialization exactly
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.flatten(x)
        return self.linear_relu_stack(x)


def train_pytorch(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    train_losses = []
    
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_losses.append(loss.item())

        if batch % 100 == 0:
            current = (batch + 1) * len(X)
            print(f"loss: {loss.item():>7f}  [{current:>5d}/{size:>5d}]")
    
    return np.mean(train_losses)


def test_pytorch(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    
    test_loss /= num_batches
    accuracy = 100 * correct / size
    print(f"Test Error: \n Accuracy: {accuracy:>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return test_loss, accuracy


# ==================== TRAINING & EVALUATION ====================

print("\n" + "="*60)
print("TRAINING NUMPY MODEL")
print("="*60)

model_numpy = NeuralNetworkNumpy()
history_numpy = {'train_loss': [], 'test_loss': [], 'test_acc': []}

for t in range(EPOCHS):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loss = train_numpy(train_dataloader, model_numpy, LR)
    test_loss, test_acc = test_numpy(test_dataloader, model_numpy)
    history_numpy['train_loss'].append(train_loss)
    history_numpy['test_loss'].append(test_loss)
    history_numpy['test_acc'].append(test_acc)

print("NumPy training completed")

print("\n" + "="*60)
print("TRAINING PYTORCH MODEL")
print("="*60)

model_pytorch = NeuralNetworkPyTorch().to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model_pytorch.parameters(), lr=LR)

history_pytorch = {'train_loss': [], 'test_loss': [], 'test_acc': []}

for t in range(EPOCHS):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loss = train_pytorch(train_dataloader, model_pytorch, loss_fn, optimizer)
    test_loss, test_acc = test_pytorch(test_dataloader, model_pytorch, loss_fn)
    history_pytorch['train_loss'].append(train_loss)
    history_pytorch['test_loss'].append(test_loss)
    history_pytorch['test_acc'].append(test_acc)

print("PyTorch training completed")

# ==================== VISUALIZATION ====================

fig, axes = plt.subplots(2, 2, figsize=(15, 10))
epochs_range = range(1, EPOCHS + 1)

# Training loss comparison
axes[0, 0].plot(epochs_range, history_numpy['train_loss'], 'b-o', label='NumPy', linewidth=2)
axes[0, 0].plot(epochs_range, history_pytorch['train_loss'], 'r-s', label='PyTorch', linewidth=2)
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Train Loss')
axes[0, 0].set_title('Training Loss')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Test loss comparison
axes[0, 1].plot(epochs_range, history_numpy['test_loss'], 'b-o', label='NumPy', linewidth=2)
axes[0, 1].plot(epochs_range, history_pytorch['test_loss'], 'r-s', label='PyTorch', linewidth=2)
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Test Loss')
axes[0, 1].set_title('Test Loss')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Accuracy comparison
axes[1, 0].plot(epochs_range, history_numpy['test_acc'], 'b-o', label='NumPy', linewidth=2)
axes[1, 0].plot(epochs_range, history_pytorch['test_acc'], 'r-s', label='PyTorch', linewidth=2)
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('Accuracy (%)')
axes[1, 0].set_title('Test Accuracy')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Results summary
axes[1, 1].axis('off')
summary = f"""
FINAL RESULTS
{'='*40}
NumPy:
  Test Accuracy: {history_numpy['test_acc'][-1]:.2f}%
  Test Loss:    {history_numpy['test_loss'][-1]:.6f}

PyTorch:
  Test Accuracy: {history_pytorch['test_acc'][-1]:.2f}%
  Test Loss:    {history_pytorch['test_loss'][-1]:.6f}

Difference (NumPy - PyTorch):
  Accuracy: {history_numpy['test_acc'][-1] - history_pytorch['test_acc'][-1]:+.2f}%
  Loss:     {history_numpy['test_loss'][-1] - history_pytorch['test_loss'][-1]:+.6f}
"""
axes[1, 1].text(0.1, 0.5, summary, fontsize=11, family='monospace',
                verticalalignment='center', 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('training_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

# ==================== INFERENCE DEMONSTRATION ====================

print("\n" + "="*60)
print("INFERENCE COMPARISON (First test sample)")
print("="*60)

x, y_true = test_data[0]
actual_class = classes[y_true]

# NumPy inference
x_np = x.numpy().reshape(1, 1, 28, 28)
logits_np = model_numpy.forward(x_np)
pred_np = classes[np.argmax(logits_np, axis=1)[0]]

# PyTorch inference
model_pytorch.eval()
with torch.no_grad():
    x_pt = x.unsqueeze(0).to(device)
    logits_pt = model_pytorch(x_pt)
    pred_pt = classes[logits_pt.argmax(1).item()]

print(f"Actual label:    {actual_class}")
print(f"NumPy prediction: {pred_np}")
print(f"PyTorch prediction: {pred_pt}")

# Save PyTorch model
torch.save(model_pytorch.state_dict(), "fashion_mnist_model.pth")
print("\nPyTorch model saved to fashion_mnist_model.pth")

# External image prediction (if available)
def predict_external_image(model, image_path):
    try:
        img = Image.open(image_path).convert('L').resize((28, 28))
        img_array = (np.array(img, dtype=np.float32) / 255.0)
        img_array = 1.0 - img_array  # MNIST-style inversion
        img_array = img_array.reshape(1, 1, 28, 28)
        logits = model.forward(img_array)
        return classes[np.argmax(logits, axis=1)[0]]
    except FileNotFoundError:
        return None

print("\n" + "="*60)
print("EXTERNAL IMAGE PREDICTION")
print("="*60)
result = predict_external_image(model_numpy, "bag.jpg")
if result:
    print(f"Predicted class for bag.jpg: {result}")
else:
    print("bag.jpg not found - skipping external prediction")