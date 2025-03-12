import time
import torch
import os
import numpy as np

print("Checking available PyTorch backends:")
print(f"CPU: Always available")
print(f"CUDA: {torch.cuda.is_available()}")
print(f"MPS: {torch.backends.mps.is_available()}")

# Select the best available device
device = torch.device(
    "cuda" if torch.cuda.is_available() 
    else "mps" if torch.backends.mps.is_available() 
    else "cpu"
)

print("Device:", device)

print("\n\nCurrent directory:", os.getcwd())
print(os.listdir('.'))

def cpu_intensive_benchmark(matrix_size=1000, iterations=50):
    """
    Performs intensive CPU operations:
    - Matrix multiplications
    - Prime number calculations
    - Array operations
    """
    start_time = time.time()
    
    # Matrix operations
    for _ in range(iterations):
        matrix_a = np.random.rand(matrix_size, matrix_size)
        matrix_b = np.random.rand(matrix_size, matrix_size)
        _ = np.dot(matrix_a, matrix_b)
        
        # Additional numerical computations
        for i in range(2, 50000):
            # Prime number calculation
            for j in range(2, int(i ** 0.5) + 1):
                if i % j == 0:
                    break
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    print(f"Matrix size: {matrix_size}x{matrix_size}")
    print(f"Number of iterations: {iterations}")
    print(f"Total time taken: {elapsed_time:.2f} seconds")
    print(f"Average time per iteration: {elapsed_time/iterations:.2f} seconds")
    
    return elapsed_time

print("\n\nRunning CPU benchmark...")
cpu_intensive_benchmark(matrix_size=1000, iterations=50)

def pytorch_benchmark(n_samples=10000, n_features=100, n_epochs=10):
    """
    Performs complex PyTorch operations:
    - Creates a neural network
    - Generates synthetic data
    - Trains the model
    - Makes predictions
    """
    print(f"\nRunning PyTorch benchmark on {device}...")
    start_time = time.time()

    # Define a simple neural network
    class NeuralNetwork(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.Sequential(
                torch.nn.Linear(n_features, 256),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.2),
                torch.nn.Linear(256, 128),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.2),
                torch.nn.Linear(128, 64),
                torch.nn.ReLU(),
                torch.nn.Linear(64, 1)
            )

        def forward(self, x):
            return self.layers(x)

    # Generate synthetic data
    X = torch.randn(n_samples, n_features).to(device)
    y = torch.sum(X[:, :10], dim=1, keepdim=True).to(device)  # Target is sum of first 10 features
    y += torch.randn_like(y) * 0.1  # Add some noise

    # Create and move model to device
    model = NeuralNetwork().to(device)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    batch_size = 128
    for epoch in range(n_epochs):
        total_loss = 0
        batches = 0
        
        for i in range(0, n_samples, batch_size):
            batch_X = X[i:i+batch_size]
            batch_y = y[i:i+batch_size]

            # Forward pass
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)

            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            batches += 1

        avg_loss = total_loss / batches
        if epoch % 2 == 0:
            print(f"Epoch [{epoch+1}/{n_epochs}], Loss: {avg_loss:.4f}")

    # Make predictions
    model.eval()
    with torch.no_grad():
        test_X = torch.randn(1000, n_features).to(device)
        predictions = model(test_X)

    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"\nPyTorch Benchmark Results:")
    print(f"Samples: {n_samples}, Features: {n_features}, Epochs: {n_epochs}")
    print(f"Total time taken: {elapsed_time:.2f} seconds")
    print(f"Average time per epoch: {elapsed_time/n_epochs:.2f} seconds")
    
    return elapsed_time

print("\n\nRunning PyTorch benchmark...")
pytorch_benchmark(n_samples=10000, n_features=100, n_epochs=10)