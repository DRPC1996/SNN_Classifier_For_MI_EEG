import torch
import torch.nn as nn

class XORModel(nn.Module):
    def __init__(self):
        super(XORModel, self).__init__()
        # Define the layers
        self.fc1 = nn.Linear(2, 2)  # Input layer to first hidden layer
        self.fc2 = nn.Linear(2, 1)  # Second hidden layer to output layer
        
        # Define predefined weights for XOR gate
        self.fc1.weight.data = torch.tensor([[10., -10.], [10., -10.]], dtype=torch.float32)
        self.fc1.bias.data = torch.tensor([5.0, -5.0], dtype=torch.float32)
        
        self.fc2.weight.data = torch.tensor([[-10., 10.]], dtype=torch.float32)
        self.fc2.bias.data = torch.tensor([5.0], dtype=torch.float32)

    def forward(self, x):
        # Define forward pass
        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x

# Define input
input_data = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)

# Create an instance of the model
model = XORModel()

# Evaluate the model
with torch.no_grad():
    output = model(input_data)
    print("Output:")
    print(output)