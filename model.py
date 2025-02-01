import torch
import torch.nn as nn
import torch.nn.functional as F 

class Expert(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Expert, self).__init__()  # Properly call the parent constructor
        self.fc1 = nn.Linear(input_dim, output_dim)
        self.fc2 = nn.Linear(output_dim, output_dim)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

class MoELayer(nn.Module):
    def __init__(self, input_dim, output_dim, num_experts, top_k=1):
        """
        Args:
            input_dim (int): Dimension of the input features.
            output_dim (int): Dimension of the output features.
            num_experts (int): Total number of experts.
            top_k (int): Number of experts to select per sample.
        """
        super(MoELayer, self).__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        
        # Create a module list of experts 
        self.experts = nn.ModuleList([Expert(input_dim, output_dim) for _ in range(num_experts)])
        
        # The gating network produces one logit per expert.
        self.gate = nn.Linear(input_dim, num_experts)
    
    def forward(self, x):
        # Compute gating scores and apply softmax for probabilities.
        gate_logits = self.gate(x)         # Shape: (batch_size, num_experts)
        gate_weights = F.softmax(gate_logits, dim=1)  # Still (batch_size, num_experts)
        
        # Optionally apply top-k gating: keep only top_k experts per sample.
        if self.top_k < self.num_experts:
            # Get the top_k gating weights and corresponding indices.
            topk_values, topk_indices = torch.topk(gate_weights, self.top_k, dim=1)
            
            # Create a mask that is zero everywhere, then scatter the topk values.
            mask = torch.zeros_like(gate_weights)
            mask.scatter_(1, topk_indices, topk_values)
            gate_weights = mask
        
        # Evaluate each expert on the input.
        expert_outputs = [expert(x) for expert in self.experts]  # Each has shape: (batch_size, output_dim)
        
        # Stack outputs to shape: (batch_size, num_experts, output_dim)
        expert_outputs = torch.stack(expert_outputs, dim=1)
        
        # Reshape gate weights to (batch_size, num_experts, 1) for proper broadcasting.
        gate_weights = gate_weights.unsqueeze(-1)
        
        # Compute the weighted sum of expert outputs over the experts dimension.
        output = torch.sum(gate_weights * expert_outputs, dim=1)  # Resulting shape: (batch_size, output_dim)
        return output

# Example usage:
if __name__ == "__main__":
    # Define input and output dimensions
    input_dim = 10
    output_dim = 5
    batch_size = 8

    # Create a random input tensor.
    x = torch.randn(batch_size, input_dim)

    # Create the MoE layer with 4 experts and top 2 experts used per sample.
    moe_layer = MoELayer(input_dim, output_dim, num_experts=4, top_k=2)

    # Forward pass.
    output = moe_layer(x)
    print("Output shape:", output.shape)
    print("Output:", output)
