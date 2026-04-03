import torch
import torch.nn as nn
import torch.nn.functional as F 

def symplectic_form(n):
    I = torch.eye(n)
    Z = torch.zeros(n, n)
    top = torch.cat([Z, I], dim=1)
    bottom = torch.cat([-I, Z], dim=1)
    return torch.cat([top, bottom], dim=0)

class MLP(nn.Module):
    def __init__(self, n, n_hidden):
        super().__init__()
        self.operations = nn.Sequential(
            nn.Linear(2*n, n_hidden),
            nn.Tanh(),
            nn.Linear(n_hidden, n_hidden),
            nn.Tanh(),
            nn.Linear(n_hidden, n_hidden),
            nn.Tanh(),
            nn.Linear(n_hidden, 1),
        )
    
    def forward(self, x):
        return self.operations(x)

class HNN(nn.Module):
    def __init__(self, n, n_hidden):
        super().__init__()
        self.model = MLP(n, n_hidden)
        J = symplectic_form(n)
        self.register_buffer("J", J)
    
    def forward(self, x):
        return self.model(x)
    
    def derivative(self, x):
        x = x.detach().requires_grad_(True)
        H = self.forward(x).squeeze(-1)
        gradH = torch.autograd.grad(H.sum(), x, create_graph=True)[0]
        vector_field = torch.einsum('ij, aj -> ai', self.J.T, gradH)

        return vector_field



