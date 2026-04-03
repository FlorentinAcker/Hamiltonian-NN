import torch

def l2_loss(u, v):
    return (u-v).pow(2).mean()

def s_forward_euler(y0, y1):
    return y0 

def s_symplectic_euler(y0, y1):
    q0, p0 = y0[:, :2], y0[:, 2:]
    q1, p1 = y1[:, :2], y1[:, 2:]
    return torch.cat([q0, p1], dim=1) 

def s_midpoint(y0, y1):
    return (y0 + y1) / 2 

def finite_differences(y0, y1, h):
    return (y1 - y0) / h

def hnn_loss(model, y0, y1, h, s):
    dxdt_hat = finite_differences(y0, y1, h)
    y_eval = s(y0, y1)
    dxdt = model.derivative(y_eval)
    return l2_loss(dxdt_hat, dxdt)

def train(model, y0, y1, h, s, n_epochs, lr, weight_decay, verbose=True):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    hist = []
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        loss = hnn_loss(model, y0, y1, h, s)
        loss.backward()
        optimizer.step()
        hist.append(loss.item())
        if verbose and (epoch + 1) % 100 == 0:
            print(f"epoch {epoch+1} / {n_epochs}, loss: {loss.item()}")
    return hist, model



          

