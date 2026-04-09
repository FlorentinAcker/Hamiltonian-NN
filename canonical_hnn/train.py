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

def train(model, train_loader, test_loader, h, s, n_epochs=1500, lr=1e-3, weight_decay=1e-2, verbose=True, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    hist_train ,hist_test = [], []
    for epoch in range(n_epochs):
        model.train()
        batch_loss = 0.
        for y0, y1 in train_loader:
            y0, y1 = y0.to(device, non_blocking=True), y1.to(device, non_blocking=True)
            optimizer.zero_grad()
            loss = hnn_loss(model, y0, y1, h, s)
            loss.backward()
            optimizer.step()
            batch_loss += loss.item() * len(y0)
        hist_train.append(batch_loss / len(train_loader.dataset))
        model.eval()
        test_loss = sum(
            hnn_loss(model, y0, y1, h, s).item() * len(y0)
            for y0, y1 in test_loader
        ) / len(test_loader.dataset)
        hist_test.append(test_loss)
    return model, hist_train, hist_test





          

