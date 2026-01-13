"""Training loop for complex-valued networks."""

import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
loss_history = []

def train(model, dataloader, optimizer, criterion, epochs=10):
    model.train()
    model.to(device)
    running_loss = 0.0

    for epoch in range(epochs):
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets).to(device)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
    epoch_loss = running_loss / len(dataloader.dataset)
    print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}")
    loss_history.append(epoch_loss)
    running_loss = 0.0

    return loss_history