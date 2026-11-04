"""CIFAR-10 with a three-layer MLP and Muon."""
import time

import torch
from torch import nn
from torch.utils.data import DataLoader

from i8muon import Muon

BATCH_SIZE = 128
HIDDEN = 2048
EPOCHS = 50
LR = 1e-3


class MLP3(nn.Module):
    def __init__(self, hidden=HIDDEN):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(3 * 32 * 32, hidden, bias=False)
        self.fc2 = nn.Linear(hidden, hidden, bias=False)
        self.fc3 = nn.Linear(hidden, 10, bias=False)
        self.gelu = nn.GELU()

    def forward(self, x):
        x = self.flatten(x)
        x = self.gelu(self.fc1(x))
        x = self.gelu(self.fc2(x))
        return self.fc3(x)


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    device = next(model.parameters()).device
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        correct += (model(x).argmax(1) == y).sum().item()
        total += y.size(0)
    return 100.0 * correct / total


def main():
    from torchvision import transforms
    from torchvision.datasets import CIFAR10

    torch.manual_seed(42)
    model = MLP3().cuda()
    optimizer = Muon(model.parameters(), lr=LR, weight_decay=0.01,
                     precision="int8", use_gram=False, use_cuda_graph=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    train = CIFAR10("data", train=True, download=True, transform=transform)
    test = CIFAR10("data", train=False, download=True, transform=transform)
    train_loader = DataLoader(train, BATCH_SIZE, shuffle=True, num_workers=2,
                              pin_memory=True, drop_last=True)
    test_loader = DataLoader(test, BATCH_SIZE, num_workers=2, pin_memory=True)
    criterion = nn.CrossEntropyLoss()

    print(f"{'Epoch':>5} {'Loss':>8} {'Train%':>7} {'Test%':>7} {'Time(s)':>8}")
    for epoch in range(1, EPOCHS + 1):
        started = time.perf_counter()
        model.train()
        loss_sum, correct, total = 0., 0, 0
        for x, y in train_loader:
            x, y = x.cuda(), y.cuda()
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * y.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            total += y.size(0)
        scheduler.step()
        accuracy = evaluate(model, test_loader)
        print(f"{epoch:5d} {loss_sum / total:8.4f} {100. * correct / total:7.2f} "
              f"{accuracy:7.2f} {time.perf_counter() - started:8.1f}")


if __name__ == "__main__":
    main()
