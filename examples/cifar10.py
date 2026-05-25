
import torch
import torch.nn as nn
import torch.optim as optim
import time
import sys
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader

torch.manual_seed(42)
torch.cuda.manual_seed(42)


# ─── Config ───
BSZ        = 128
HIDDEN     = 2048
NUM_CLASSES = 10
LR         = 1e-3
N_ROUNDS   = 50

MEAN = (0.4914, 0.4822, 0.4465)
STD  = (0.2470, 0.2435, 0.2616)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device} | {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
print(f"Batch size: {BSZ} | Hidden dim: {HIDDEN}")

# ─── Data ───
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

print("Loading CIFAR-10 ...")

ds_train = CIFAR10(root="./data", train=True, download=True, transform=transform)
ds_test  = CIFAR10(root="./data", train=False, download=True, transform=transform)

trainloader = DataLoader(ds_train, batch_size=BSZ, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
testloader  = DataLoader(ds_test,  batch_size=BSZ, shuffle=False, num_workers=2, pin_memory=True)

print(f"Train samples: {len(ds_train)} | Test samples: {len(ds_test)}")

# ─── Model ───
class MLP3(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(3 * 32 * 32, hidden, bias=False)
        self.fc2 = nn.Linear(hidden, hidden, bias=False)
        self.fc3 = nn.Linear(hidden, NUM_CLASSES, bias=False)
        self.gelu = nn.GELU()

    def forward(self, x):
        x = self.flatten(x)
        x = self.gelu(self.fc1(x))
        x = self.gelu(self.fc2(x))
        return self.fc3(x)

def build_model(hidden):
    m = MLP3(hidden).to(device)
    print(f"Model Params: {sum(p.numel() for p in m.parameters()):,}")
    return m

def evaluate(model, loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            correct += (logits.argmax(1) == y).sum().item()
            total += y.size(0)
    return 100.0 * correct / total

from i8muon import Muon
# ─── Train ───
def run(hidden):
    model = build_model(hidden)
    # opt = optim.Adam(model.parameters(), lr=LR, weight_decay=0.01)
    opt = Muon(
        model.parameters(), 
        lr=0.001, 
        momentum=(0.95, 0.95), 
        weight_decay=0.01,
        precision="INT8",
        autotune=False,
        use_gram=False,
        use_cuda_graph=True
    )
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=N_ROUNDS)
    crit = nn.CrossEntropyLoss()

    print(f"\n{'Round':>5} {'Loss':>8} {'Train%':>7} {'Test%':>7} {'Time':>6}")
    print("-" * 40)

    for rnd in range(1, N_ROUNDS + 1):
        t0 = time.time()
        
        # --- Training Phase ---
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for x, y in trainloader:
            x, y = x.to(device), y.to(device)
            
            opt.zero_grad()
            logits = model(x)
            loss = crit(logits, y)
            loss.backward()
            opt.step()
            
            running_loss += loss.item() * y.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            total += y.size(0)

        # Update Scheduler after each epoch
        sched.step()
        
        train_acc = 100.0 * correct / total
        avg_loss  = running_loss / total
        test_acc  = evaluate(model, testloader)
        dt = time.time() - t0

        print(f"{rnd:>5d} {avg_loss:>8.4f} {train_acc:>6.2f}% {test_acc:>6.2f}% {dt:>5.1f}s")

    return test_acc

# ─── Main ───
if __name__ == "__main__":
    try:
        final_acc = run(HIDDEN)
        print(f"\nFinal Test Accuracy: {final_acc:.2f}%")
    except Exception as e:
        print(f"Error occurred: {e}")
        sys.exit(1)